import Foundation

/// Picks the harness for an agent: its own `harness:` command if it has one, else the built-in loop.
public struct HarnessRouter: Harness {
    public let builtIn: Harness
    public let external: ExternalHarness
    public init(builtIn: Harness, external: ExternalHarness) { self.builtIn = builtIn; self.external = external }

    public func run(_ input: TurnInput, context: TurnContext, emit: @Sendable @escaping (TurnEvent) -> Void) async throws {
        if input.agent.harness != nil { try await external.run(input, context: context, emit: emit) }
        else { try await builtIn.run(input, context: context, emit: emit) }
    }
}

/// Lines from a harness's stdout, handed to whoever is waiting. Cancellation wakes the waiter.
final class LineQueue: @unchecked Sendable {
    enum Item { case line(JSON), eof, cancelled }
    private let lock = NSLock()
    private var items: [JSON] = []
    private var waiter: CheckedContinuation<Item, Never>?
    private var closed = false
    private var partial = Data()

    func feed(_ data: Data) {
        lock.lock()
        partial.append(data)
        var ready: [JSON] = []
        while let nl = partial.firstIndex(of: 10) {
            let line = partial[partial.startIndex..<nl]
            partial = Data(partial[(nl + 1)...])
            if !line.isEmpty, let j = try? JSONDecoder().decode(JSON.self, from: line) { ready.append(j) }
        }
        var wake: (CheckedContinuation<Item, Never>, JSON)?
        for j in ready {
            if let w = waiter, wake == nil { waiter = nil; wake = (w, j) } else { items.append(j) }
        }
        lock.unlock()
        if let (w, j) = wake { w.resume(returning: .line(j)) }
    }

    func close() {
        lock.lock(); closed = true
        let w = waiter; waiter = nil
        lock.unlock()
        w?.resume(returning: .eof)
    }

    var isClosed: Bool { lock.lock(); defer { lock.unlock() }; return closed }

    func next() async -> Item {
        await withTaskCancellationHandler {
            await withCheckedContinuation { (k: CheckedContinuation<Item, Never>) in
                lock.lock()
                if !items.isEmpty { let j = items.removeFirst(); lock.unlock(); k.resume(returning: .line(j)) }
                else if closed { lock.unlock(); k.resume(returning: .eof) }
                else if Task.isCancelled { lock.unlock(); k.resume(returning: .cancelled) }
                else { waiter = k; lock.unlock() }
            }
        } onCancel: {
            lock.lock(); let w = waiter; waiter = nil; lock.unlock()
            w?.resume(returning: .cancelled)
        }
    }
}

/// One running harness process: a child speaking JSON lines on stdin and stdout.
final class HarnessProcess: @unchecked Sendable {
    let command: String
    let agentID: String
    let lines = LineQueue()
    private let process = Process()
    private let stdin = Pipe()
    private let stdout = Pipe()
    private let writeLock = NSLock()
    private let turnLock = TurnLock()

    var isAlive: Bool { process.isRunning && !lines.isClosed }
    var pid: Int32? { process.isRunning ? process.processIdentifier : nil }

    init(agent: Agent, command: String, cwd: URL, environment: [String: String], home: ParlorHome) throws {
        self.command = command; self.agentID = agent.id
        try FileManager.default.createDirectory(at: cwd, withIntermediateDirectories: true)
        let log = ServiceLog.open(home: home, service: "harness:" + agent.id, banner: "starting: \(command)")
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-l", "-c", "exec " + command]
        process.currentDirectoryURL = cwd
        var env = ProcessInfo.processInfo.environment
        for (k, v) in environment { env[k] = v }
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = log
        let q = lines
        stdout.fileHandleForReading.readabilityHandler = { h in
            let d = h.availableData
            if d.isEmpty { h.readabilityHandler = nil; q.close() } else { q.feed(d) }
        }
        process.terminationHandler = { p in
            try? log?.write(contentsOf: Data("--- exited \(p.terminationStatus)\n".utf8))
            // Give the reader a moment to drain what the child printed last.
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.3) { q.close() }
        }
        try process.run()
    }

    func send(_ j: JSON) {
        writeLock.lock(); defer { writeLock.unlock() }
        guard process.isRunning else { return }
        try? stdin.fileHandleForWriting.write(contentsOf: Data((j.text + "\n").utf8))
    }

    func acquire() async { await turnLock.acquire() }
    func release() { Task { await turnLock.release() } }

    func stop() {
        stdout.fileHandleForReading.readabilityHandler = nil
        try? stdin.fileHandleForWriting.close()
        if process.isRunning {
            process.terminate()
            let p = process
            DispatchQueue.global().asyncAfter(deadline: .now() + 2) { if p.isRunning { kill(p.processIdentifier, SIGKILL) } }
        }
        lines.close()
    }
}

/// One turn at a time per process.
actor TurnLock {
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func acquire() async {
        if !busy { busy = true; return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func release() {
        if waiters.isEmpty { busy = false } else { waiters.removeFirst().resume() }
    }
}

/// Runs a turn in an outside process over the JSON-lines protocol of SPEC §2.5. Tool calls come back to the app
/// and run through the same `ToolExecutor` (allowlist and tier included); the log looks the same as with the
/// built-in harness.
public final class ExternalHarness: Harness, @unchecked Sendable {
    public let home: ParlorHome
    /// Environment for a launched harness: runtime API URL and token, parlor MCP URL, agent id.
    private let environment: @Sendable (Agent) async -> [String: String]
    private let lock = NSLock()
    private var processes: [String: HarnessProcess] = [:]

    public init(home: ParlorHome, environment: @escaping @Sendable (Agent) async -> [String: String]) {
        self.home = home; self.environment = environment
    }

    public func workspace(for agentID: String) -> URL { home.root.appendingPathComponent("workspaces/\(agentID)") }

    private func process(for agent: Agent, command: String) async throws -> HarnessProcess {
        lock.lock()
        if let p = processes[agent.id], p.command == command, p.isAlive { lock.unlock(); return p }
        let stale = processes[agent.id]
        processes[agent.id] = nil
        lock.unlock()
        stale?.stop()
        let env = await environment(agent)
        let p: HarnessProcess
        do { p = try HarnessProcess(agent: agent, command: command, cwd: workspace(for: agent.id), environment: env, home: home) }
        catch { throw ParlorError.message("Could not start the harness for \(agent.name): \(error.localizedDescription)") }
        lock.lock(); processes[agent.id] = p; lock.unlock()
        return p
    }

    public func stop(agent id: String) {
        lock.lock(); let p = processes[id]; processes[id] = nil; lock.unlock()
        p?.stop()
    }

    public func stopAll() {
        lock.lock(); let all = Array(processes.values); processes = [:]; lock.unlock()
        for p in all { p.stop() }
    }

    public func pid(agent id: String) -> Int32? { lock.lock(); defer { lock.unlock() }; return processes[id]?.pid }
    public func running() -> [String] { lock.lock(); defer { lock.unlock() }; return processes.filter { $0.value.isAlive }.keys.sorted() }

    // MARK: the turn

    public func run(_ input: TurnInput, context: TurnContext, emit: @Sendable @escaping (TurnEvent) -> Void) async throws {
        guard let command = input.agent.harness else { throw ParlorError.message("\(input.agent.name) has no harness command.") }
        let proc = try await process(for: input.agent, command: command)
        await proc.acquire()
        defer { proc.release() }

        let turn = UUID().uuidString
        proc.send(Self.turnStart(input, turn: turn))
        emit(.status("Thinking"))

        var announced = Set<String>()
        var pending: String?
        func finishPending(_ why: String) {
            if let id = pending {
                proc.send(.object(["type": .string("tool.result"), "turn": .string(turn), "callID": .string(id), "text": .string(why), "isError": .bool(true)]))
                pending = nil
            }
        }
        while true {
            switch await proc.lines.next() {
            case .cancelled:
                proc.send(.object(["type": .string("turn.cancel"), "turn": .string(turn)]))
                finishPending("cancelled")
                throw CancellationError()
            case .eof:
                let tail = ServiceLog.tail(home: home, service: "harness:" + input.agent.id, lines: 6)
                stop(agent: input.agent.id)
                throw ParlorError.message("The harness for \(input.agent.name) exited mid-turn." + (tail.isEmpty ? "" : "\n" + tail))
            case .line(let msg):
                guard let o = msg.objectValue, o["turn"]?.stringValue == turn, let type = o["type"]?.stringValue else { continue }
                switch type {
                case "text.delta": emit(.delta(o["text"]?.stringValue ?? ""))
                case "status": emit(.status(o["text"]?.stringValue ?? ""))
                case "stats":
                    emit(.stats(LLMStats(promptTokens: o["promptTokens"]?.intValue ?? 0, completionTokens: o["completionTokens"]?.intValue ?? 0,
                                         tokensPerSecond: Self.double(o["tokensPerSecond"]) ?? 0, cachedTokens: o["cachedTokens"]?.intValue ?? 0)))
                case "message":
                    guard (o["role"]?.stringValue ?? "assistant") == "assistant" else { continue }
                    let calls = Self.calls(o["toolCalls"])
                    for c in calls { announced.insert(c.id) }
                    emit(.message(ChatMessage(role: .assistant, text: o["text"]?.stringValue ?? "", toolCalls: calls.isEmpty ? nil : calls)))
                case "tool.call":
                    guard let name = o["name"]?.stringValue else { continue }
                    let id = o["callID"]?.stringValue ?? UUID().uuidString
                    let call = ToolCallRecord(id: id, name: name, arguments: o["arguments"] ?? .object([:]))
                    if !announced.contains(id) {
                        announced.insert(id)
                        emit(.message(ChatMessage(role: .assistant, text: "", toolCalls: [call])))
                    }
                    emit(.toolStarted(call))
                    pending = id
                    let started = Date()
                    let out = await context.tools.call(call, agent: input.agent, settings: input.settings)
                    let secs = Date().timeIntervalSince(started)
                    pending = nil
                    emit(.message(ChatMessage(role: .tool, text: out.text, toolCallID: id, toolName: name, isError: out.isError, seconds: secs,
                                              images: out.images.isEmpty ? nil : out.images)))
                    proc.send(.object(["type": .string("tool.result"), "turn": .string(turn), "callID": .string(id), "text": .string(out.text),
                                       "isError": .bool(out.isError), "images": .array(out.images.map { .string($0) })]))
                    if Task.isCancelled { proc.send(.object(["type": .string("turn.cancel"), "turn": .string(turn)])); throw CancellationError() }
                case "turn.done": return
                case "error": throw ParlorError.message(o["message"]?.stringValue ?? "The harness reported an error.")
                default: continue
                }
            }
        }
    }

    // MARK: wire

    static func double(_ j: JSON?) -> Double? {
        switch j { case .double(let d)?: return d; case .int(let i)?: return Double(i); default: return nil }
    }

    static func calls(_ j: JSON?) -> [ToolCallRecord] {
        guard case .array(let a)? = j else { return [] }
        return a.compactMap { c in
            guard let o = c.objectValue, let name = o["name"]?.stringValue else { return nil }
            return ToolCallRecord(id: o["id"]?.stringValue ?? UUID().uuidString, name: name, arguments: o["arguments"] ?? .object([:]))
        }
    }

    static func turnStart(_ input: TurnInput, turn: String) -> JSON {
        let a = input.agent
        var agent: [String: JSON] = [
            "id": .string(a.id), "name": .string(a.name), "description": .string(a.summary), "kind": .string(a.kind.rawValue),
            "model": .string(a.model), "tools": .array(a.tools.map { .string($0) }), "tier": .string(a.tier.rawValue),
            "context": .int(a.contextTokens), "compact_at": .double(a.compactAt), "temperature": .double(a.temperature),
            "thinking": .bool(a.thinking), "persona": .string(a.persona)]
        if let m = a.imageModel { agent["image_model"] = .string(m) }
        for f in a.extra { if case .scalar(let s) = f.value { agent[f.key] = .string(s) } }
        let transcript: [JSON] = input.transcript.compactMap { m in
            guard let d = try? ChatLog.encoder.encode(m) else { return nil }
            return try? JSONDecoder().decode(JSON.self, from: d)
        }
        let s = input.settings
        var settings: [String: JSON] = [
            "temperature": .double(s.temperature), "top_p": .double(s.topP), "thinking": .bool(s.thinking),
            "context": .int(s.contextTokens), "compact_at": .double(s.compactAt), "keep_recent": .int(s.keepRecent),
            "summary_tokens": .int(s.summaryTokens), "max_steps": .int(s.maxSteps),
            "tool_result_chars": .int(s.toolResultChars), "tool_timeout": .int(s.toolTimeoutSeconds)]
        if let r = s.repetitionPenalty { settings["repetition_penalty"] = .double(r) }
        if let m = s.maxReplyTokens { settings["max_reply_tokens"] = .int(m) }
        return .object([
            "type": .string("turn.start"), "version": .int(1), "turn": .string(turn), "conversation": .string(input.conversationID),
            "agent": .object(agent), "participants": .array(input.participants.map { p in
                p.isUser ? .object(["user": .bool(true)]) : .object(["agent": .string(p.agent ?? ""), "name": .string(input.names[p.agent ?? ""] ?? p.agent ?? "")])
            }),
            "transcript": .array(transcript),
            "tools": .array(input.tools.map { .object(["name": .string($0.wireName), "description": .string($0.description), "schema": $0.schema]) }),
            "settings": .object(settings), "memory": .string(input.memory)])
    }
}

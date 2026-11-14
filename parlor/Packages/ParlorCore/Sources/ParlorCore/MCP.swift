import Foundation
import MCP
import System

/// One entry of `mcp.json` (Claude Desktop format).
public struct MCPServerConfig: Codable, Sendable, Equatable {
    public var command: String?
    public var args: [String]?
    public var env: [String: String]?
    public var cwd: String?
    public var url: String?
    public var disabled: Bool?
    /// `always` (default) starts with the app; `onDemand` starts on first use and stops after `idleMinutes`.
    public var lifecycle: String?
    public var idleMinutes: Int?

    public init(command: String? = nil, args: [String]? = nil, env: [String: String]? = nil,
                cwd: String? = nil, url: String? = nil, disabled: Bool? = nil,
                lifecycle: String? = nil, idleMinutes: Int? = nil) {
        self.command = command; self.args = args; self.env = env
        self.cwd = cwd; self.url = url; self.disabled = disabled
        self.lifecycle = lifecycle; self.idleMinutes = idleMinutes
    }

    public var isOnDemand: Bool { lifecycle == "onDemand" }
}

public struct MCPConfigFile: Codable, Sendable, Equatable {
    public var mcpServers: [String: MCPServerConfig]
    public init(mcpServers: [String: MCPServerConfig] = [:]) { self.mcpServers = mcpServers }

    public static func load(_ url: URL) -> MCPConfigFile {
        guard let d = try? Data(contentsOf: url), let c = try? JSONDecoder().decode(MCPConfigFile.self, from: d) else {
            return MCPConfigFile()
        }
        return c
    }

    public func save(_ url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try enc.encode(self).write(to: url)
    }
}

/// Resolves exactly once, from whichever of connect, timeout or process exit gets there first.
final class StartGate: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<Void, Error>?
    private var waiter: CheckedContinuation<Void, Error>?

    func finish(_ r: Result<Void, Error>) {
        lock.lock()
        guard result == nil else { lock.unlock(); return }
        result = r
        let w = waiter; waiter = nil
        lock.unlock()
        w?.resume(with: r)
    }

    func wait() async throws {
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
            lock.lock()
            if let r = result { lock.unlock(); c.resume(with: r) } else { waiter = c; lock.unlock() }
        }
    }
}

/// A connection to one MCP server (stdio child process or streamable HTTP).
public actor MCPServerConnection: ToolProvider {
    public nonisolated let id: String
    private let config: MCPServerConfig
    private let home: ParlorHome
    private var client: Client?
    private var process: Process?
    private var logHandle: FileHandle?
    private var stopping = false
    private var startGate: StartGate?
    private weak var hub: ToolHub?
    /// Called when the child process ends without `stop()` having been asked for.
    private let onExit: (@Sendable (MCPServerConnection, Int32) -> Void)?
    private let onUse: (@Sendable () -> Void)?
    /// Log name; defaults to `mcp:<id>`.
    private let logName: String

    public init(id: String, config: MCPServerConfig, home: ParlorHome, hub: ToolHub?, logName: String? = nil,
                onExit: (@Sendable (MCPServerConnection, Int32) -> Void)? = nil, onUse: (@Sendable () -> Void)? = nil) {
        self.id = id; self.config = config; self.home = home; self.hub = hub
        self.logName = logName ?? "mcp:\(id)"; self.onExit = onExit; self.onUse = onUse
    }

    public var pid: Int32? { process?.isRunning == true ? process?.processIdentifier : nil }

    /// Connect with a deadline so a hung or crashing server cannot block the app. Also fails at once when the
    /// process exits before it is ready (a bad command, a missing interpreter).
    public func start(timeout: TimeInterval = 25) async throws {
        let gate = StartGate()
        startGate = gate
        defer { startGate = nil }
        let connecting = Task { try await self.connect() }
        let timer = Task {
            try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
            gate.finish(.failure(ParlorError.message("Timed out waiting for \(self.id) to start. Check the command in mcp.json.")))
        }
        Task {
            do { try await connecting.value; gate.finish(.success(())) } catch { gate.finish(.failure(error)) }
        }
        do { try await gate.wait() } catch {
            timer.cancel(); connecting.cancel()
            await stop()
            throw error
        }
        timer.cancel()
    }

    private func connect() async throws {
        stopping = false
        let client = Client(name: "parlor", version: "0.1.0")
        if let urlString = config.url, let url = URL(string: urlString) {
            _ = try await client.connect(transport: HTTPClientTransport(endpoint: url))
        } else if let command = config.command {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/bin/zsh")
            // Login shell so Homebrew/uv/npm paths resolve the way they do in Terminal.
            proc.arguments = ["-l", "-c", "exec \"$0\" \"$@\"", command] + (config.args ?? [])
            var env = ProcessInfo.processInfo.environment
            let extra = "/opt/homebrew/bin:/usr/local/bin:\(NSHomeDirectory())/.local/bin:\(NSHomeDirectory())/.cargo/bin"
            env["PATH"] = extra + ":" + (env["PATH"] ?? "/usr/bin:/bin")
            for (k, v) in config.env ?? [:] { env[k] = v }
            proc.environment = env
            if let cwd = config.cwd {
                proc.currentDirectoryURL = URL(fileURLWithPath: (cwd as NSString).expandingTildeInPath)
            } else {
                proc.currentDirectoryURL = home.root
            }
            let stdin = Pipe(), stdout = Pipe()
            proc.standardInput = stdin
            proc.standardOutput = stdout
            logHandle = ServiceLog.open(home: home, service: logName, banner: "start: " + ([command] + (config.args ?? [])).joined(separator: " "))
            proc.standardError = logHandle ?? FileHandle.nullDevice
            let exit = onExit
            proc.terminationHandler = { [weak self] p in
                guard let self else { return }
                Task { await self.processEnded(status: p.terminationStatus, callback: exit) }
            }
            try proc.run()
            process = proc
            let transport = StdioTransport(
                input: FileDescriptor(rawValue: stdout.fileHandleForReading.fileDescriptor),
                output: FileDescriptor(rawValue: stdin.fileHandleForWriting.fileDescriptor))
            _ = try await client.connect(transport: transport)
        } else {
            throw ParlorError.message("Server \(id) needs a command or a url")
        }
        let me = id
        let hub = self.hub
        await client.onNotification(ToolListChangedNotification.self) { _ in
            await hub?.refresh(me)
        }
        self.client = client
    }

    private func processEnded(status: Int32, callback: (@Sendable (MCPServerConnection, Int32) -> Void)?) {
        let wasStopping = stopping
        if let h = logHandle { try? h.write(contentsOf: Data("--- exited with status \(status)\n".utf8)); try? h.close() }
        logHandle = nil
        if let gate = startGate {
            // Died before it was ready: report why instead of waiting out the deadline.
            let tail = ServiceLog.tail(home: home, service: logName, lines: 4)
            gate.finish(.failure(ParlorError.message("\(id) exited with status \(status) before it was ready." + (tail.isEmpty ? "" : "\n" + tail))))
            return
        }
        if !wasStopping { callback?(self, status) }
    }

    public func stop() async {
        stopping = true
        await client?.disconnect()
        client = nil
        if let p = process, p.isRunning { p.terminate() }
        process = nil
    }

    public func listTools() async throws -> [ToolDef] {
        guard let client else { throw ParlorError.message("Server \(id) is not connected") }
        var out: [ToolDef] = []
        var cursor: String?
        repeat {
            let page = try await client.listTools(cursor: cursor)
            for t in page.tools {
                let schema = (try? JSONEncoder().encode(t.inputSchema)).flatMap { try? JSONDecoder().decode(JSON.self, from: $0) } ?? .object([:])
                out.append(ToolDef(server: id, name: t.name, description: t.description ?? "", schema: schema))
            }
            cursor = page.nextCursor
        } while cursor != nil
        return out
    }

    public func call(tool: String, arguments: JSON) async throws -> ToolOutput {
        guard let client else { throw ParlorError.message("Server \(id) is not connected") }
        onUse?()
        defer { onUse?() }
        var args: [String: Value]?
        if let obj = arguments.objectValue {
            let data = try JSONEncoder().encode(JSON.object(obj))
            args = try JSONDecoder().decode([String: Value].self, from: data)
        }
        let result = try await client.callTool(name: tool, arguments: args)
        var texts: [String] = []
        var images: [String] = []
        for c in result.content {
            switch c {
            case .text(let t, _, _): texts.append(t)
            case .image(let data, let mime, _, _):
                if let path = Media.save(base64: data, mime: mime, home: home) {
                    images.append(path); texts.append("[image saved: \(path)]")
                }
            case .resource(let r, _, _):
                texts.append(r.text ?? "[resource \(r.uri)]")
            case .resourceLink(let uri, let name, _, _, _, _): texts.append("[\(name): \(uri)]")
            default: break
            }
        }
        return ToolOutput(text: texts.joined(separator: "\n"), images: images, isError: result.isError ?? false)
    }
}

enum Media {
    /// Save base64 image data under `media/YYYY-MM/` and return the home-relative path.
    static func save(base64: String, mime: String, home: ParlorHome) -> String? {
        guard let data = Data(base64Encoded: base64) else { return nil }
        let ext = mime.contains("png") ? "png" : mime.contains("jpeg") || mime.contains("jpg") ? "jpg" : "img"
        return save(data: data, ext: ext, home: home)
    }

    static func save(data: Data, ext: String, home: ParlorHome) -> String? {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM"
        let sub = "media/" + f.string(from: Date())
        let dir = home.root.appendingPathComponent(sub)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let name = UUID().uuidString.prefix(8).lowercased() + "." + ext
        do { try data.write(to: dir.appendingPathComponent(String(name))); return sub + "/" + name }
        catch { return nil }
    }
}

/// Stands in for an on-demand server that is not running: it advertises the tool list from the last run and
/// starts the real server on the first call.
actor LazyProvider: ToolProvider {
    nonisolated let id: String
    private let tools: [ToolDef]
    private let starter: @Sendable (String) async throws -> MCPServerConnection
    init(id: String, tools: [ToolDef], starter: @escaping @Sendable (String) async throws -> MCPServerConnection) {
        self.id = id; self.tools = tools; self.starter = starter
    }
    func listTools() async throws -> [ToolDef] { tools }
    func call(tool: String, arguments: JSON) async throws -> ToolOutput {
        let conn = try await starter(id)
        return try await conn.call(tool: tool, arguments: arguments)
    }
}

/// What the dashboard shows for one mount.
public struct MCPServiceState: Sendable, Equatable {
    public var id: String
    public var isModule = false
    public var config: MCPServerConfig
    public var state: ServiceState
    public var pid: Int32?
    public var lastUsed: Date?
    public var restarts: Int
}

/// Keeps the hub in sync with `mcp.json` and supervises each server: lifecycle (always or on demand with an
/// idle stop), restart on crash with backoff (1 s, 5 s, 30 s; `failed` after three crashes in ten minutes),
/// and a log per server in `logs/`.
public actor MCPManager {
    struct Entry {
        var config: MCPServerConfig
        var conn: MCPServerConnection?
        var state: ServiceState = .stopped
        var lastUsed: Date?
        var crashes: [Date] = []
        var restarts = 0
        var manuallyStopped = false
        var idleTask: Task<Void, Never>?
        var starting: Task<MCPServerConnection, Error>?
    }

    private let hub: ToolHub
    private let home: ParlorHome
    private var entries: [String: Entry] = [:]
    /// Approved modules, supplied by the engine; they run under the same supervision as `mcp.json` servers.
    private var moduleProvider: (@Sendable () async -> [String: MCPServerConfig])?
    private var moduleNames: Set<String> = []
    /// Backoff before restart n (1-based); a variable so tests can shrink it.
    public var backoff: [TimeInterval] = [1, 5, 30]
    public var crashWindow: TimeInterval = 600
    /// Seconds per `idleMinutes` unit; tests shrink it.
    public var idleUnit: TimeInterval = 60

    public init(hub: ToolHub, home: ParlorHome) { self.hub = hub; self.home = home }

    public func setModuleProvider(_ p: @escaping @Sendable () async -> [String: MCPServerConfig]) { moduleProvider = p }
    func logName(_ id: String) -> String { moduleNames.contains(id) ? "module:\(id)" : "mcp:\(id)" }

    public func setBackoff(_ b: [TimeInterval]) { backoff = b }
    public func setIdleUnit(_ u: TimeInterval) { idleUnit = u }

    // MARK: configuration

    public func reload() async {
        var file = MCPConfigFile.load(home.mcpConfig)
        let mods = await moduleProvider?() ?? [:]
        moduleNames = Set(mods.keys).subtracting(file.mcpServers.keys)
        for (k, v) in mods where file.mcpServers[k] == nil { file.mcpServers[k] = v }
        // Drop servers that were removed or changed; keep the rest running.
        for (id, e) in entries where file.mcpServers[id] == nil || file.mcpServers[id] != e.config {
            await teardown(id)
            entries[id] = nil
            await hub.unregister(id)
        }
        var launches: [String] = []
        for (id, cfg) in file.mcpServers where entries[id] == nil {
            entries[id] = Entry(config: cfg)
            if cfg.disabled == true { continue }
            if cfg.isOnDemand { await prepareOnDemand(id) } else { launches.append(id) }
        }
        // Start the rest side by side; a slow or broken server must not hold up the others.
        await withTaskGroup(of: Void.self) { group in
            for id in launches { group.addTask { await self.launch(id) } }
        }
    }

    public func snapshot() -> [MCPServiceState] {
        entries.map { id, e in
            MCPServiceState(id: id, isModule: moduleNames.contains(id), config: e.config, state: e.config.disabled == true ? .stopped : e.state,
                            pid: nil, lastUsed: e.lastUsed, restarts: e.restarts)
        }.sorted { $0.id < $1.id }
    }

    public func pid(_ id: String) async -> Int32? { await entries[id]?.conn?.pid }

    // MARK: manual control

    public func start(_ id: String) async {
        guard entries[id] != nil else { return }
        entries[id]?.manuallyStopped = false
        entries[id]?.crashes = []
        await launch(id)
    }

    public func stop(_ id: String) async {
        guard entries[id] != nil else { return }
        entries[id]?.manuallyStopped = true
        await teardown(id)
        entries[id]?.state = .stopped
        await hub.setStatus(status(id, state: .stopped))
    }

    public func restart(_ id: String) async {
        var file = MCPConfigFile.load(home.mcpConfig)
        for (k, v) in await moduleProvider?() ?? [:] where file.mcpServers[k] == nil { file.mcpServers[k] = v }
        if let cfg = file.mcpServers[id] {
            if entries[id] == nil { entries[id] = Entry(config: cfg) } else { entries[id]?.config = cfg }
        }
        await teardown(id)
        entries[id]?.manuallyStopped = false
        entries[id]?.crashes = []
        if entries[id] != nil { await launch(id) }
    }

    public func stopAll() async { for id in Array(entries.keys) { await teardown(id) } ; entries = [:] }

    // MARK: launching

    private func status(_ id: String, state: ServerStatus.State) -> ServerStatus {
        let cfg = entries[id]?.config
        let kind = cfg?.url != nil ? "http" : "stdio"
        let detail = cfg?.url ?? ([cfg?.command ?? ""] + (cfg?.args ?? [])).joined(separator: " ")
        return ServerStatus(id: id, kind: kind, detail: detail, state: state, toolCount: 0, toolNames: [])
    }

    private func teardown(_ id: String) async {
        entries[id]?.idleTask?.cancel(); entries[id]?.idleTask = nil
        entries[id]?.starting?.cancel(); entries[id]?.starting = nil
        if let c = entries[id]?.conn { await c.stop() }
        entries[id]?.conn = nil
    }

    private func makeConnection(_ id: String) -> MCPServerConnection {
        MCPServerConnection(id: id, config: entries[id]!.config, home: home, hub: hub, logName: logName(id),
            onExit: { [weak self] conn, code in Task { await self?.exited(id, conn, code) } },
            onUse: { [weak self] in Task { await self?.touch(id) } })
    }

    /// Start a server and register it. Concurrent callers share one start.
    @discardableResult
    private func launch(_ id: String) async -> MCPServerConnection? {
        guard entries[id] != nil else { return nil }
        if let t = entries[id]?.starting { return try? await t.value }
        if let c = entries[id]?.conn { return c }
        let task = Task<MCPServerConnection, Error> { [self] in
            let conn = await self.makeConnection(id)
            try await conn.start()
            return conn
        }
        entries[id]?.starting = task
        entries[id]?.state = .starting
        await hub.setStatus(status(id, state: .connecting))
        do {
            let conn = try await task.value
            guard entries[id] != nil, entries[id]?.starting != nil else { await conn.stop(); return nil }
            entries[id]?.starting = nil
            entries[id]?.conn = conn
            entries[id]?.state = .running
            entries[id]?.lastUsed = Date()
            await hub.register(conn, status: status(id, state: .connecting))
            await cacheTools(id)
            scheduleIdle(id)
            return conn
        } catch {
            guard entries[id] != nil else { return nil }
            entries[id]?.starting = nil
            let tail = ServiceLog.tail(home: home, service: logName(id), lines: 4)
            let base = error.localizedDescription
            let msg = base + (tail.isEmpty || base.contains(tail) ? "" : "\n" + tail)
            entries[id]?.state = .failed(msg)
            await hub.setStatus(status(id, state: .failed(msg)))
            return nil
        }
    }

    // MARK: on demand

    private func cacheURL(_ id: String) -> URL { home.cacheDir.appendingPathComponent("mcp-\(id).json") }

    private func cacheTools(_ id: String) async {
        guard entries[id]?.config.isOnDemand == true, let c = entries[id]?.conn, let tools = try? await c.listTools() else { return }
        try? FileManager.default.createDirectory(at: home.cacheDir, withIntermediateDirectories: true)
        if let d = try? JSONEncoder().encode(tools) { try? d.write(to: cacheURL(id)) }
    }

    /// Register a stand-in from the cached tool list. With no cache yet, start once to learn the tools.
    private func prepareOnDemand(_ id: String) async {
        if let d = try? Data(contentsOf: cacheURL(id)), let tools = try? JSONDecoder().decode([ToolDef].self, from: d) {
            await registerLazy(id, tools: tools)
        } else {
            Task { [self] in
                await self.launch(id)
                if await self.entries[id]?.state == .running { await self.sleepOnDemand(id) }
            }
        }
    }

    private func registerLazy(_ id: String, tools: [ToolDef]) async {
        let lazy = LazyProvider(id: id, tools: tools) { [weak self] sid in
            guard let self, let c = await self.launch(sid) else { throw ParlorError.message("Server \(sid) could not start. See its log.") }
            return c
        }
        entries[id]?.state = .stopped
        await hub.register(lazy, status: status(id, state: .connecting))
    }

    /// Stop the process and put the stand-in back.
    private func sleepOnDemand(_ id: String) async {
        guard entries[id]?.config.isOnDemand == true else { return }
        let tools = (try? await entries[id]?.conn?.listTools()) ?? []
        await teardown(id)
        entries[id]?.state = .stopped
        await registerLazy(id, tools: tools.isEmpty ? await hub.allTools().filter { $0.server == id } : tools)
    }

    private func touch(_ id: String) {
        entries[id]?.lastUsed = Date()
        scheduleIdle(id)
    }

    private func scheduleIdle(_ id: String) {
        guard let e = entries[id], e.config.isOnDemand else { return }
        e.idleTask?.cancel()
        let secs = Double(e.config.idleMinutes ?? 10) * idleUnit
        entries[id]?.idleTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(secs * 1_000_000_000))
            if !Task.isCancelled { await self?.sleepOnDemand(id) }
        }
    }

    // MARK: crashes

    private func exited(_ id: String, _ conn: MCPServerConnection, _ code: Int32) async {
        guard entries[id]?.conn === conn else { return }
        entries[id]?.conn = nil
        entries[id]?.idleTask?.cancel()
        let now = Date()
        let recent = (entries[id]?.crashes ?? []).filter { now.timeIntervalSince($0) < crashWindow } + [now]
        entries[id]?.crashes = recent
        let n = recent.count
        if entries[id]?.config.isOnDemand == true {
            // Nothing is waiting on it; the next call starts it again.
            await registerLazy(id, tools: await hub.allTools().filter { $0.server == id })
            return
        }
        if n > backoff.count {
            let tail = ServiceLog.tail(home: home, service: logName(id), lines: 4)
            let msg = "Exited with status \(code) \(n) times in ten minutes; not restarting.\n" + tail
            entries[id]?.state = .failed(msg)
            await hub.setStatus(status(id, state: .failed(msg)))
            return
        }
        entries[id]?.state = .starting
        entries[id]?.restarts += 1
        await hub.setStatus(status(id, state: .connecting))
        let wait = backoff[n - 1]
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000))
            guard let self, await self.entries[id]?.manuallyStopped == false, await self.entries[id]?.conn == nil else { return }
            await self.launch(id)
        }
    }
}

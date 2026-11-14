import Foundation
import ParlorCore

func out(_ s: String) { FileHandle.standardOutput.write((s + "\n").data(using: .utf8)!) }

@main
struct CLIMain {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let cmd = args.first else { usage(); exit(2) }
        let rest = Array(args.dropFirst())
        let code = await MainActor.run { Task { @MainActor in await run(cmd, rest) } }.value
        exit(code)
    }

    static func usage() {
        out("""
        parlor-cli — drives the same engine as the app.
          models                         list installed models
          download <repo> [--image]      download a model
          agents                         list agents
          chat <agent-id> <message>      send one message, print the turn
          history <agent-id>             print the chat log
          spike                          tool-calling reliability gate
          bench [model]                  prompt-cache speed and correctness on a real model
          bench-alt [model]              two agents alternating: one cache slot vs several
          verbs                          every control the app exposes, with its tier
          call <verb> [json]             run a verb. In-process as you, or against the running app when PARLOR_TOKEN is set
          services                       the services dashboard (state, memory, last used)
          image <prompt>                 generate through the image MCP worker, showing the service states
          flow                           end-to-end flow check (agent → chat → compaction → relaunch)
        env: PARLOR_HOME, PARLOR_MODELS
        """)
    }

    @MainActor static func run(_ cmd: String, _ args: [String]) async -> Int32 {
        let engine = ParlorEngine()
        await engine.start()
        defer { Task { await engine.shutdown() } }
        switch cmd {
        case "models":
            for m in engine.library.installed { out("\(m.kind.rawValue)\t\(m.repo)\t\(byte(m.bytes))") }
            return 0
        case "agents":
            for a in engine.agents { out("\(a.id)\t\(a.kind.rawValue)\t\(a.model)\ttools=\(a.tools.joined(separator: ","))") }
            return 0
        case "download":
            guard let repo = args.first else { usage(); return 2 }
            let kind: ModelKind = args.contains("--image") ? .image : .language
            let start = Date()
            engine.library.download(repo, kind: kind)
            let printer = Task { @MainActor in
                while !Task.isCancelled {
                    if let d = engine.library.downloads[repo] { out(String(format: "  %.0f%%  %@/s", d.fraction * 100, byte(Int64(d.bytesPerSecond)))) }
                    try? await Task.sleep(nanoseconds: 5_000_000_000)
                }
            }
            await engine.library.waitForDownload(repo)
            printer.cancel()
            if let d = engine.library.downloads[repo], case .failed(let e) = d.status { out("FAILED: \(e)"); return 1 }
            out(String(format: "downloaded %@ in %.0fs", repo, Date().timeIntervalSince(start)))
            return 0
        case "history":
            guard let id = args.first else { usage(); return 2 }
            for m in engine.room(for: id).messages { out("[\(m.role.rawValue)] \(m.text)") }
            return 0
        case "chat":
            guard args.count >= 2 else { usage(); return 2 }
            return await chat(engine, args[0], args.dropFirst().joined(separator: " "))
        case "verbs":
            for v in engine.commands.sorted { out("  \(v.name.padding(toLength: 24, withPad: " ", startingAt: 0)) \(v.tier.rawValue.padding(toLength: 10, withPad: " ", startingAt: 0)) \(v.destructive ? "asks  " : "      ")\(v.summary)") }
            return 0
        case "call":
            guard let verb = args.first else { usage(); return 2 }
            let json = args.count > 1 ? args.dropFirst().joined(separator: " ") : "{}"
            guard let data = json.data(using: .utf8), let arg = try? JSONDecoder().decode(JSON.self, from: data) else { out("arguments must be JSON"); return 2 }
            if let token = ProcessInfo.processInfo.environment["PARLOR_TOKEN"] { return await callOverHTTP(verb, arg, token) }
            let r = await engine.commands.call(verb, arg, caller: .local)
            out(r.text)
            return r.isError ? 1 : 0
        case "services":
            await engine.refreshServices(); printServices(engine); return 0
        case "image": return await imageCheck(engine, args.joined(separator: " "))
        case "spike": return await Spike.run(engine)
        case "flow": return await Flow.run(engine)
        case "bench-alt": return await Bench.alternate(engine, model: args.first ?? "mlx-community/Qwen3-4B-4bit")
        case "bench": return await Bench.run(engine, model: args.first ?? "mlx-community/Qwen3-4B-4bit")
        default: usage(); return 2
        }
    }

    /// The thin form: talk to the running app's `parlor` server with a token made in System.
    static func callOverHTTP(_ verb: String, _ args: JSON, _ token: String) async -> Int32 {
        let url = URL(string: ProcessInfo.processInfo.environment["PARLOR_URL"] ?? "http://127.0.0.1:11501/mcp")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        let body: JSON = .object(["jsonrpc": .string("2.0"), "id": .int(1), "method": .string("tools/call"),
                                  "params": .object(["name": .string(verb), "arguments": args])])
        req.httpBody = try? JSONEncoder().encode(body)
        guard let (data, resp) = try? await URLSession.shared.data(for: req), let http = resp as? HTTPURLResponse else {
            out("could not reach \(url). Is the app running, and the Parlor server on?"); return 1
        }
        if http.statusCode == 401 { out("token refused (401)"); return 1 }
        guard let j = try? JSONDecoder().decode(JSON.self, from: data), let result = j.objectValue?["result"]?.objectValue else {
            out(String(data: data, encoding: .utf8) ?? "bad reply"); return 1
        }
        let text = result["content"]?.arrayValue?.first?.objectValue?["text"]?.stringValue ?? ""
        out(text)
        return result["isError"]?.boolValue == true ? 1 : 0
    }

    @MainActor static func printServices(_ engine: ParlorEngine) {
        for s in engine.services {
            let mem = s.memoryBytes.map { byte(Int64($0)) } ?? "–"
            var state = s.state.label
            if case .failed(let e) = s.state { state += " (\(e.split(separator: "\n").first.map(String.init) ?? ""))" }
            out("  \(s.id.padding(toLength: 14, withPad: " ", startingAt: 0)) \(state.padding(toLength: 10, withPad: " ", startingAt: 0)) \(mem.padding(toLength: 9, withPad: " ", startingAt: 0)) \(s.detail.prefix(60))")
        }
    }

    @MainActor static func imageCheck(_ engine: ParlorEngine, _ prompt: String) async -> Int32 {
        out("before:"); await engine.refreshServices(); printServices(engine)
        let t0 = Date()
        let r = await engine.hub.call(wireName: "image__generate", arguments: .object(["prompt": .string(prompt.isEmpty ? "a red apple on a table" : prompt), "detail": .string("fast")]))
        out(String(format: "result (%.1fs): error=%@ images=%@ text=%@", Date().timeIntervalSince(t0), "\(r.isError)", r.images.joined(separator: ","), String(r.text.prefix(80))))
        out("after:"); await engine.refreshServices(); printServices(engine)
        engine.stopService("image"); try? await Task.sleep(nanoseconds: 1_500_000_000)
        out("after stop:"); await engine.refreshServices(); printServices(engine)
        return r.isError ? 1 : 0
    }

    static func byte(_ b: Int64) -> String { ByteCountFormatter.string(fromByteCount: b, countStyle: .file) }

    @MainActor static func chat(_ engine: ParlorEngine, _ id: String, _ text: String) async -> Int32 {
        guard let agent = engine.agent(id) else { out("no such agent \(id)"); return 1 }
        let room = engine.room(for: id)
        let before = room.messages.count
        let t0 = Date()
        await room.sendAndWait(text, agent: agent)
        for m in room.messages.dropFirst(before) { describe(m) }
        if let e = room.errorText { out("ERROR: \(e)"); return 1 }
        out(String(format: "(%.1fs, %@)", Date().timeIntervalSince(t0), room.lastStats.map { String(format: "%.0f tok/s", $0.tokensPerSecond) } ?? ""))
        return 0
    }

    static func describe(_ m: ChatMessage) {
        switch m.role {
        case .user: out("you> \(m.text)")
        case .assistant:
            for c in m.toolCalls ?? [] { out("  → call \(c.qualified) \(c.arguments.text)") }
            if !m.text.isEmpty { out("agent> \(m.text)") }
        case .tool: out("  ← \(m.toolName ?? "") \(m.isError == true ? "ERROR " : "")\(String(m.text.prefix(160)).replacingOccurrences(of: "\n", with: " ")) \(m.images.map { "images=\($0)" } ?? "")")
        case .marker: out("  — \(m.text)")
        }
    }
}

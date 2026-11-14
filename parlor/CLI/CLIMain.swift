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
        case "spike": return await Spike.run(engine)
        case "flow": return await Flow.run(engine)
        default: usage(); return 2
        }
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

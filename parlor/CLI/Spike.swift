import Foundation
import ParlorCore

/// Phase 0 gate: does the chosen model call tools correctly on scripted prompts?
enum Spike {
    struct Case { var prompt: String; var expectTool: String; var argContains: String? }

    @MainActor static func run(_ engine: ParlorEngine) async -> Int32 {
        let model = ProcessInfo.processInfo.environment["SPIKE_MODEL"] ?? Defaults.languageModel
        guard engine.library.installedDirectory(for: model) != nil else { out("model \(model) not installed"); return 1 }
        let cases: [Case] = [
            .init(prompt: "What do I need to do today?", expectTool: "zass.briefing", argContains: nil),
            .init(prompt: "Give me my morning briefing.", expectTool: "zass.briefing", argContains: nil),
            .init(prompt: "List my open todos.", expectTool: "zass.todo_list", argContains: nil),
            .init(prompt: "Show me all my notes.", expectTool: "zass.note_list", argContains: nil),
            .init(prompt: "Search my notes and memory for the word lighthouse.", expectTool: "zass.search", argContains: "lighthouse"),
            .init(prompt: "Who are the people I keep track of?", expectTool: "zass.person_list", argContains: nil),
            .init(prompt: "What did I write in my journal in the last 2 days?", expectTool: "zass.journal_read", argContains: nil),
            .init(prompt: "What's in my inbox?", expectTool: "zass.inbox_read", argContains: nil),
            .init(prompt: "What do you remember about me?", expectTool: "zass.memory_list", argContains: nil),
            .init(prompt: "Tell me a one-line joke about compilers.", expectTool: "", argContains: nil),
        ]
        let a = Agent(id: "spike", name: "Spike", persona: "You are a concise personal assistant. Use your tools to look things up instead of guessing.",
                      kind: .assistant, model: model, tools: ["zass.briefing", "zass.todo_list", "zass.note_list", "zass.search", "zass.person_list", "zass.journal_read", "zass.inbox_read", "zass.memory_list"], contextTokens: 8192, temperature: 0.2)
        engine.save(a)
        // wait for the server to come up
        for _ in 0..<60 {
            if engine.tools.contains(where: { $0.server == "zass" }) { break }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        out("tools mounted: \(engine.tools.map(\.qualified).count) · model \(shortName(model))")
        var pass = 0
        for (i, c) in cases.enumerated() {
            let room = engine.room(for: "spike")
            room.reset()
            let t0 = Date()
            await room.sendAndWait(c.prompt, agent: a)
            let calls = room.messages.compactMap { $0.toolCalls }.flatMap { $0 }
            let called = calls.map(\.qualified)
            var ok: Bool
            if c.expectTool.isEmpty { ok = calls.isEmpty }
            else {
                let match = calls.filter { $0.qualified == c.expectTool }
                if let sub = c.argContains { ok = match.contains { $0.arguments.text.lowercased().contains(sub) } }
                else { ok = !match.isEmpty }
            }
            let toolErrors = room.messages.filter { $0.role == .tool && $0.isError == true }.count
            let final = room.messages.last(where: { $0.role == .assistant && !$0.text.isEmpty })?.text ?? "(none)"
            out(String(format: "%2d. %@ %@ → called=%@ errors=%d %.1fs\n      %@", i + 1, ok && toolErrors == 0 ? "PASS" : "FAIL", c.prompt, called.isEmpty ? "-" : called.joined(separator: ","), toolErrors, Date().timeIntervalSince(t0), String(final.prefix(140)).replacingOccurrences(of: "\n", with: " ")))
            if ok && toolErrors == 0 { pass += 1 }
        }
        out("RESULT \(pass)/\(cases.count)")
        return pass >= 9 ? 0 : 1
    }
    static func shortName(_ r: String) -> String { r.split(separator: "/").last.map(String.init) ?? r }
}

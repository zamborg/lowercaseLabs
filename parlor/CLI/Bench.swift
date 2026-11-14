import Foundation
import ParlorCore

/// Prompt-cache check on the real model: the same four-message chat with the cache on and off.
/// Reports prefill reuse and time per turn, and whether the replies match (temperature 0).
enum Bench {
    @MainActor static func run(_ engine: ParlorEngine, model: String) async -> Int32 {
        let persona = (1...60).map { "Fact \($0): the village of Brindle keeps a ledger of every lantern lit on night \($0)." }.joined(separator: "\n")
        var a = engine.create(kind: .assistant, name: "Bench")
        a.model = model; a.temperature = 0; a.thinking = false; a.tools = []; a.persona = "You are a terse assistant. Answer in one sentence.\n" + persona
        engine.save(a)
        let prompts = ["What is the capital of France?", "And of Italy?", "Which of those two is further north?", "Name one river in each."]
        var replies: [[String]] = []
        var seconds: [[Double]] = []
        for cache in [true, false] {
            engine.updateSettings { $0.llm.promptCache = cache; $0.llm.maxReplyTokens = 60 }
            await engine.runtime.unload()
            let room = engine.room(for: a.id)
            room.reset()
            out("— prompt cache \(cache ? "ON" : "OFF")")
            var r: [String] = [], s: [Double] = []
            for p in prompts {
                let t0 = Date()
                await room.sendAndWait(p, agent: a)
                let dt = Date().timeIntervalSince(t0)
                let st = room.lastStats
                let text = room.messages.last(where: { $0.role == .assistant })?.text ?? ""
                out(String(format: "  %.1fs  prompt %d tok, %d from cache, %.0f tok/s  %@", dt, st?.promptTokens ?? 0, st?.cachedTokens ?? 0, st?.tokensPerSecond ?? 0, String(text.prefix(60))))
                r.append(text); s.append(dt)
            }
            replies.append(r); seconds.append(s)
        }
        engine.delete(a.id)
        let same = replies[0] == replies[1]
        out("replies identical with and without cache: \(same)")
        out(String(format: "turn 2-4 total: cache on %.1fs, off %.1fs", seconds[0].dropFirst().reduce(0, +), seconds[1].dropFirst().reduce(0, +)))
        return same ? 0 : 1
    }
}

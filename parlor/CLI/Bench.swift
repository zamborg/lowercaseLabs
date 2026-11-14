import Foundation
import ParlorCore

/// Prompt-cache check on the real model: the same four-message chat with the cache on and off.
/// Reports prefill reuse and time per turn, and whether the replies match (temperature 0).
enum Bench {
    /// Two agents with different personas taking alternate turns, with one cache slot and with four.
    /// With one slot every turn re-processes its whole prompt; with several, each agent keeps its own cache.
    @MainActor static func alternate(_ engine: ParlorEngine, model: String) async -> Int32 {
        func make(_ name: String, _ fact: String) -> Agent {
            var a = engine.create(kind: .assistant, name: name)
            a.model = model; a.temperature = 0; a.tools = []
            a.persona = "You are a terse assistant. Answer in one sentence.\n" + (1...50).map { "Fact \($0): \(fact) \($0)." }.joined(separator: "\n")
            engine.save(a); return a
        }
        let a = make("BenchA", "the clockmaker of Alder winds clock number")
        let b = make("BenchB", "the lighthouse at Cobb Point burned lamp number")
        var cachedTotals: [Int: Int] = [:]
        for slots in [1, 4] {
            engine.updateSettings { $0.cacheSlotsPerModel = slots; $0.llm.promptCache = true; $0.llm.maxReplyTokens = 40 }
            try? await Task.sleep(nanoseconds: 300_000_000)
            await engine.runtime.unload()
            engine.room(for: a.id).reset(); engine.room(for: b.id).reset()
            out("— \(slots) cache slot\(slots == 1 ? "" : "s"), alternating A B A B A B")
            var cached = 0, secs = 0.0
            for i in 0..<6 {
                let ag = i % 2 == 0 ? a : b
                let room = engine.room(for: ag.id)
                let t0 = Date()
                await room.sendAndWait("Say hello number \(i).", agent: ag)
                let dt = Date().timeIntervalSince(t0)
                let st = room.lastStats
                out(String(format: "  %@ %.1fs  prompt %d tok, %d from cache", ag.name, dt, st?.promptTokens ?? 0, st?.cachedTokens ?? 0))
                if i >= 2 { cached += st?.cachedTokens ?? 0; secs += dt }
            }
            cachedTotals[slots] = cached
            out(String(format: "  turns 3-6: %d tokens from cache, %.1fs", cached, secs))
        }
        engine.delete(a.id); engine.delete(b.id)
        let ok = (cachedTotals[4] ?? 0) > (cachedTotals[1] ?? 0)
        out("alternating chats keep their caches with several slots: \(ok)")
        return ok ? 0 : 1
    }

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

import Foundation

/// Folds older messages into a rolling summary so the context stays small.
public struct Compactor: Sendable {
    public var keepRecent: Int
    public var summaryTokens: Int
    public init(keepRecent: Int = 6, summaryTokens: Int = 500) { self.keepRecent = keepRecent; self.summaryTokens = summaryTokens }

    /// Rough token estimate when the model has not reported a real count yet.
    public static func estimateTokens(_ messages: [LLMMessage]) -> Int {
        messages.reduce(0) { $0 + ($1.text.count + $1.toolCalls.reduce(0) { $0 + $1.arguments.text.count }) / 3 + 8 }
    }

    /// Returns the new summary, or nil when there is nothing old enough to fold.
    public func compact(
        agent: Agent, messages: [ChatMessage], existing: ChatLog.Summary?, llm: LanguageModelRuntime,
        priority: LLMPriority = .user
    ) async throws -> ChatLog.Summary? {
        let real = messages.filter { $0.role != .marker }
        guard real.count > keepRecent + 1 else { return nil }
        // Cut at a user message so tool-call/result pairs are never split.
        var cut = real.count - keepRecent
        while cut > 0, real[cut].role != .user { cut -= 1 }
        guard cut > 0 else { return nil }
        let old = Array(real[..<cut])
        var transcript = ""
        for m in old {
            switch m.role {
            case .user: transcript += "User: \(m.text)\n"
            case .assistant:
                if !m.text.isEmpty { transcript += "\(agent.name): \(m.text)\n" }
                for c in m.toolCalls ?? [] { transcript += "(\(agent.name) called \(c.qualified))\n" }
            case .tool: transcript += "(tool \(m.toolName ?? "") returned: \(String(m.text.prefix(300))))\n"
            case .marker: break
            }
        }
        let prior = existing.map { "Existing summary:\n\($0.text)\n\n" } ?? ""
        let prompt = """
        \(prior)New conversation to fold in:
        \(transcript)
        Write an updated summary of the whole conversation so far for \(agent.name) to continue from. \
        Keep durable facts, decisions, preferences, names, dates, open tasks, and the tone of the relationship. \
        Drop pleasantries. Under 250 words. Plain prose or short bullets.
        """
        var out = ""
        let req = LLMRequest(
            messages: [
                LLMMessage(role: .system, text: "You write faithful, compact conversation summaries."),
                LLMMessage(role: .user, text: prompt),
            ], temperature: 0.2, maxTokens: summaryTokens, thinking: false,
            tuning: LLMTuning(useCache: false), priority: priority)
        for try await ev in llm.generate(model: agent.model, request: req) {
            if case .text(let t) = ev { out += t }
        }
        let text = Think.strip(out)
        guard !text.isEmpty else { return nil }
        return ChatLog.Summary(text: text, through: old.last?.id)
    }
}

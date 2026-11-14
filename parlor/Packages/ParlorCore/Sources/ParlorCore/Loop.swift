import Foundation

public enum LoopEvent: Sendable {
    case delta(String)                 // streaming assistant text
    case assistant(ChatMessage)        // finished assistant message (may carry tool calls)
    case toolStarted(ToolCallRecord)
    case tool(ChatMessage)             // tool result
    case stats(LLMStats)
}

/// Swap this to change how an agent thinks and acts. Everything else stays the same.
public protocol AgentLoop: Sendable {
    func run(
        agent: Agent, history: [LLMMessage], tools: [ToolDef],
        llm: LanguageModelRuntime, hub: ToolHub, emit: @Sendable @escaping (LoopEvent) -> Void
    ) async throws
}

/// Stream, run tool calls, feed results back, repeat until the model answers in plain text.
public struct SimpleLoop: AgentLoop {
    public var maxSteps: Int
    public init(maxSteps: Int = 8) { self.maxSteps = maxSteps }

    public func run(
        agent: Agent, history: [LLMMessage], tools: [ToolDef],
        llm: LanguageModelRuntime, hub: ToolHub, emit: @Sendable @escaping (LoopEvent) -> Void
    ) async throws {
        var messages = history
        for step in 0..<maxSteps {
            try Task.checkCancellation()
            // On the last step withhold tools so the model has to answer.
            let offered = step == maxSteps - 1 ? [] : tools
            let req = LLMRequest(messages: messages, tools: offered, temperature: agent.temperature, thinking: agent.thinking)
            var raw = ""
            var calls: [ToolCallRecord] = []
            for try await ev in llm.generate(model: agent.model, request: req) {
                switch ev {
                case .text(let t):
                    raw += t
                    emit(.delta(Think.strip(raw)))
                case .toolCall(let c): calls.append(c)
                case .stats(let s): emit(.stats(s))
                }
            }
            let text = Think.strip(raw)
            emit(.assistant(ChatMessage(role: .assistant, text: text, toolCalls: calls.isEmpty ? nil : calls)))
            messages.append(LLMMessage(role: .assistant, text: text, toolCalls: calls))
            if calls.isEmpty { return }

            for call in calls {
                try Task.checkCancellation()
                emit(.toolStarted(call))
                let started = Date()
                let out: ToolOutput
                if agent.allows(call.qualified) {
                    out = await hub.call(wireName: call.name, arguments: call.arguments)
                } else {
                    out = ToolOutput(text: "Tool \(call.qualified) is not enabled for this agent.", isError: true)
                }
                let secs = Date().timeIntervalSince(started)
                emit(.tool(ChatMessage(
                    role: .tool, text: out.text, toolCallID: call.id, toolName: call.name,
                    isError: out.isError, seconds: secs, images: out.images.isEmpty ? nil : out.images)))
                messages.append(LLMMessage(role: .tool, text: out.text, toolCallID: call.id, toolName: call.name))
            }
        }
    }
}

/// Folds older messages into a rolling summary so the context stays small.
public struct Compactor: Sendable {
    public var keepRecent: Int
    public init(keepRecent: Int = 6) { self.keepRecent = keepRecent }

    /// Rough token estimate when the model has not reported a real count yet.
    public static func estimateTokens(_ messages: [LLMMessage]) -> Int {
        messages.reduce(0) { $0 + ($1.text.count + $1.toolCalls.reduce(0) { $0 + $1.arguments.text.count }) / 3 + 8 }
    }

    /// Returns the new summary, or nil when there is nothing old enough to fold.
    public func compact(
        agent: Agent, messages: [ChatMessage], existing: ChatLog.Summary?, llm: LanguageModelRuntime
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
            ], temperature: 0.2, maxTokens: 500, thinking: false)
        for try await ev in llm.generate(model: agent.model, request: req) {
            if case .text(let t) = ev { out += t }
        }
        let text = Think.strip(out)
        guard !text.isEmpty else { return nil }
        return ChatLog.Summary(text: text, through: old.last?.id)
    }
}

import Foundation

/// Everything a harness needs to produce one reply.
public struct TurnInput: Sendable {
    public var agent: Agent
    public var conversationID: String
    public var participants: [Participant]
    /// The full log including markers; the harness filters.
    public var transcript: [ChatMessage]
    /// The built-in harness's rolling summary.
    public var summary: ChatLog.Summary?
    /// Already filtered by the agent's allowlist.
    public var tools: [ToolDef]
    public var settings: ResolvedSettings
    /// Prompt size reported on the previous turn, 0 if unknown.
    public var lastPromptTokens: Int
    public var now: Date
    public var priority: LLMPriority
    /// Agent id to display name for everyone in the conversation, used for group framing.
    public var names: [String: String]

    public init(
        agent: Agent, conversationID: String, participants: [Participant], transcript: [ChatMessage],
        summary: ChatLog.Summary?, tools: [ToolDef], settings: ResolvedSettings,
        lastPromptTokens: Int = 0, now: Date = Date(), priority: LLMPriority = .user, names: [String: String] = [:]
    ) {
        self.agent = agent; self.conversationID = conversationID; self.participants = participants
        self.transcript = transcript; self.summary = summary; self.tools = tools; self.settings = settings
        self.lastPromptTokens = lastPromptTokens; self.now = now; self.priority = priority; self.names = names
    }
}

public enum TurnEvent: Sendable {
    /// The whole visible text of the current step so far (replace, not append), `<think>` removed.
    case delta(String)
    case status(String)
    case toolStarted(ToolCallRecord)
    /// A finished assistant or tool line, to append to the log.
    case message(ChatMessage)
    /// The harness compacted: the app persists the summary and notes how many messages were folded.
    case summary(ChatLog.Summary, folded: Int)
    case stats(LLMStats)
}

/// The only way a harness runs a tool. Checks the allowlist, applies the timeout and the clip.
public protocol ToolExecutor: Sendable {
    /// Never throws: failures come back as `isError` output.
    func call(_ call: ToolCallRecord, agent: Agent, settings: ResolvedSettings) async -> ToolOutput
}

public struct TurnContext: Sendable {
    public var llm: LanguageModelRuntime
    public var tools: ToolExecutor
    public init(llm: LanguageModelRuntime, tools: ToolExecutor) { self.llm = llm; self.tools = tools }
}

/// Swap this to change how an agent thinks and acts. Everything else stays the same.
public protocol Harness: Sendable {
    func run(_ input: TurnInput, context: TurnContext, emit: @Sendable @escaping (TurnEvent) -> Void) async throws
}

/// Runs tool calls against the shared `ToolHub`.
public struct HubToolExecutor: ToolExecutor {
    public let hub: ToolHub
    public init(hub: ToolHub) { self.hub = hub }

    public func call(_ call: ToolCallRecord, agent: Agent, settings: ResolvedSettings) async -> ToolOutput {
        guard agent.allows(call.qualified) else {
            return ToolOutput(text: "Tool \(call.qualified) is not enabled for this agent.", isError: true)
        }
        var out = await hub.call(wireName: call.name, arguments: call.arguments,
                                 timeout: TimeInterval(settings.toolTimeoutSeconds))
        out.text = BuiltInHarness.clip(out.text, to: settings.toolResultChars)
        return out
    }
}

/// The default harness: compact if needed, stream, run tool calls, feed results back, repeat until the
/// model answers in plain text. Behaviour is the V1 `SimpleLoop` + `Room.buildHistory` + compaction, moved.
extension TurnInput {
    /// The other agents in the conversation (empty in a one-to-one chat).
    var others: [(id: String, name: String)] {
        participants.compactMap(\.agent).filter { $0 != agent.id }.map { ($0, names[$0] ?? $0) }
    }
}

public struct BuiltInHarness: Harness {
    public init() {}

    public func run(_ input: TurnInput, context: TurnContext, emit: @Sendable @escaping (TurnEvent) -> Void) async throws {
        var summary = input.summary
        let s = input.settings
        emit(.status("Thinking"))

        // 1. Compaction first.
        let estimate = Compactor.estimateTokens(
            Self.buildRequestMessages(agent: input.agent, transcript: input.transcript, summary: summary, tools: [], now: input.now, others: input.others))
        if Double(max(input.lastPromptTokens, estimate)) > s.compactAt * Double(s.contextTokens) {
            emit(.status("Compacting"))
            if let (new, folded) = try await Self.compact(input: input, summary: summary, llm: context.llm) {
                summary = new
                emit(.summary(new, folded: folded))
            }
            emit(.status("Thinking"))
        }

        // 2. The loop.
        var messages = Self.buildRequestMessages(agent: input.agent, transcript: input.transcript, summary: summary, tools: input.tools, now: input.now, others: input.others)
        let maxSteps = max(1, s.maxSteps)
        for step in 0..<maxSteps {
            try Task.checkCancellation()
            // On the last step withhold tools so the model has to answer.
            let offered = step == maxSteps - 1 ? [] : input.tools
            let req = LLMRequest(
                messages: messages, tools: offered, temperature: s.temperature,
                maxTokens: s.maxReplyTokens, thinking: s.thinking, tuning: LLMTuning(settings: s.llm),
                priority: input.priority)
            var raw = ""
            var calls: [ToolCallRecord] = []
            for try await ev in context.llm.generate(model: input.agent.model, request: req) {
                switch ev {
                case .text(let t):
                    raw += t
                    emit(.delta(Think.strip(raw)))
                case .toolCall(let c): calls.append(c)
                case .stats(let st): emit(.stats(st))
                }
            }
            let text = Think.strip(raw)
            emit(.message(ChatMessage(role: .assistant, text: text, toolCalls: calls.isEmpty ? nil : calls)))
            messages.append(LLMMessage(role: .assistant, text: text, toolCalls: calls))
            if calls.isEmpty { return }

            for call in calls {
                try Task.checkCancellation()
                emit(.toolStarted(call))
                let started = Date()
                let out = await context.tools.call(call, agent: input.agent, settings: s)
                let secs = Date().timeIntervalSince(started)
                emit(.message(ChatMessage(
                    role: .tool, text: out.text, toolCallID: call.id, toolName: call.name,
                    isError: out.isError, seconds: secs, images: out.images.isEmpty ? nil : out.images)))
                messages.append(LLMMessage(role: .tool, text: out.text, toolCallID: call.id, toolName: call.name))
            }
        }
    }

    // MARK: compaction

    /// Fold older messages into the rolling summary. nil when there is nothing old enough.
    public static func compact(input: TurnInput, summary: ChatLog.Summary?, llm: LanguageModelRuntime) async throws -> (ChatLog.Summary, folded: Int)? {
        let s = input.settings
        guard let new = try await Compactor(keepRecent: s.keepRecent, summaryTokens: s.summaryTokens)
            .compact(agent: input.agent, messages: input.transcript, existing: summary, llm: llm, priority: input.priority, names: input.names) else { return nil }
        let folded = input.transcript.filter { $0.role != .marker }.firstIndex(where: { $0.id == new.through }).map { $0 + 1 } ?? 0
        return (new, folded)
    }

    // MARK: request building

    public static func buildRequestMessages(_ input: TurnInput) -> [LLMMessage] {
        buildRequestMessages(agent: input.agent, transcript: input.transcript, summary: input.summary, tools: input.tools, now: input.now, others: input.others)
    }

    public static func buildRequestMessages(
        agent: Agent, transcript: [ChatMessage], summary: ChatLog.Summary?, tools: [ToolDef], now: Date,
        others: [(id: String, name: String)] = []
    ) -> [LLMMessage] {
        var system = agent.persona.trimmingCharacters(in: .whitespacesAndNewlines)
        if system.isEmpty { system = "You are \(agent.name)." }
        // Date only: a clock in the system prompt would change every minute and defeat the prompt cache.
        let f = DateFormatter(); f.dateStyle = .full; f.timeStyle = .none
        system += "\n\nToday is \(f.string(from: now))."
        if !others.isEmpty {
            let list = others.map(\.name).joined(separator: ", ")
            system += "\nYou are \(agent.name), in a conversation with the user and \(list). Reply only as \(agent.name): never write other participants' lines and never start your reply with your own name."
        }
        if !tools.isEmpty {
            system += "\nYou have tools. When a request needs information or an action from a tool, call the tool immediately in this reply. Never say you will look something up or do something without actually calling the tool. Never ask permission to use a tool for a read-only lookup. After the tool result arrives, answer in plain text."
        }
        if let s = summary { system += "\n\n## Earlier conversation (summary)\n\(s.text)" }

        var live = transcript.filter { $0.role != .marker }
        if let through = summary?.through, let i = live.lastIndex(where: { $0.id == through }) {
            live = Array(live[(i + 1)...])
        }
        var out: [LLMMessage] = [LLMMessage(role: .system, text: system)]
        guard !others.isEmpty else {
            for m in live {
                switch m.role {
                case .user: out.append(LLMMessage(role: .user, text: m.text))
                case .assistant: out.append(LLMMessage(role: .assistant, text: m.text, toolCalls: m.toolCalls ?? []))
                case .tool: out.append(LLMMessage(role: .tool, text: m.text, toolCallID: m.toolCallID, toolName: m.toolName))
                case .marker: break
                }
            }
            return out
        }
        // Group framing: this agent's own lines stay assistant lines; everyone else's final replies become
        // attributed user lines, and their tool steps are left out. Neighbouring user lines are merged.
        let nameOf = Dictionary(others.map { ($0.id, $0.name) }, uniquingKeysWith: { a, _ in a })
        func addUser(_ text: String) {
            if let last = out.last, last.role == .user { out[out.count - 1].text += "\n" + text }
            else { out.append(LLMMessage(role: .user, text: text)) }
        }
        for m in live {
            let who = m.speaker ?? (m.role == .user ? "user" : agent.id)
            switch m.role {
            case .user: addUser("User: \(m.text)")
            case .assistant:
                if who == agent.id { out.append(LLMMessage(role: .assistant, text: m.text, toolCalls: m.toolCalls ?? [])) }
                else if m.toolCalls == nil, !m.text.isEmpty { addUser("\(nameOf[who] ?? who): \(m.text)") }
            case .tool:
                if who == agent.id { out.append(LLMMessage(role: .tool, text: m.text, toolCallID: m.toolCallID, toolName: m.toolName)) }
            case .marker: break
            }
        }
        // A model that just spoke is asked to carry on, so autonomous turns have something to answer.
        if out.last?.role == .assistant { addUser("(Carry on the conversation.)") }
        return out
    }

    /// Cut an oversized tool result, keeping the start and the end, which is where answers and errors live.
    static func clip(_ text: String, to limit: Int) -> String {
        guard limit > 200, text.count > limit else { return text }
        let head = text.prefix(limit * 2 / 3), tail = text.suffix(limit / 3)
        return "\(head)\n… [\(text.count - limit) characters cut to save context] …\n\(tail)"
    }
}

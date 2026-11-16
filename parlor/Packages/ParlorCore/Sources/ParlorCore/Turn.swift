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
    /// The agent's `memory.md`, for the system prompt.
    public var memory: String
    /// The shell line that starts this agent's harness.
    public var command: String = ""
    /// The harness config after schema defaults.
    public var config: [String: JSON] = [:]
    /// A button the user pressed, instead of a message.
    public var action: TurnAction?
    /// The gateway an agent's hosts are served from (set by the engine).
    public var gatewayURL: String = ""

    public init(
        agent: Agent, conversationID: String, participants: [Participant], transcript: [ChatMessage],
        summary: ChatLog.Summary?, tools: [ToolDef], settings: ResolvedSettings,
        lastPromptTokens: Int = 0, now: Date = Date(), priority: LLMPriority = .user, names: [String: String] = [:], memory: String = ""
    ) {
        self.agent = agent; self.conversationID = conversationID; self.participants = participants
        self.transcript = transcript; self.summary = summary; self.tools = tools; self.settings = settings
        self.lastPromptTokens = lastPromptTokens; self.now = now; self.priority = priority; self.names = names; self.memory = memory
    }
}

public struct TurnAction: Sendable, Equatable {
    public var id: String
    public var args: JSON
    public var messageID: String?
    public init(id: String, args: JSON = .object([:]), messageID: String? = nil) { self.id = id; self.args = args; self.messageID = messageID }
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
    public var tools: ToolExecutor
    public init(tools: ToolExecutor) { self.tools = tools }
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
        let need = await hub.requiredTier(call.qualified)
        guard need <= agent.tier else {
            return ToolOutput(text: "Tool \(call.qualified) needs tier \(need.rawValue); this agent has tier \(agent.tier.rawValue).", isError: true)
        }
        var out = await hub.call(wireName: call.name, arguments: call.arguments,
                                 timeout: TimeInterval(settings.toolTimeoutSeconds), caller: .agent(agent))
        out.text = ToolClip.clip(out.text, to: settings.toolResultChars)
        return out
    }
}

extension TurnInput {
    /// The other agents in the conversation (empty in a one-to-one chat).
    var others: [(id: String, name: String)] {
        participants.compactMap(\.agent).filter { $0 != agent.id }.map { ($0, names[$0] ?? $0) }
    }
}

/// Cut an oversized tool result, keeping the start and the end, which is where answers and errors live.
enum ToolClip {
    static func clip(_ text: String, to limit: Int) -> String {
        guard limit > 200, text.count > limit else { return text }
        let head = text.prefix(limit * 2 / 3), tail = text.suffix(limit / 3)
        return "\(head)\n… [\(text.count - limit) characters cut to save context] …\n\(tail)"
    }
}

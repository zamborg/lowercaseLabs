import Foundation

public struct ToolDef: Sendable, Hashable, Identifiable, Codable {
    public var server: String
    public var name: String
    public var description: String
    public var schema: JSON

    public init(server: String, name: String, description: String, schema: JSON) {
        self.server = server; self.name = name; self.description = description; self.schema = schema
    }
    public var id: String { wireName }
    /// Name the model sees. Double underscore keeps server and tool unambiguous.
    public var wireName: String { "\(server)__\(name)" }
    /// Name humans and agent files use.
    public var qualified: String { "\(server).\(name)" }
}

public struct LLMMessage: Sendable {
    public enum Role: String, Sendable { case system, user, assistant, tool }
    public var role: Role
    public var text: String
    public var toolCalls: [ToolCallRecord]
    public var toolCallID: String?
    public var toolName: String?

    public init(role: Role, text: String, toolCalls: [ToolCallRecord] = [], toolCallID: String? = nil, toolName: String? = nil) {
        self.role = role; self.text = text; self.toolCalls = toolCalls
        self.toolCallID = toolCallID; self.toolName = toolName
    }
}

/// Who is waiting on a generation. Lower runs sooner.
public enum LLMPriority: Int, Sendable, Comparable {
    case user = 0        // a person is watching the reply
    case room = 1        // a turn in a multi-agent conversation
    case background = 2  // summaries, scheduled and autonomous work
    public static func < (a: LLMPriority, b: LLMPriority) -> Bool { a.rawValue < b.rawValue }
}

public struct LLMRequest: Sendable {
    public var messages: [LLMMessage]
    public var tools: [ToolDef]
    public var temperature: Double
    public var maxTokens: Int?
    public var thinking: Bool
    public var tuning: LLMTuning
    public var priority: LLMPriority

    public init(messages: [LLMMessage], tools: [ToolDef] = [], temperature: Double = 0.6, maxTokens: Int? = nil, thinking: Bool = false, tuning: LLMTuning = LLMTuning(), priority: LLMPriority = .user) {
        self.messages = messages; self.tools = tools; self.temperature = temperature
        self.maxTokens = maxTokens; self.thinking = thinking; self.tuning = tuning; self.priority = priority
    }
}

public struct LLMStats: Sendable, Equatable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var tokensPerSecond: Double
    /// Prompt tokens served from the reused attention cache instead of being processed (included in `promptTokens`).
    public var cachedTokens: Int
    public init(promptTokens: Int, completionTokens: Int, tokensPerSecond: Double, cachedTokens: Int = 0) {
        self.promptTokens = promptTokens; self.completionTokens = completionTokens
        self.tokensPerSecond = tokensPerSecond; self.cachedTokens = cachedTokens
    }
}

public enum LLMEvent: Sendable {
    case text(String)
    case toolCall(ToolCallRecord)
    case stats(LLMStats)
}

public protocol LanguageModelRuntime: Sendable {
    /// Stream a completion from the given model, loading it first if needed.
    func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error>
    /// Repo id of the model currently in memory, if any.
    func loadedModel() async -> String?
    func unload() async
}

public enum ParlorError: LocalizedError {
    case modelNotInstalled(String)
    case message(String)
    public var errorDescription: String? {
        switch self {
        case .modelNotInstalled(let m): return "Model \(m) is not downloaded yet. Open Models to download it."
        case .message(let s): return s
        }
    }
}

enum Think {
    /// Remove `<think>…</think>` blocks (and an unterminated trailing one) from model text.
    static func strip(_ s: String) -> String {
        var out = s
        while let a = out.range(of: "<think>") {
            if let b = out.range(of: "</think>", range: a.upperBound..<out.endIndex) {
                out.removeSubrange(a.lowerBound..<b.upperBound)
            } else {
                out.removeSubrange(a.lowerBound..<out.endIndex)
            }
        }
        // A bare closing tag can appear when the opening one was part of the prompt.
        if let b = out.range(of: "</think>") { out.removeSubrange(out.startIndex..<b.upperBound) }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

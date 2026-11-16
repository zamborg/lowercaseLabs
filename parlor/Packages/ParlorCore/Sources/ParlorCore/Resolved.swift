import Foundation

/// Optional per-model hints from `.parlor-model.json`: the "model" scope of settings resolution.
public struct ModelMetadata: Sendable, Equatable {
    public var temperature: Double?
    public var topP: Double?
    public var context: Int?
    public var supportsTools: Bool?
    public var supportsVision: Bool?
    public var maxContext: Int?
    public init() {}

    /// nil when the model folder has no metadata file at all.
    public static func load(dir: URL) -> ModelMetadata? {
        guard let d = try? Data(contentsOf: dir.appendingPathComponent(".parlor-model.json")),
              let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return nil }
        var m = ModelMetadata()
        if let defaults = o["defaults"] as? [String: Any] {
            m.temperature = (defaults["temperature"] as? NSNumber)?.doubleValue
            m.topP = (defaults["top_p"] as? NSNumber)?.doubleValue
            m.context = (defaults["context"] as? NSNumber)?.intValue
        }
        if let caps = o["capabilities"] as? [String: Any] {
            m.supportsTools = caps["tools"] as? Bool
            m.supportsVision = caps["vision"] as? Bool
            m.maxContext = (caps["maxContext"] as? NSNumber)?.intValue
        }
        return m
    }
}

/// Everything one turn needs to know about limits and sampling, resolved once: agent, else model, else machine.
public struct ResolvedSettings: Sendable, Equatable {
    public var temperature: Double
    public var topP: Double
    public var repetitionPenalty: Double?
    public var maxReplyTokens: Int?
    public var thinking: Bool
    public var contextTokens: Int
    public var compactAt: Double
    public var keepRecent: Int
    public var summaryTokens: Int
    public var maxSteps: Int
    public var toolResultChars: Int
    public var toolTimeoutSeconds: Int
    /// Machine LLM knobs with sampling and reply limit replaced by the resolved values.
    public var llm: ParlorSettings.LLM

    public static func resolve(agent: Agent, model: ModelMetadata?, machine: ParlorSettings) -> ResolvedSettings {
        var llm = machine.llm
        llm.topP = agent.configNumber("top_p") ?? model?.topP ?? machine.llm.topP
        llm.repetitionPenalty = agent.configNumber("repetition_penalty") ?? machine.llm.repetitionPenalty
        if let r = agent.configInt("max_reply_tokens") { llm.maxReplyTokens = r == 0 ? nil : r }
        var context = agent.configInt("context") ?? 16384
        if let cap = model?.maxContext, cap > 0 { context = min(context, cap) }
        return ResolvedSettings(
            temperature: agent.configNumber("temperature") ?? 0.6, topP: llm.topP, repetitionPenalty: llm.repetitionPenalty,
            maxReplyTokens: llm.maxReplyTokens, thinking: agent.configBool("thinking") ?? false,
            contextTokens: context, compactAt: agent.configNumber("compact_at") ?? 0.7,
            keepRecent: agent.configInt("keep_recent") ?? machine.loop.keepRecent,
            summaryTokens: agent.configInt("summary_tokens") ?? machine.loop.summaryTokens,
            maxSteps: agent.configInt("max_steps") ?? machine.loop.maxSteps,
            toolResultChars: agent.configInt("tool_result_chars") ?? machine.loop.toolResultChars,
            toolTimeoutSeconds: agent.configInt("tool_timeout") ?? machine.loop.toolTimeoutSeconds,
            llm: llm)
    }
}

import Foundation

/// Optional per-model hints from `.parlor-model.json`: the "model" scope of settings resolution.
public struct ModelMetadata: Sendable, Equatable {
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

/// The loop knobs one turn needs, resolved once: the agent's config, else the model's hints, else the machine's loop
/// settings. Sampling (temperature, top_p, repetition penalty, reply length) is not here: the gateway decides it from the
/// request, the host's defaults and the machine's LLM settings.
public struct ResolvedSettings: Sendable, Equatable {
    public var contextTokens: Int
    public var compactAt: Double
    public var keepRecent: Int
    public var summaryTokens: Int
    public var maxSteps: Int
    public var toolResultChars: Int
    public var toolTimeoutSeconds: Int

    public static func resolve(agent: Agent, model: ModelMetadata?, machine: ParlorSettings) -> ResolvedSettings {
        var context = agent.configInt("context") ?? 16384
        if let cap = model?.maxContext, cap > 0 { context = min(context, cap) }
        return ResolvedSettings(
            contextTokens: context, compactAt: agent.configNumber("compact_at") ?? 0.7,
            keepRecent: agent.configInt("keep_recent") ?? machine.loop.keepRecent,
            summaryTokens: agent.configInt("summary_tokens") ?? machine.loop.summaryTokens,
            maxSteps: agent.configInt("max_steps") ?? machine.loop.maxSteps,
            toolResultChars: agent.configInt("tool_result_chars") ?? machine.loop.toolResultChars,
            toolTimeoutSeconds: agent.configInt("tool_timeout") ?? machine.loop.toolTimeoutSeconds)
    }
}

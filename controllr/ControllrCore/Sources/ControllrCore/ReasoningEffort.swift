public enum ReasoningEffort: String, CaseIterable, Sendable {
    case automatic
    case none
    case minimal
    case low
    case medium
    case high
    case xhigh
    case max

    public var title: String {
        switch self {
        case .automatic: "Automatic"
        case .none: "None"
        case .minimal: "Minimal"
        case .low: "Low"
        case .medium: "Medium"
        case .high: "High"
        case .xhigh: "Extra High"
        case .max: "Maximum"
        }
    }

    var outputTokenLimit: Int {
        switch self {
        case .automatic, .none: 400
        case .minimal, .low: 4_096
        case .medium: 8_192
        case .high: 16_384
        case .xhigh: 32_768
        case .max: 65_536
        }
    }

    var requestTimeout: Double {
        switch self {
        case .automatic, .none: 10
        case .minimal, .low: 40
        case .medium: 60
        case .high, .xhigh, .max: 120
        }
    }

    static func openAIOptions(for model: String) -> [Self] {
        let base: [Self] = [.low, .medium, .high]
        if model.hasPrefix("gpt-6") || model.hasPrefix("gpt-5.6") {
            let requiresThinking = ["gpt-6-astra", "gpt-6.1-sol", "gpt-6.2-astra", "gpt-6.2-sol"]
                .contains(where: model.hasPrefix)
            return (requiresThinking ? [] : [.none]) + base + [.xhigh, .max]
        }
        if model.hasPrefix("gpt-5.2") || model.hasPrefix("gpt-5.4") || model.hasPrefix("gpt-5.5") {
            if model.contains("-pro") { return [.medium, .high, .xhigh] }
            return [.none] + base + [.xhigh]
        }
        if model.hasPrefix("gpt-5.1") { return [.none] + base }
        if model.hasPrefix("gpt-5") { return [.minimal] + base }
        if ["o1", "o3", "o4"].contains(where: model.hasPrefix) { return base }
        return []
    }

    static func anthropicOptions(for model: String) -> [Self] {
        let base: [Self] = [.low, .medium, .high]
        if ["claude-opus-4-6", "claude-sonnet-4-6"].contains(where: model.hasPrefix) { return base + [.max] }
        if ["claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-haiku-5-5",
            "claude-fable-5", "claude-mythos-5", "claude-mythos-preview"].contains(where: model.hasPrefix) {
            return base + [.xhigh, .max]
        }
        return []
    }
}

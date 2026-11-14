import Foundation

public struct Agent: Identifiable, Hashable, Sendable {
    public enum Kind: String, Sendable, CaseIterable { case assistant, character }

    /// Slug, also the file name (`agents/<id>.md`) and chat folder name.
    public var id: String
    public var name: String
    public var summary: String
    public var persona: String
    public var kind: Kind
    /// Hugging Face repo id of the language model, e.g. `mlx-community/Qwen3-4B-4bit`.
    public var model: String
    public var imageModel: String?
    /// Tool patterns: `server.tool`, `server.*`, or `*`.
    public var tools: [String]
    public var contextTokens: Int
    public var compactAt: Double
    public var temperature: Double
    public var thinking: Bool
    /// Palette index for the avatar.
    public var color: Int
    /// Per-agent overrides of machine settings. nil inherits.
    public var overrides = AgentOverrides()
    /// Front-matter keys this version does not understand, kept so a save never drops them.
    public var extra: [ExtraField] = []

    public init(
        id: String, name: String, summary: String = "", persona: String = "",
        kind: Kind = .assistant, model: String = Defaults.languageModel,
        imageModel: String? = nil, tools: [String] = [], contextTokens: Int = 16384,
        compactAt: Double = 0.7, temperature: Double = 0.6, thinking: Bool = false, color: Int = 0,
        overrides: AgentOverrides = AgentOverrides(), extra: [ExtraField] = []
    ) {
        self.id = id; self.name = name; self.summary = summary; self.persona = persona
        self.kind = kind; self.model = model; self.imageModel = imageModel; self.tools = tools
        self.contextTokens = contextTokens; self.compactAt = compactAt
        self.temperature = temperature; self.thinking = thinking; self.color = color
        self.overrides = overrides; self.extra = extra
    }

    /// What this agent may do through `parlor.*` tools. Stored as the `tier` front-matter key.
    public var tier: Tier {
        get { Tier(rawValue: extra.first { $0.key == "tier" }.flatMap { f -> String? in if case .scalar(let s) = f.value { return s } else { return nil } } ?? "") ?? .none }
        set {
            extra.removeAll { $0.key == "tier" }
            if newValue != .none { extra.append(ExtraField(key: "tier", value: .scalar(newValue.rawValue))) }
        }
    }

    /// A command that runs this agent's turns instead of the built-in loop (the `harness` front-matter key).
    public var harness: String? {
        get { extra.first { $0.key == "harness" }.flatMap { f -> String? in if case .scalar(let s) = f.value, !s.trimmingCharacters(in: .whitespaces).isEmpty { return s } else { return nil } } }
        set {
            extra.removeAll { $0.key == "harness" }
            if let v = newValue, !v.isEmpty { extra.append(ExtraField(key: "harness", value: .scalar(v))) }
        }
    }

    /// The sidebar section. Defaults from `kind`; the `group` front-matter key overrides it.
    public var defaultGroup: String { kind == .assistant ? "Assistants" : "Characters" }
    public var group: String {
        get { extra.first { $0.key == "group" }.flatMap { f -> String? in if case .scalar(let s) = f.value, !s.trimmingCharacters(in: .whitespaces).isEmpty { return s } else { return nil } } ?? defaultGroup }
        set {
            extra.removeAll { $0.key == "group" }
            let v = newValue.trimmingCharacters(in: .whitespaces)
            if !v.isEmpty, v != defaultGroup { extra.append(ExtraField(key: "group", value: .scalar(v))) }
        }
    }

    public var initial: String { String(name.first.map { String($0).uppercased() } ?? "?") }

    public static func slug(_ name: String) -> String {
        let lowered = name.lowercased()
        var out = ""
        var lastDash = false
        for ch in lowered.unicodeScalars {
            if CharacterSet.alphanumerics.contains(ch) && ch.isASCII { out.unicodeScalars.append(ch); lastDash = false }
            else if !lastDash && !out.isEmpty { out.append("-"); lastDash = true }
        }
        while out.hasSuffix("-") { out.removeLast() }
        return out.isEmpty ? "agent" : out
    }

    /// Whether this agent may call a tool with the given `server.tool` name.
    public func allows(_ qualifiedTool: String) -> Bool {
        for p in tools {
            if p == "*" || p == qualifiedTool { return true }
            if p.hasSuffix(".*"), qualifiedTool.hasPrefix(String(p.dropLast())) { return true }
        }
        return false
    }
}

/// Overrides an agent may set for settings that otherwise come from the model or the machine.
public struct AgentOverrides: Hashable, Sendable {
    public var topP: Double?
    public var repetitionPenalty: Double?
    /// nil inherits; 0 means no cap.
    public var maxReplyTokens: Int?
    public var maxSteps: Int?
    public var toolResultChars: Int?
    public var toolTimeoutSeconds: Int?
    public var keepRecent: Int?
    public var summaryTokens: Int?
    public init() {}
}

public struct ExtraField: Hashable, Sendable {
    public var key: String
    public var value: Frontmatter.Value
    public init(key: String, value: Frontmatter.Value) { self.key = key; self.value = value }
}

public enum Defaults {
    public static let languageModel = "mlx-community/Qwen3-4B-4bit"
    public static let characterModel = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    public static let imageModel = "mlx-community/Z-Image-Turbo-bf16"
}

extension Agent {
    /// Front-matter keys this version reads. Everything else is carried in `extra`.
    static let knownKeys: Set<String> = [
        "name", "description", "kind", "model", "image_model", "tools", "context", "compact_at",
        "temperature", "thinking", "color", "top_p", "repetition_penalty", "max_reply_tokens",
        "max_steps", "tool_result_chars", "tool_timeout", "keep_recent", "summary_tokens",
    ]

    public init(id: String, file text: String) {
        let fm = Frontmatter.parse(text)
        var o = AgentOverrides()
        o.topP = fm.string("top_p").flatMap(Double.init)
        o.repetitionPenalty = fm.string("repetition_penalty").flatMap(Double.init)
        if let r = fm.string("max_reply_tokens") { o.maxReplyTokens = r == "none" ? 0 : Int(r) }
        o.maxSteps = fm.string("max_steps").flatMap(Int.init)
        o.toolResultChars = fm.string("tool_result_chars").flatMap(Int.init)
        o.toolTimeoutSeconds = fm.string("tool_timeout").flatMap(Int.init)
        o.keepRecent = fm.string("keep_recent").flatMap(Int.init)
        o.summaryTokens = fm.string("summary_tokens").flatMap(Int.init)
        self.init(
            id: id,
            name: fm.string("name") ?? id,
            summary: fm.string("description") ?? "",
            persona: fm.body,
            kind: Kind(rawValue: fm.string("kind") ?? "") ?? .assistant,
            model: fm.string("model") ?? Defaults.languageModel,
            imageModel: fm.string("image_model").flatMap { $0.isEmpty ? nil : $0 },
            tools: fm.list("tools") ?? [],
            contextTokens: fm.string("context").flatMap(Int.init) ?? 16384,
            compactAt: fm.string("compact_at").flatMap(Double.init) ?? 0.7,
            temperature: fm.string("temperature").flatMap(Double.init) ?? 0.6,
            thinking: fm.string("thinking") == "true",
            color: fm.string("color").flatMap(Int.init) ?? 0,
            overrides: o,
            extra: fm.fields.filter { !Agent.knownKeys.contains($0.key) }.map { ExtraField(key: $0.key, value: $0.value) }
        )
    }

    public func fileText() -> String {
        var fm = Frontmatter(body: persona)
        fm["name"] = .scalar(name)
        fm["description"] = .scalar(summary)
        fm["kind"] = .scalar(kind.rawValue)
        fm["model"] = .scalar(model)
        if let imageModel, !imageModel.isEmpty { fm["image_model"] = .scalar(imageModel) }
        fm["tools"] = .list(tools)
        fm["context"] = .scalar(String(contextTokens))
        fm["compact_at"] = .scalar(String(compactAt))
        fm["temperature"] = .scalar(String(temperature))
        fm["thinking"] = .scalar(thinking ? "true" : "false")
        fm["color"] = .scalar(String(color))
        if let v = overrides.topP { fm["top_p"] = .scalar(String(v)) }
        if let v = overrides.repetitionPenalty { fm["repetition_penalty"] = .scalar(String(v)) }
        if let v = overrides.maxReplyTokens { fm["max_reply_tokens"] = .scalar(v == 0 ? "none" : String(v)) }
        if let v = overrides.maxSteps { fm["max_steps"] = .scalar(String(v)) }
        if let v = overrides.toolResultChars { fm["tool_result_chars"] = .scalar(String(v)) }
        if let v = overrides.toolTimeoutSeconds { fm["tool_timeout"] = .scalar(String(v)) }
        if let v = overrides.keepRecent { fm["keep_recent"] = .scalar(String(v)) }
        if let v = overrides.summaryTokens { fm["summary_tokens"] = .scalar(String(v)) }
        for f in extra { fm[f.key] = f.value }
        return fm.render()
    }
}

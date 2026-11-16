import Foundation

/// An agent: a harness, a persona, the hosts it uses, its tools and its harness config. On disk it is a folder:
/// `agents/<id>/agent.json` (this, minus the persona), `persona.md`, and `memory.md`.
public struct Agent: Identifiable, Hashable, Sendable {
    public enum Kind: String, Sendable, CaseIterable { case assistant, character }

    /// Slug, also the folder name and the chat folder name.
    public var id: String
    public var name: String
    public var summary: String
    public var persona: String
    public var kind: Kind
    /// A harness id (`chat`, `character`, a folder in `harnesses/`) or an inline command. nil means `chat`.
    public var harness: String?
    /// What the agent uses: `chat` and `image` are host ids.
    public var hosts: [String: String]
    /// Tool patterns: `server.tool`, `server.*`, or `*`.
    public var tools: [String]
    /// What the agent may do through `parlor.*` tools.
    public var tier: Tier
    /// Sidebar section; nil uses the default for `kind`.
    public var groupName: String?
    /// Palette index for the avatar.
    public var color: Int
    /// The harness's settings, validated against its manifest. Unknown keys are kept.
    public var config: [String: JSON]

    public init(
        id: String, name: String, summary: String = "", persona: String = "",
        kind: Kind = .assistant, model: String = "", imageModel: String? = nil, tools: [String] = [],
        contextTokens: Int? = nil, compactAt: Double? = nil, temperature: Double? = nil, thinking: Bool? = nil, color: Int = 0,
        harness: String? = nil, config: [String: JSON] = [:], tier: Tier = .none, group: String? = nil
    ) {
        self.id = id; self.name = name; self.summary = summary; self.persona = persona
        self.kind = kind; self.harness = harness; self.tools = tools; self.tier = tier; self.groupName = group
        self.color = color; self.config = config
        self.hosts = [:]
        if !model.isEmpty { hosts["chat"] = model }
        if let imageModel, !imageModel.isEmpty { hosts["image"] = imageModel }
        if let v = contextTokens { self.config["context"] = .int(v) }
        if let v = compactAt { self.config["compact_at"] = .double(v) }
        if let v = temperature { self.config["temperature"] = .double(v) }
        if let v = thinking { self.config["thinking"] = .bool(v) }
    }

    // MARK: hosts

    /// The chat host id (V2 called this `model`).
    public var model: String {
        get { hosts["chat"] ?? "" }
        set { if newValue.isEmpty { hosts["chat"] = nil } else { hosts["chat"] = newValue } }
    }

    public var imageModel: String? {
        get { hosts["image"] }
        set { if let v = newValue, !v.isEmpty { hosts["image"] = v } else { hosts["image"] = nil } }
    }

    /// The harness this agent runs: its own, or `chat`.
    public var harnessID: String { harness.flatMap { $0.trimmingCharacters(in: .whitespaces).isEmpty ? nil : $0 } ?? "chat" }
    /// True when `harness` is a command line, not a manifest id.
    public var hasInlineHarness: Bool { harnessID.contains(" ") || harnessID.contains("/") }

    // MARK: config accessors (the editor's typed controls)

    private func number(_ k: String) -> Double? {
        switch config[k] { case .double(let d)?: return d; case .int(let i)?: return Double(i); default: return nil }
    }
    private func whole(_ k: String) -> Int? {
        switch config[k] { case .int(let i)?: return i; case .double(let d)?: return Int(d); default: return nil }
    }
    private mutating func set(_ k: String, _ v: JSON?) { config[k] = v }

    public var contextTokens: Int {
        get { whole("context") ?? 16384 }
        set { set("context", .int(newValue)) }
    }
    public var compactAt: Double {
        get { number("compact_at") ?? 0.7 }
        set { set("compact_at", .double(newValue)) }
    }
    public var temperature: Double {
        get { number("temperature") ?? 0.6 }
        set { set("temperature", .double(newValue)) }
    }
    public var thinking: Bool {
        get { if case .bool(let b)? = config["thinking"] { return b } else { return false } }
        set { set("thinking", .bool(newValue)) }
    }

    /// The optional knobs of the `chat` harness. nil means "not set": the host or machine default applies.
    public var overrides: AgentOverrides {
        get {
            var o = AgentOverrides()
            o.topP = number("top_p"); o.repetitionPenalty = number("repetition_penalty"); o.maxReplyTokens = whole("max_reply_tokens")
            o.maxSteps = whole("max_steps"); o.toolResultChars = whole("tool_result_chars"); o.toolTimeoutSeconds = whole("tool_timeout")
            o.keepRecent = whole("keep_recent"); o.summaryTokens = whole("summary_tokens")
            return o
        }
        set {
            set("top_p", newValue.topP.map { .double($0) }); set("repetition_penalty", newValue.repetitionPenalty.map { .double($0) })
            set("max_reply_tokens", newValue.maxReplyTokens.map { .int($0) }); set("max_steps", newValue.maxSteps.map { .int($0) })
            set("tool_result_chars", newValue.toolResultChars.map { .int($0) }); set("tool_timeout", newValue.toolTimeoutSeconds.map { .int($0) })
            set("keep_recent", newValue.keepRecent.map { .int($0) }); set("summary_tokens", newValue.summaryTokens.map { .int($0) })
        }
    }

    // MARK: presentation

    /// The sidebar section. Defaults from `kind`; the `group` key overrides it.
    public var defaultGroup: String { kind == .assistant ? "Assistants" : "Characters" }
    public var group: String {
        get { groupName.flatMap { $0.trimmingCharacters(in: .whitespaces).isEmpty ? nil : $0 } ?? defaultGroup }
        set {
            let v = newValue.trimmingCharacters(in: .whitespaces)
            groupName = (v.isEmpty || v == defaultGroup) ? nil : v
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

/// The `chat` harness's optional knobs, as the editor sees them.
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

public enum Defaults {
    public static let languageModel = "mlx-community/Qwen3-4B-4bit"
    public static let characterModel = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    public static let imageModel = "mlx-community/Z-Image-Turbo-bf16"
}

// MARK: - the file

/// `agents/<id>/agent.json`, the agent minus its persona. A noun like the others.
public struct AgentRecord: Noun, Equatable {
    public static let kind = "agent"
    public var version: Int = 1
    public var id: String
    public var name: String
    public var description: String = ""
    public var harness: String?
    public var hosts: [String: String] = [:]
    public var tools: [String] = []
    public var tier: String?
    public var group: String?
    public var color: Int = 0
    public var style: String?
    public var config: [String: JSON] = [:]

    enum CodingKeys: String, CodingKey { case version, id, name, description, harness, hosts, tools, tier, group, color, style = "kind", config }

    static let knownKeys = Set(["version", "id", "name", "description", "harness", "hosts", "tools", "tier", "group", "color", "kind", "config"])

    public init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        version = try c.decodeIfPresent(Int.self, forKey: .version) ?? 1
        id = try c.decode(String.self, forKey: .id)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? id
        description = try c.decodeIfPresent(String.self, forKey: .description) ?? ""
        harness = try c.decodeIfPresent(String.self, forKey: .harness)
        hosts = try c.decodeIfPresent([String: String].self, forKey: .hosts) ?? [:]
        tools = try c.decodeIfPresent([String].self, forKey: .tools) ?? []
        tier = try c.decodeIfPresent(String.self, forKey: .tier)
        group = try c.decodeIfPresent(String.self, forKey: .group)
        color = try c.decodeIfPresent(Int.self, forKey: .color) ?? 0
        style = try c.decodeIfPresent(String.self, forKey: .style)
        config = try c.decodeIfPresent([String: JSON].self, forKey: .config) ?? [:]
    }

    public init(_ a: Agent) {
        id = a.id; name = a.name; description = a.summary; harness = a.harness
        hosts = a.hosts; tools = a.tools; tier = a.tier == .none ? nil : a.tier.rawValue
        group = a.groupName; color = a.color; style = a.kind.rawValue; config = a.config
    }

    public static func check(_ d: [String: JSON]) -> [Issue] {
        var out: [Issue] = []
        if let t = d["tier"]?.stringValue, Tier(rawValue: t) == nil {
            out.append(Issue(path: "tier", message: "unknown tier \"\(t)\"", hint: "one of " + Tier.allCases.map(\.rawValue).joined(separator: ", ")))
        }
        if let k = d["kind"]?.stringValue, Agent.Kind(rawValue: k) == nil {
            out.append(Issue(path: "kind", message: "unknown kind \"\(k)\"", hint: "assistant or character"))
        }
        return out
    }

    public func agent(persona: String) -> Agent {
        var a = Agent(id: id, name: name, summary: description, persona: persona,
                      kind: style.flatMap(Agent.Kind.init) ?? ((harness == "character") ? .character : .assistant),
                      tools: tools, color: color, harness: harness, config: config,
                      tier: tier.flatMap(Tier.init) ?? .none, group: group)
        a.hosts = hosts
        return a
    }
}

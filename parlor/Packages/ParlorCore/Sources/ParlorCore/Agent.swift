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

    public init(
        id: String, name: String, summary: String = "", persona: String = "",
        kind: Kind = .assistant, model: String = Defaults.languageModel,
        imageModel: String? = nil, tools: [String] = [], contextTokens: Int = 16384,
        compactAt: Double = 0.7, temperature: Double = 0.6, thinking: Bool = false, color: Int = 0
    ) {
        self.id = id; self.name = name; self.summary = summary; self.persona = persona
        self.kind = kind; self.model = model; self.imageModel = imageModel; self.tools = tools
        self.contextTokens = contextTokens; self.compactAt = compactAt
        self.temperature = temperature; self.thinking = thinking; self.color = color
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

public enum Defaults {
    public static let languageModel = "mlx-community/Qwen3-4B-4bit"
    public static let characterModel = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    public static let imageModel = "mlx-community/Z-Image-Turbo-bf16"
}

extension Agent {
    public init(id: String, file text: String) {
        let fm = Frontmatter.parse(text)
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
            color: fm.string("color").flatMap(Int.init) ?? 0
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
        return fm.render()
    }
}

import Foundation

public struct ToolCallRecord: Codable, Hashable, Sendable, Identifiable {
    public var id: String
    public var name: String          // wire name, `server__tool`
    public var arguments: JSON
    public init(id: String = UUID().uuidString, name: String, arguments: JSON) {
        self.id = id; self.name = name; self.arguments = arguments
    }
    /// `server.tool` for display and permission checks.
    public var qualified: String { name.replacingOccurrences(of: "__", with: ".") }
}

public struct ChatMessage: Identifiable, Codable, Hashable, Sendable {
    public enum Role: String, Codable, Sendable { case user, assistant, tool, marker }

    public var id: UUID
    public var role: Role
    public var text: String
    public var toolCalls: [ToolCallRecord]?
    public var toolCallID: String?
    public var toolName: String?
    public var isError: Bool?
    public var seconds: Double?
    /// Paths relative to the parlor home, e.g. `media/2026-11/abc.png`.
    public var images: [String]?
    public var createdAt: Date
    /// Who said it: an agent id or `"user"`. Absent on V1 lines; see `speaker(inChatOf:)`.
    public var speaker: String?
    /// Keys this version does not know, kept so rewriting the log never drops them.
    public var extra: [String: JSON]?

    public init(
        id: UUID = UUID(), role: Role, text: String, toolCalls: [ToolCallRecord]? = nil,
        toolCallID: String? = nil, toolName: String? = nil, isError: Bool? = nil,
        seconds: Double? = nil, images: [String]? = nil, createdAt: Date = Date(), speaker: String? = nil
    ) {
        self.id = id; self.role = role; self.text = text; self.toolCalls = toolCalls
        self.toolCallID = toolCallID; self.toolName = toolName; self.isError = isError
        self.seconds = seconds; self.images = images; self.createdAt = createdAt; self.speaker = speaker
    }

    /// The speaker, defaulting V1 lines: a user line is the user, anything else is the chat's agent.
    public func speaker(inChatOf agentID: String) -> String {
        speaker ?? (role == .user ? "user" : agentID)
    }

    private struct Key: CodingKey {
        var stringValue: String; var intValue: Int? { nil }
        init(_ s: String) { stringValue = s }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }
    private static let known: Set<String> = [
        "id", "role", "text", "toolCalls", "toolCallID", "toolName", "isError", "seconds", "images", "createdAt", "speaker",
    ]

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        id = try c.decode(UUID.self, forKey: Key("id"))
        role = try c.decode(Role.self, forKey: Key("role"))
        text = try c.decode(String.self, forKey: Key("text"))
        toolCalls = try c.decodeIfPresent([ToolCallRecord].self, forKey: Key("toolCalls"))
        toolCallID = try c.decodeIfPresent(String.self, forKey: Key("toolCallID"))
        toolName = try c.decodeIfPresent(String.self, forKey: Key("toolName"))
        isError = try c.decodeIfPresent(Bool.self, forKey: Key("isError"))
        seconds = try c.decodeIfPresent(Double.self, forKey: Key("seconds"))
        images = try c.decodeIfPresent([String].self, forKey: Key("images"))
        createdAt = try c.decode(Date.self, forKey: Key("createdAt"))
        speaker = try c.decodeIfPresent(String.self, forKey: Key("speaker"))
        var rest: [String: JSON] = [:]
        for k in c.allKeys where !Self.known.contains(k.stringValue) {
            rest[k.stringValue] = try c.decode(JSON.self, forKey: k)
        }
        extra = rest.isEmpty ? nil : rest
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: Key.self)
        try c.encode(id, forKey: Key("id"))
        try c.encode(role, forKey: Key("role"))
        try c.encode(text, forKey: Key("text"))
        try c.encodeIfPresent(toolCalls, forKey: Key("toolCalls"))
        try c.encodeIfPresent(toolCallID, forKey: Key("toolCallID"))
        try c.encodeIfPresent(toolName, forKey: Key("toolName"))
        try c.encodeIfPresent(isError, forKey: Key("isError"))
        try c.encodeIfPresent(seconds, forKey: Key("seconds"))
        try c.encodeIfPresent(images, forKey: Key("images"))
        try c.encode(createdAt, forKey: Key("createdAt"))
        try c.encodeIfPresent(speaker, forKey: Key("speaker"))
        for (k, v) in extra ?? [:] where !Self.known.contains(k) { try c.encode(v, forKey: Key(k)) }
    }

    public static func marker(_ text: String) -> ChatMessage { ChatMessage(role: .marker, text: text) }
}

/// One agent's chat: append-only `log.jsonl` plus a rolling `summary.md`.
public final class ChatLog: @unchecked Sendable {
    public let dir: URL
    private var logURL: URL { dir.appendingPathComponent("log.jsonl") }
    private var summaryURL: URL { dir.appendingPathComponent("summary.md") }
    private var toolsURL: URL { dir.appendingPathComponent("tools.json") }
    private let lock = NSLock()

    public init(dir: URL) {
        self.dir = dir
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.withoutEscapingSlashes]
        return e
    }()
    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    public func load() -> [ChatMessage] {
        lock.lock(); defer { lock.unlock() }
        guard let text = try? String(contentsOf: logURL, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap { line in
            try? Self.decoder.decode(ChatMessage.self, from: Data(line.utf8))
        }
    }

    public func append(_ m: ChatMessage) {
        lock.lock(); defer { lock.unlock() }
        guard var data = try? Self.encoder.encode(m) else { return }
        data.append(0x0A)
        if let h = try? FileHandle(forWritingTo: logURL) {
            defer { try? h.close() }
            _ = try? h.seekToEnd()
            try? h.write(contentsOf: data)
        } else {
            try? data.write(to: logURL)
        }
    }

    /// Replace the whole log (used by prune operations).
    public func rewrite(_ messages: [ChatMessage]) {
        lock.lock(); defer { lock.unlock() }
        var out = Data()
        for m in messages {
            if let d = try? Self.encoder.encode(m) { out.append(d); out.append(0x0A) }
        }
        try? out.write(to: logURL)
    }

    /// The tool set offered on the last turn, so "Tools updated" markers survive a relaunch.
    public func loadToolSnapshot() -> Set<String>? {
        guard let d = try? Data(contentsOf: toolsURL),
              let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
              let t = o["tools"] as? [String] else { return nil }
        return Set(t)
    }

    public func saveToolSnapshot(_ tools: Set<String>) {
        let o: [String: Any] = ["version": 1, "tools": tools.sorted()]
        guard let d = try? JSONSerialization.data(withJSONObject: o, options: [.prettyPrinted, .sortedKeys]) else { return }
        try? d.write(to: toolsURL, options: .atomic)
    }

    public struct Summary: Sendable, Equatable {
        public var text: String
        /// Everything up to and including this message id is folded into `text`.
        public var through: UUID?
    }

    public func loadSummary() -> Summary? {
        guard let text = try? String(contentsOf: summaryURL, encoding: .utf8) else { return nil }
        let fm = Frontmatter.parse(text)
        return Summary(text: fm.body, through: fm.string("through").flatMap(UUID.init))
    }

    public func saveSummary(_ s: Summary?) {
        guard let s else { try? FileManager.default.removeItem(at: summaryURL); return }
        var fm = Frontmatter(body: s.text)
        if let t = s.through { fm["through"] = .scalar(t.uuidString) }
        try? fm.render().write(to: summaryURL, atomically: true, encoding: .utf8)
    }
}

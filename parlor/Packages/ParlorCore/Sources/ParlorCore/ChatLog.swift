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

    public init(
        id: UUID = UUID(), role: Role, text: String, toolCalls: [ToolCallRecord]? = nil,
        toolCallID: String? = nil, toolName: String? = nil, isError: Bool? = nil,
        seconds: Double? = nil, images: [String]? = nil, createdAt: Date = Date()
    ) {
        self.id = id; self.role = role; self.text = text; self.toolCalls = toolCalls
        self.toolCallID = toolCallID; self.toolName = toolName; self.isError = isError
        self.seconds = seconds; self.images = images; self.createdAt = createdAt
    }

    public static func marker(_ text: String) -> ChatMessage { ChatMessage(role: .marker, text: text) }
}

/// One agent's chat: append-only `log.jsonl` plus a rolling `summary.md`.
public final class ChatLog: @unchecked Sendable {
    public let dir: URL
    private var logURL: URL { dir.appendingPathComponent("log.jsonl") }
    private var summaryURL: URL { dir.appendingPathComponent("summary.md") }
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

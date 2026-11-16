import Foundation

/// `agents/<id>/memory.md`: durable notes an agent keeps across all its conversations.
public struct MemoryStore: Sendable {
    public let dir: URL
    /// Characters kept in the system prompt; longer memory is refused on append so it never silently eats the context.
    public static let maxChars = 6000

    public init(home: ParlorHome) { dir = home.agentsDir }

    public func url(_ id: String) -> URL { dir.appendingPathComponent(id).appendingPathComponent("memory.md") }

    public func read(_ id: String) -> String {
        (try? String(contentsOf: url(id), encoding: .utf8)) ?? ""
    }

    public func write(_ id: String, _ text: String) throws {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        try FileManager.default.createDirectory(at: url(id).deletingLastPathComponent(), withIntermediateDirectories: true)
        if t.isEmpty { try? FileManager.default.removeItem(at: url(id)); return }
        try (t + "\n").write(to: url(id), atomically: true, encoding: .utf8)
    }

    public func append(_ id: String, _ text: String) throws {
        let add = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !add.isEmpty else { throw CommandError("Nothing to remember.") }
        let cur = read(id).trimmingCharacters(in: .whitespacesAndNewlines)
        let next = cur.isEmpty ? add : cur + "\n" + add
        guard next.count <= Self.maxChars else {
            throw CommandError("Memory is full (\(cur.count) of \(Self.maxChars) characters). Condense it with memory.replace, then append.")
        }
        try write(id, next)
    }
}

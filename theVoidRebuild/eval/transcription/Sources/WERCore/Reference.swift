import Foundation

public struct EvalError: Error, LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

public struct Reference: Codable, Sendable {
    public enum Source: String, Codable, Sendable { case synthetic, recorded, unspecified }
    public let file: String
    public let text: String
    public let source: Source?

    public static func load(from url: URL, audioDirectory: URL) throws -> [Self] {
        let contents = try String(contentsOf: url, encoding: .utf8)
        var references: [Self] = []
        var seen = Set<String>()
        var seenURLs = Set<URL>()
        for (index, line) in contents.components(separatedBy: .newlines).enumerated() {
            if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
            let reference: Self
            do {
                reference = try JSONDecoder().decode(Self.self, from: Data(line.utf8))
            } catch {
                throw EvalError("Reference line \(index + 1): \(error.localizedDescription)")
            }
            guard !reference.file.isEmpty, !reference.file.hasPrefix("/"),
                  !reference.file.split(separator: "/").contains("..") else {
                throw EvalError("Reference line \(index + 1): file must be a nonempty relative path inside --audio.")
            }
            let audioURL = try reference.audioURL(in: audioDirectory)
            guard seen.insert(reference.file).inserted, seenURLs.insert(audioURL).inserted else {
                throw EvalError("Reference line \(index + 1): duplicate audio file \(reference.file).")
            }
            references.append(reference)
        }
        guard !references.isEmpty else { throw EvalError("Reference manifest contains no entries.") }
        return references
    }

    /// Reject symlink escapes as well as literal parent-path traversal.
    public func audioURL(in directory: URL) throws -> URL {
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        let candidate = root.appendingPathComponent(file).resolvingSymlinksInPath().standardizedFileURL
        let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard candidate.path.hasPrefix(prefix) else {
            throw EvalError("Audio file escapes --audio: \(file)")
        }
        return candidate
    }
}

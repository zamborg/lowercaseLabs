import Foundation
import CoreGraphics

public struct Secret {
    public let label: String
    public let value: String
    public init(label: String, value: String) { self.label = label; self.value = value }
}

public struct Match: Equatable {
    /// UTF-16 offsets, as required by macOS accessibility text ranges.
    public var range: NSRange
    public var labels: Set<String>
}

public struct Matcher {
    public let threshold: Int
    private let fragments: [String: Set<String>]

    public init(secrets: [Secret], threshold: Int = 6) throws {
        guard threshold >= 6 else { throw ConfigurationError("Use a threshold of at least six characters.") }
        self.threshold = threshold
        var index: [String: Set<String>] = [:]
        for secret in secrets {
            guard !secret.label.trimmingCharacters(in: .whitespaces).isEmpty else {
                throw ConfigurationError("Each secret needs a label.")
            }
            guard secret.value.utf8.allSatisfy({ $0 >= 33 && $0 <= 126 }), secret.value.utf8.count >= threshold else {
                throw ConfigurationError("Secrets must contain at least \(threshold) printable ASCII characters without spaces.")
            }
            let characters = Array(secret.value.utf8)
            for offset in 0...(characters.count - threshold) {
                let fragment = String(decoding: characters[offset..<(offset + threshold)], as: UTF8.self)
                index[fragment, default: []].insert(secret.label)
            }
        }
        fragments = index
    }

    public func matches(in text: String, offset: Int = 0) -> [Match] {
        let units = Array(text.utf16)
        guard units.count >= threshold else { return [] }
        var matches: [Match] = []
        for index in 0...(units.count - threshold) {
            // Secrets are ASCII; avoid constructing strings for non-ASCII windows.
            let window = units[index..<(index + threshold)]
            guard window.allSatisfy({ $0 >= 33 && $0 <= 126 }) else { continue }
            let fragment = String(decoding: window, as: UTF16.self)
            if let labels = fragments[fragment] {
                let next = NSRange(location: offset + index, length: threshold)
                if let previous = matches.last, NSMaxRange(previous.range) >= next.location {
                    matches[matches.count - 1].range = NSUnionRange(previous.range, next)
                    matches[matches.count - 1].labels.formUnion(labels)
                } else {
                    matches.append(Match(range: next, labels: labels))
                }
            }
        }
        return matches
    }
}

public struct ConfigurationError: LocalizedError {
    public let errorDescription: String?
    public init(_ message: String) { errorDescription = message }
}

public enum Geometry {
    /// AX is top-left based; AppKit uses the primary display's bottom-left origin.
    public static func appKitRect(_ ax: CGRect, primaryHeight: CGFloat) -> CGRect {
        CGRect(x: ax.minX, y: primaryHeight - ax.maxY, width: ax.width, height: ax.height)
    }
}

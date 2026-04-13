import Foundation

struct DictationRecord: Identifiable, Hashable {
    let id: String
    let fileURL: URL
    let createdAt: Date
    let text: String

    var previewText: String {
        let flattened = text
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if flattened.count <= 120 {
            return flattened
        }

        let index = flattened.index(flattened.startIndex, offsetBy: 120)
        return "\(flattened[..<index])…"
    }

    var timestampLabel: String {
        DictationRecord.displayFormatter.string(from: createdAt)
    }

    private static let displayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()
}


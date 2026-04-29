import Foundation

struct SSEMessage: Equatable, Sendable {
    let event: String
    let data: String
}

struct SSEParser {
    private var currentEvent: String?
    private var currentDataLines: [String] = []

    mutating func ingest(line: String) -> SSEMessage? {
        if line.hasPrefix("event: ") {
            currentEvent = String(line.dropFirst(7))
            return nil
        }

        if line.hasPrefix("data: ") {
            currentDataLines.append(String(line.dropFirst(6)))
            return nil
        }

        if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            defer {
                currentEvent = nil
                currentDataLines = []
            }
            guard let currentEvent else {
                return nil
            }
            return SSEMessage(event: currentEvent, data: currentDataLines.joined(separator: "\n"))
        }

        return nil
    }
}

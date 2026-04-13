import Foundation

struct DictationStore {
    private let fileManager = FileManager.default
    private let directoryURL: URL

    init() {
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        directoryURL = documentsURL.appendingPathComponent("Dictations", isDirectory: true)

        if !fileManager.fileExists(atPath: directoryURL.path) {
            try? fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        }
    }

    func save(text: String, date: Date = Date()) throws -> DictationRecord {
        let cleanedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let targetURL = makeUniqueURL(for: date)
        try cleanedText.write(to: targetURL, atomically: true, encoding: .utf8)

        return DictationRecord(
            id: targetURL.lastPathComponent,
            fileURL: targetURL,
            createdAt: date,
            text: cleanedText
        )
    }

    func listRecords() -> [DictationRecord] {
        let urls = ((try? fileManager.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.creationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? [])
        .filter { $0.pathExtension == "txt" }

        return urls.compactMap { url in
            guard let text = try? String(contentsOf: url, encoding: .utf8) else {
                return nil
            }

            let values = try? url.resourceValues(forKeys: [.creationDateKey])
            let createdAt = values?.creationDate ?? dateFromFilename(url) ?? .distantPast

            return DictationRecord(
                id: url.lastPathComponent,
                fileURL: url,
                createdAt: createdAt,
                text: text
            )
        }
        .sorted { $0.createdAt > $1.createdAt }
    }

    func delete(_ record: DictationRecord) throws {
        try fileManager.removeItem(at: record.fileURL)
    }

    private func makeUniqueURL(for date: Date) -> URL {
        let baseName = Self.filenameFormatter.string(from: date)
        var candidate = directoryURL.appendingPathComponent(baseName).appendingPathExtension("txt")
        var suffix = 1

        while fileManager.fileExists(atPath: candidate.path) {
            candidate = directoryURL
                .appendingPathComponent("\(baseName)-\(suffix)")
                .appendingPathExtension("txt")
            suffix += 1
        }

        return candidate
    }

    private func dateFromFilename(_ url: URL) -> Date? {
        let rawName = url.deletingPathExtension().lastPathComponent
        let baseName = rawName.split(separator: "-").prefix(6).joined(separator: "-")
        return Self.filenameFormatter.date(from: baseName)
    }

    private static let filenameFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyy-MM-dd-HH-mm-ss"
        return formatter
    }()
}


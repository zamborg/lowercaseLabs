import Foundation

enum LocalReflectionError: LocalizedError {
    case invalidUserID
    case entryNotFound
    case missingInsightForEntry
    case missingAudioForEntry
    case speechPermissionDenied
    case speechRecognizerUnavailable
    case onDeviceRecognitionUnavailable
    case transcriptionFailed

    var errorDescription: String? {
        switch self {
        case .invalidUserID:
            return "Missing user identity for local storage."
        case .entryNotFound:
            return "Reflection entry was not found."
        case .missingInsightForEntry:
            return "This reflection does not have insight tags yet."
        case .missingAudioForEntry:
            return "Audio not found for this local reflection."
        case .speechPermissionDenied:
            return "Speech recognition permission was denied."
        case .speechRecognizerUnavailable:
            return "Speech recognizer is currently unavailable."
        case .onDeviceRecognitionUnavailable:
            return "On-device speech recognition is unavailable on this device."
        case .transcriptionFailed:
            return "Could not transcribe reflection on device."
        }
    }
}

private struct StoredLocalEntry: Codable {
    let entry: APIEntry
    let audioFileName: String
    let wasSharedToSocial: Bool?
}

struct LocalDeleteResult {
    let deletedEntry: APIEntry
    let replacementSharedEntryForDate: APIEntry?
}

struct LocalEntryTagUpdateResult {
    let updatedEntry: APIEntry
    let wasSharedToSocial: Bool
}

struct LocalEntryAnalysisUpdateResult {
    let updatedEntry: APIEntry
    let wasSharedToSocial: Bool
}

struct LocalJournalStore {
    private let rootDirectory: URL
    private let fileManager = FileManager.default
    private let formatter = ISO8601DateFormatter()

    init() {
        let root = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        rootDirectory = root.appendingPathComponent("VoidLocalJournal", isDirectory: true)
        if !fileManager.fileExists(atPath: rootDirectory.path) {
            try? fileManager.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        }
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    }

    func listEntries(for userID: String) -> [APIEntry] {
        guard let payload = loadPayload(for: userID) else {
            return []
        }
        return payload
            .map(\.entry)
            .sorted(by: Self.sortNewestFirst)
    }

    func saveReflection(
        for userID: String,
        draftURL: URL,
        durationSeconds: Int,
        transcript: APITranscript?,
        insight: APIInsight?,
        localDate: String,
        wasSharedToSocial: Bool
    ) throws -> APIEntry {
        guard !userID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LocalReflectionError.invalidUserID
        }

        let userDirectory = ensureUserDirectory(for: userID)
        let entryID = UUID().uuidString
        let createdAt = formatter.string(from: Date())
        let fileName = "\(entryID).m4a"
        let destinationURL = userDirectory.appendingPathComponent(fileName)

        do {
            if fileManager.fileExists(atPath: destinationURL.path) {
                try fileManager.removeItem(at: destinationURL)
            }
            try fileManager.moveItem(at: draftURL, to: destinationURL)
        } catch {
            let copiedData = try Data(contentsOf: draftURL)
            try copiedData.write(to: destinationURL, options: [.atomic])
            try? fileManager.removeItem(at: draftURL)
        }

        let effectiveSharedToSocial = wasSharedToSocial && (insight != nil)
        let entry = APIEntry(
            id: entryID,
            localDate: localDate,
            durationSeconds: max(1, min(durationSeconds, 300)),
            status: Self.entryStatus(transcript: transcript, insight: insight),
            createdAt: createdAt,
            transcript: transcript,
            insight: insight,
            title: "Entry"
        )

        var payload = loadPayload(for: userID) ?? []
        payload.removeAll(where: { $0.entry.id == entryID })
        payload.append(
            StoredLocalEntry(
                entry: entry,
                audioFileName: fileName,
                wasSharedToSocial: effectiveSharedToSocial
            )
        )
        try savePayload(payload, for: userID)

        return entry
    }

    func deleteEntry(for userID: String, entryID: String) throws -> LocalDeleteResult {
        guard !userID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LocalReflectionError.invalidUserID
        }
        var payload = loadPayload(for: userID) ?? []
        guard let index = payload.firstIndex(where: { $0.entry.id == entryID }) else {
            throw LocalReflectionError.entryNotFound
        }

        let removed = payload.remove(at: index)
        let audioURL = ensureUserDirectory(for: userID).appendingPathComponent(removed.audioFileName)
        if fileManager.fileExists(atPath: audioURL.path) {
            try? fileManager.removeItem(at: audioURL)
        }

        let replacement = payload
            .filter { record in
                record.entry.localDate == removed.entry.localDate && (record.wasSharedToSocial ?? false)
            }
            .map(\.entry)
            .sorted(by: Self.sortNewestFirst)
            .first

        try savePayload(payload, for: userID)
        return LocalDeleteResult(
            deletedEntry: removed.entry,
            replacementSharedEntryForDate: replacement
        )
    }

    func updateEntryTags(
        for userID: String,
        entryID: String,
        moodTags: [String]
    ) throws -> LocalEntryTagUpdateResult {
        guard !userID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LocalReflectionError.invalidUserID
        }

        var payload = loadPayload(for: userID) ?? []
        guard let index = payload.firstIndex(where: { $0.entry.id == entryID }) else {
            throw LocalReflectionError.entryNotFound
        }
        guard let existingInsight = payload[index].entry.insight else {
            throw LocalReflectionError.missingInsightForEntry
        }

        let normalizedTags = Self.sanitizedTags(moodTags)
        let updatedInsight = APIInsight(
            moodScore: EmotionTaxonomy.moodScore(for: normalizedTags),
            moodTags: normalizedTags,
            themes: existingInsight.themes,
            signals: existingInsight.signals,
            safetyFlags: existingInsight.safetyFlags
        )

        let originalEntry = payload[index].entry
        let updatedEntry = APIEntry(
            id: originalEntry.id,
            localDate: originalEntry.localDate,
            durationSeconds: originalEntry.durationSeconds,
            status: originalEntry.status,
            createdAt: originalEntry.createdAt,
            transcript: originalEntry.transcript,
            insight: updatedInsight,
            title: originalEntry.title
        )

        payload[index] = StoredLocalEntry(
            entry: updatedEntry,
            audioFileName: payload[index].audioFileName,
            wasSharedToSocial: payload[index].wasSharedToSocial
        )
        try savePayload(payload, for: userID)

        return LocalEntryTagUpdateResult(
            updatedEntry: updatedEntry,
            wasSharedToSocial: payload[index].wasSharedToSocial ?? false
        )
    }

    func updateEntryAnalysis(
        for userID: String,
        entryID: String,
        transcript: APITranscript,
        insight: APIInsight?
    ) throws -> LocalEntryAnalysisUpdateResult {
        guard !userID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LocalReflectionError.invalidUserID
        }

        var payload = loadPayload(for: userID) ?? []
        guard let index = payload.firstIndex(where: { $0.entry.id == entryID }) else {
            throw LocalReflectionError.entryNotFound
        }

        let originalEntry = payload[index].entry
        let mergedInsight = insight ?? originalEntry.insight
        let updatedEntry = APIEntry(
            id: originalEntry.id,
            localDate: originalEntry.localDate,
            durationSeconds: originalEntry.durationSeconds,
            status: Self.entryStatus(transcript: transcript, insight: mergedInsight),
            createdAt: originalEntry.createdAt,
            transcript: transcript,
            insight: mergedInsight,
            title: originalEntry.title
        )

        payload[index] = StoredLocalEntry(
            entry: updatedEntry,
            audioFileName: payload[index].audioFileName,
            wasSharedToSocial: payload[index].wasSharedToSocial
        )
        try savePayload(payload, for: userID)

        return LocalEntryAnalysisUpdateResult(
            updatedEntry: updatedEntry,
            wasSharedToSocial: payload[index].wasSharedToSocial ?? false
        )
    }

    func updateEntryTitle(
        for userID: String,
        entryID: String,
        title: String
    ) throws -> APIEntry {
        guard !userID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LocalReflectionError.invalidUserID
        }

        var payload = loadPayload(for: userID) ?? []
        guard let index = payload.firstIndex(where: { $0.entry.id == entryID }) else {
            throw LocalReflectionError.entryNotFound
        }

        let originalEntry = payload[index].entry
        let updatedEntry = APIEntry(
            id: originalEntry.id,
            localDate: originalEntry.localDate,
            durationSeconds: originalEntry.durationSeconds,
            status: originalEntry.status,
            createdAt: originalEntry.createdAt,
            transcript: originalEntry.transcript,
            insight: originalEntry.insight,
            title: APIEntry.sanitizeTitle(title)
        )

        payload[index] = StoredLocalEntry(
            entry: updatedEntry,
            audioFileName: payload[index].audioFileName,
            wasSharedToSocial: payload[index].wasSharedToSocial
        )
        try savePayload(payload, for: userID)
        return updatedEntry
    }

    func audioData(for userID: String, entryID: String) throws -> Data {
        let url = try audioURL(for: userID, entryID: entryID)
        return try Data(contentsOf: url)
    }

    func audioURL(for userID: String, entryID: String) throws -> URL {
        guard let record = loadPayload(for: userID)?.first(where: { $0.entry.id == entryID }) else {
            throw LocalReflectionError.missingAudioForEntry
        }
        let url = ensureUserDirectory(for: userID).appendingPathComponent(record.audioFileName)
        guard fileManager.fileExists(atPath: url.path) else {
            throw LocalReflectionError.missingAudioForEntry
        }
        return url
    }

    private func ensureUserDirectory(for userID: String) -> URL {
        let directory = rootDirectory.appendingPathComponent(userID, isDirectory: true)
        if !fileManager.fileExists(atPath: directory.path) {
            try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        return directory
    }

    private func payloadURL(for userID: String) -> URL {
        ensureUserDirectory(for: userID).appendingPathComponent("entries.json")
    }

    private func loadPayload(for userID: String) -> [StoredLocalEntry]? {
        guard !userID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        let url = payloadURL(for: userID)
        guard fileManager.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url) else {
            return []
        }
        return try? JSONDecoder().decode([StoredLocalEntry].self, from: data)
    }

    private func savePayload(_ payload: [StoredLocalEntry], for userID: String) throws {
        let encoded = try JSONEncoder().encode(payload)
        try encoded.write(to: payloadURL(for: userID), options: [.atomic])
    }

    private static func sortNewestFirst(lhs: APIEntry, rhs: APIEntry) -> Bool {
        if lhs.localDate != rhs.localDate {
            return lhs.localDate > rhs.localDate
        }
        return lhs.createdAt > rhs.createdAt
    }

    private static func entryStatus(transcript: APITranscript?, insight: APIInsight?) -> String {
        if insight != nil {
            return "ready"
        }
        if transcript != nil {
            return "transcript_only"
        }
        return "audio_only"
    }

    private static func sanitizedTags(_ tags: [String]) -> [String] {
        var ordered: [String] = []
        for tag in tags {
            let canonical = EmotionTaxonomy.canonicalTag(for: tag)
            if canonical.isEmpty || ordered.contains(canonical) {
                continue
            }
            ordered.append(canonical)
            if ordered.count >= 4 {
                break
            }
        }
        if ordered.isEmpty {
            return ["reflective"]
        }
        return ordered
    }
}

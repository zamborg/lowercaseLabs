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

    init(rootDirectoryOverride: URL? = nil) {
        let root = rootDirectoryOverride ?? fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        rootDirectory = rootDirectoryOverride == nil
            ? root.appendingPathComponent("VoidLocalJournal", isDirectory: true)
            : root
        if !fileManager.fileExists(atPath: rootDirectory.path) {
            try? fileManager.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        }
        excludeFromBackup(rootDirectory)
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    }

    func listEntries(for userID: String) -> [APIEntry] {
        guard let payload = loadPayload(for: userID) else {
            return []
        }
        return payload
            .filter { !$0.isDeleted }
            .map(\.entry)
            .sorted(by: Self.sortNewestFirst)
    }

    func storedEntries(for userID: String) -> [StoredLocalEntry] {
        loadPayload(for: userID) ?? []
    }

    func storedEntry(for userID: String, entryID: String) -> StoredLocalEntry? {
        loadPayload(for: userID)?.first(where: { $0.entry.id == entryID })
    }

    func saveReflection(
        for userID: String,
        draftURL: URL,
        durationSeconds: Int,
        transcript: APITranscript?,
        insight: APIInsight?,
        localDate: String,
        wasSharedToSocial: Bool,
        healthSnapshot: EntryHealthSnapshot? = nil,
        title: String? = nil
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
            title: title ?? "Entry",
            healthSnapshot: healthSnapshot
        )

        var payload = loadPayload(for: userID) ?? []
        payload.removeAll(where: { $0.entry.id == entryID })
        payload.append(
            StoredLocalEntry(
                entry: entry,
                audioFileName: fileName,
                wasSharedToSocial: effectiveSharedToSocial,
                updatedAt: createdAt,
                cloudRecordChangeTag: nil,
                isDeleted: false
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
                record.entry.localDate == removed.entry.localDate && (record.wasSharedToSocial ?? false) && !record.isDeleted
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

    func deleteEntryForSync(for userID: String, entryID: String) throws {
        guard !userID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LocalReflectionError.invalidUserID
        }
        var payload = loadPayload(for: userID) ?? []
        guard let index = payload.firstIndex(where: { $0.entry.id == entryID }) else {
            return
        }
        let removed = payload.remove(at: index)
        let audioURL = ensureUserDirectory(for: userID).appendingPathComponent(removed.audioFileName)
        if fileManager.fileExists(atPath: audioURL.path) {
            try? fileManager.removeItem(at: audioURL)
        }
        try savePayload(payload, for: userID)
    }

    func deleteAudioForSync(for userID: String, entryID: String) throws {
        guard let record = loadPayload(for: userID)?.first(where: { $0.entry.id == entryID }) else {
            return
        }
        let url = ensureUserDirectory(for: userID).appendingPathComponent(record.audioFileName)
        if fileManager.fileExists(atPath: url.path) {
            try? fileManager.removeItem(at: url)
        }
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
            title: originalEntry.title,
            healthSnapshot: originalEntry.healthSnapshot
        )

        payload[index] = StoredLocalEntry(
            entry: updatedEntry,
            audioFileName: payload[index].audioFileName,
            wasSharedToSocial: payload[index].wasSharedToSocial,
            updatedAt: formatter.string(from: Date()),
            cloudRecordChangeTag: nil,
            isDeleted: false
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
        insight: APIInsight?,
        title: String? = nil
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
            title: title ?? originalEntry.title,
            healthSnapshot: originalEntry.healthSnapshot
        )

        payload[index] = StoredLocalEntry(
            entry: updatedEntry,
            audioFileName: payload[index].audioFileName,
            wasSharedToSocial: payload[index].wasSharedToSocial,
            updatedAt: formatter.string(from: Date()),
            cloudRecordChangeTag: nil,
            isDeleted: false
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
            title: APIEntry.sanitizeTitle(title),
            healthSnapshot: originalEntry.healthSnapshot
        )

        payload[index] = StoredLocalEntry(
            entry: updatedEntry,
            audioFileName: payload[index].audioFileName,
            wasSharedToSocial: payload[index].wasSharedToSocial,
            updatedAt: formatter.string(from: Date()),
            cloudRecordChangeTag: nil,
            isDeleted: false
        )
        try savePayload(payload, for: userID)
        return updatedEntry
    }

    func upsertStoredEntry(for userID: String, record: StoredLocalEntry) throws {
        guard !userID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LocalReflectionError.invalidUserID
        }

        var payload = loadPayload(for: userID) ?? []
        payload.removeAll(where: { $0.entry.id == record.entry.id })
        payload.append(record)
        try savePayload(payload, for: userID)
    }

    func updateCloudChangeTag(for userID: String, entryID: String, changeTag: String?) throws {
        guard !userID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LocalReflectionError.invalidUserID
        }

        var payload = loadPayload(for: userID) ?? []
        guard let index = payload.firstIndex(where: { $0.entry.id == entryID }) else {
            return
        }

        let current = payload[index]
        payload[index] = StoredLocalEntry(
            entry: current.entry,
            audioFileName: current.audioFileName,
            wasSharedToSocial: current.wasSharedToSocial,
            updatedAt: current.updatedAt,
            cloudRecordChangeTag: changeTag,
            isDeleted: current.isDeleted
        )
        try savePayload(payload, for: userID)
    }

    func saveAudioDataForSync(for userID: String, entryID: String, data: Data) throws {
        guard !userID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LocalReflectionError.invalidUserID
        }

        let audioFileName = loadPayload(for: userID)?.first(where: { $0.entry.id == entryID })?.audioFileName
            ?? "\(entryID).m4a"
        let url = ensureUserDirectory(for: userID).appendingPathComponent(audioFileName)
        try data.write(to: url, options: [.atomic])
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
        excludeFromBackup(directory)
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

    private func excludeFromBackup(_ url: URL) {
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableURL = url
        try? mutableURL.setResourceValues(values)
    }
}

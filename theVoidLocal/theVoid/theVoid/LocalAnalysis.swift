import Foundation
import Speech

enum LocalReflectionError: LocalizedError {
    case invalidUserID
    case entryNotFound
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
        transcript: APITranscript,
        insight: APIInsight,
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

        let entry = APIEntry(
            id: entryID,
            localDate: localDate,
            durationSeconds: max(1, min(durationSeconds, 300)),
            status: "ready",
            createdAt: createdAt,
            transcript: transcript,
            insight: insight
        )

        var payload = loadPayload(for: userID) ?? []
        payload.removeAll(where: { $0.entry.id == entryID })
        payload.append(
            StoredLocalEntry(
                entry: entry,
                audioFileName: fileName,
                wasSharedToSocial: wasSharedToSocial
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

    func audioData(for userID: String, entryID: String) throws -> Data {
        guard let record = loadPayload(for: userID)?.first(where: { $0.entry.id == entryID }) else {
            throw LocalReflectionError.missingAudioForEntry
        }
        let url = ensureUserDirectory(for: userID).appendingPathComponent(record.audioFileName)
        guard fileManager.fileExists(atPath: url.path) else {
            throw LocalReflectionError.missingAudioForEntry
        }
        return try Data(contentsOf: url)
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
}

enum LocalReflectionAnalyzer {
    private static let safetyTerms = [
        "suicide",
        "kill myself",
        "self harm",
        "hopeless",
        "want to disappear",
    ]

    private static let themeKeywords: [String: [String]] = [
        "work": ["work", "job", "meeting", "manager", "deadline"],
        "relationships": ["partner", "friend", "family", "relationship", "parents"],
        "health": ["sleep", "exercise", "health", "sick", "tired"],
        "self-worth": ["confidence", "worth", "failure", "enough", "self"],
    ]

    static func requestSpeechPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
    }

    static func analyze(audioURL: URL, durationSeconds: Int) async -> (APITranscript, APIInsight) {
        let transcriptText: String
        var provider: String = "ios_speech"

        do {
            transcriptText = try await transcribeOnDevice(audioURL: audioURL)
        } catch {
            provider = "local_stub"
            transcriptText = "Local transcription unavailable for this reflection. Tags were generated from fallback processing."
        }

        let normalized = transcriptText.trimmingCharacters(in: .whitespacesAndNewlines)
        let transcript = APITranscript(
            text: normalized.isEmpty ? "No speech detected in reflection." : normalized,
            providerMetadata: [
                "provider": .string(provider),
                "duration_seconds": .int(max(1, min(durationSeconds, 300))),
            ]
        )
        let insight = extractInsight(from: transcript.text)
        return (transcript, insight)
    }

    private static func transcribeOnDevice(audioURL: URL) async throws -> String {
        let authorization = SFSpeechRecognizer.authorizationStatus()
        guard authorization == .authorized else {
            throw LocalReflectionError.speechPermissionDenied
        }

        let locales = [Locale.current, Locale(identifier: "en_US")]
        var lastError: Error?

        for locale in locales {
            guard let recognizer = SFSpeechRecognizer(locale: locale) else {
                continue
            }
            if !recognizer.isAvailable {
                lastError = LocalReflectionError.speechRecognizerUnavailable
                continue
            }
            if !recognizer.supportsOnDeviceRecognition {
                lastError = LocalReflectionError.onDeviceRecognitionUnavailable
                continue
            }
            do {
                return try await transcribe(audioURL: audioURL, recognizer: recognizer)
            } catch {
                lastError = error
            }
        }

        throw lastError ?? LocalReflectionError.transcriptionFailed
    }

    private static func transcribe(audioURL: URL, recognizer: SFSpeechRecognizer) async throws -> String {
        let request = SFSpeechURLRecognitionRequest(url: audioURL)
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = false
        request.taskHint = .dictation

        return try await withCheckedThrowingContinuation { continuation in
            var resumed = false
            var task: SFSpeechRecognitionTask?
            task = recognizer.recognitionTask(with: request) { result, error in
                if let error {
                    guard !resumed else { return }
                    resumed = true
                    task?.cancel()
                    continuation.resume(throwing: error)
                    return
                }

                guard let result else { return }
                if result.isFinal {
                    guard !resumed else { return }
                    resumed = true
                    let text = result.bestTranscription.formattedString
                    continuation.resume(returning: text)
                }
            }
        }
    }

    private static func extractInsight(from transcriptText: String) -> APIInsight {
        let lowered = transcriptText.lowercased()
        let tags = extractMoodTags(from: lowered)
        return APIInsight(
            moodScore: EmotionTaxonomy.moodScore(for: tags),
            moodTags: tags,
            themes: extractThemes(from: lowered),
            signals: signals(from: lowered, tags: tags),
            safetyFlags: safetyFlags(from: lowered)
        )
    }

    private static func containsPhrase(_ text: String, phrase: String) -> Bool {
        if phrase.contains(" ") {
            return text.contains(phrase)
        }
        let pattern = "\\b\(NSRegularExpression.escapedPattern(for: phrase))\\b"
        return text.range(of: pattern, options: .regularExpression) != nil
    }

    private static func extractMoodTags(from lowered: String) -> [String] {
        EmotionTaxonomy.matchedTags(in: lowered, maxCount: 4) { phrase in
            containsPhrase(lowered, phrase: phrase)
        }
    }

    private static func extractThemes(from lowered: String) -> [String] {
        var themes: [String] = []
        for (theme, words) in themeKeywords {
            if words.contains(where: { containsPhrase(lowered, phrase: $0) }) {
                themes.append(theme)
            }
        }
        return Array(themes.prefix(4))
    }

    private static func signals(from lowered: String, tags: [String]) -> [String: Double] {
        let pleasantness = EmotionTaxonomy.averagePleasantness(for: tags)
        let averageEnergy = EmotionTaxonomy.averageEnergy(for: tags)
        let stressFromWords: Double = ["stress", "stressed", "anxious", "overwhelmed"]
            .contains { containsPhrase(lowered, phrase: $0) } ? 0.75 : 0.45
        let stressFromPosition = max(0.1, min(0.95, (1.0 - ((pleasantness + 1.0) / 2.0))))
        let stress = max(stressFromWords, stressFromPosition)
        let energy = max(0.1, min(0.95, (averageEnergy + 1.0) / 2.0))
        let confidenceFromWords: Double = ["confident", "sure", "ready", "capable"]
            .contains { containsPhrase(lowered, phrase: $0) } ? 0.7 : 0.4
        let confidenceFromPosition = max(0.2, min(0.9, ((pleasantness + 1.0) / 2.0)))
        let confidence = max(confidenceFromWords, confidenceFromPosition)
        return [
            "stress": stress,
            "energy": energy,
            "confidence": confidence,
        ]
    }

    private static func safetyFlags(from lowered: String) -> [String: JSONValue] {
        let matched = safetyTerms.filter { containsPhrase(lowered, phrase: $0) }
        return [
            "needs_review": .bool(!matched.isEmpty),
            "matched_terms": .array(matched.map { .string($0) }),
        ]
    }
}

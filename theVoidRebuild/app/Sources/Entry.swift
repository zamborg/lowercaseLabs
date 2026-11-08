import Foundation
import SwiftData

/// What produced an entry. v1 has one kind. A new module adds a case and a capture surface; nothing else here changes.
enum EntryKind: String { case voice }

enum Visibility: String { case `private`, shared }

/// Who set an enriched field. A user edit always wins and is never overwritten by a later model run.
enum Source: String { case none, model, user }

/// One thing dropped into the Void (the UI calls it a message).
/// CloudKit-backed SwiftData: no unique constraints, every property has a default. Shape matters:
/// CloudKit production schemas can only be extended, never changed.
@Model
final class Entry {
    var id: UUID = UUID()
    var kindValue: String = EntryKind.voice.rawValue
    var createdAt: Date = Date()
    var visibilityValue: String = Visibility.private.rawValue

    // Payload (the user's, never shared). For voice: the audio file and its transcript.
    var duration: TimeInterval = 0
    var audioFileName: String = ""
    var transcript: String = ""

    // Enrichment, each field remembering who set it.
    var title: String = ""
    var titleSourceValue: String = Source.none.rawValue
    /// Quadrant raw values, always from the fixed enum.
    var tagValues: [String] = []
    var tagsSourceValue: String = Source.none.rawValue

    init(id: UUID = UUID(), kind: EntryKind = .voice, createdAt: Date = Date(), visibility: Visibility,
         duration: TimeInterval, audioFileName: String, transcript: String) {
        self.id = id
        self.kindValue = kind.rawValue
        self.createdAt = createdAt
        self.visibilityValue = visibility.rawValue
        self.duration = duration
        self.audioFileName = audioFileName
        self.transcript = transcript
    }

    var kind: EntryKind { EntryKind(rawValue: kindValue) ?? .voice }
    var visibility: Visibility {
        get { Visibility(rawValue: visibilityValue) ?? .private }
        set { visibilityValue = newValue.rawValue }
    }
    var titleSource: Source { Source(rawValue: titleSourceValue) ?? .none }
    var tagsSource: Source { Source(rawValue: tagsSourceValue) ?? .none }
    var tags: [Quadrant] { tagValues.compactMap(Quadrant.init(rawValue:)) }
    var isShared: Bool { visibility == .shared }
    var needsEnrichment: Bool {
        (titleSource == .none || tagsSource == .none) && !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var audioURL: URL { Self.audioDirectory().appendingPathComponent(audioFileName) }

    static func audioDirectory() -> URL {
        // VOID_AUDIO_DIR is a test edge: the Mac harness keeps its files out of ~/Documents.
        let dir = ProcessInfo.processInfo.environment["VOID_AUDIO_DIR"].map { URL(fileURLWithPath: $0) }
            ?? URL.documentsDirectory.appending(path: "audio", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

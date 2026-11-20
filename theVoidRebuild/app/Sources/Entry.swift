import Foundation
import SwiftData

enum Visibility: String { case `private`, shared }

/// Who set an enriched field. A user edit always wins and is never overwritten by a later model run.
enum Source: String { case none, model, user }

/// One thing dropped into the Void (the UI calls it a message), made of **parts**, one per module (D15):
/// a voice part (audio + transcript), a photo part (a back and a front image), or both. A part is absent while
/// its file name is empty. A new module adds its own group of fields here; enrichment, dots and sharing don't change.
///
/// CloudKit-backed SwiftData: no unique constraints, every property has a default. Shape matters:
/// CloudKit production schemas can only be extended, never changed.
@Model
final class Entry {
    var id: UUID = UUID()
    var createdAt: Date = Date()
    var visibilityValue: String = Visibility.private.rawValue

    // Voice part (the user's, never shared).
    var duration: TimeInterval = 0
    var audioFileName: String = ""
    var transcript: String = ""

    // Photo part (the user's, never shared). File names inside the app's photos folder; front may be empty (back-only).
    var photoBackFileName: String = ""
    var photoFrontFileName: String = ""
    /// On-device scene labels for the photo(s): the photo's text for enrichment (D17). Not shared either.
    var photoLabels: String = ""

    // Enrichment, each field remembering who set it.
    var title: String = ""
    var titleSourceValue: String = Source.none.rawValue
    /// Quadrant raw values, always from the fixed enum.
    var tagValues: [String] = []
    var tagsSourceValue: String = Source.none.rawValue

    init(id: UUID = UUID(), createdAt: Date = Date(), visibility: Visibility,
         duration: TimeInterval = 0, audioFileName: String = "", transcript: String = "",
         photoBackFileName: String = "", photoFrontFileName: String = "") {
        self.id = id
        self.createdAt = createdAt
        self.visibilityValue = visibility.rawValue
        self.duration = duration
        self.audioFileName = audioFileName
        self.transcript = transcript
        self.photoBackFileName = photoBackFileName
        self.photoFrontFileName = photoFrontFileName
    }

    var visibility: Visibility {
        get { Visibility(rawValue: visibilityValue) ?? .private }
        set { visibilityValue = newValue.rawValue }
    }
    var titleSource: Source { Source(rawValue: titleSourceValue) ?? .none }
    var tagsSource: Source { Source(rawValue: tagsSourceValue) ?? .none }
    var tags: [Quadrant] { tagValues.compactMap(Quadrant.init(rawValue:)) }
    var isShared: Bool { visibility == .shared }

    var hasVoice: Bool { !audioFileName.isEmpty }
    var hasPhoto: Bool { !photoBackFileName.isEmpty }

    /// What the models read. The transcript leads; a photo-only message is described by its scene labels.
    var enrichmentText: String {
        let spoken = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        if !spoken.isEmpty { return spoken }
        let labels = photoLabels.trimmingCharacters(in: .whitespacesAndNewlines)
        return labels.isEmpty ? "" : "A photo. Scene: \(labels)."
    }
    var needsEnrichment: Bool {
        (titleSource == .none || tagsSource == .none) && !enrichmentText.isEmpty
    }

    var audioURL: URL { Self.audioDirectory().appendingPathComponent(audioFileName) }
    var photoBackURL: URL { Self.photoDirectory().appendingPathComponent(photoBackFileName) }
    var photoFrontURL: URL? { photoFrontFileName.isEmpty ? nil : Self.photoDirectory().appendingPathComponent(photoFrontFileName) }

    static func audioDirectory() -> URL { directory("audio", env: "VOID_AUDIO_DIR") }
    static func photoDirectory() -> URL { directory("photos", env: "VOID_PHOTO_DIR") }

    /// The env vars are test edges: the Mac harness keeps its files out of ~/Documents.
    private static func directory(_ name: String, env: String) -> URL {
        let dir = ProcessInfo.processInfo.environment[env].map { URL(fileURLWithPath: $0) }
            ?? URL.documentsDirectory.appending(path: name, directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

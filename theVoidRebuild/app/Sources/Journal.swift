import Foundation
import SwiftData

/// Everything that happens to an entry, with no UI in it. The app's views and the Mac harness drive the same code.
///
/// Order of obligations: an entry is saved before anything else can fail (priority 1), enrichment runs after
/// and never overwrites the user, and dots are always *derived* from the entries: a dot exists exactly while
/// its entry is shared and tagged (priority 3).
@MainActor
final class Journal {
    let context: ModelContext
    let tagger: Tagger
    let dots: DotStore

    init(context: ModelContext, tagger: Tagger, dots: DotStore = .shared) {
        self.context = context
        self.tagger = tagger
        self.dots = dots
    }

    // MARK: Create

    /// A voice message. Saved immediately; no network, no model.
    @discardableResult
    func save(id: UUID = UUID(), audioFileName: String, duration: TimeInterval, transcript: String,
              visibility: Visibility, createdAt: Date = Date()) -> Entry {
        let entry = Entry(id: id, createdAt: createdAt, visibility: visibility, duration: duration,
                          audioFileName: audioFileName, transcript: transcript)
        context.insert(entry)
        try? context.save()
        return entry
    }

    /// A photo message: the raw back image and, when the front camera produced one, the front image.
    /// The files are written (metadata stripped) and the entry saved before anything else can fail.
    @discardableResult
    func savePhoto(id: UUID = UUID(), back: Data, front: Data?, visibility: Visibility, createdAt: Date = Date()) throws -> Entry {
        let (backName, frontName) = try storePhotos(id: id, back: back, front: front)
        let entry = Entry(id: id, createdAt: createdAt, visibility: visibility,
                          photoBackFileName: backName, photoFrontFileName: frontName)
        context.insert(entry)
        try? context.save()
        return entry
    }

    /// A photo joins an existing message (D15). Whatever the message already is stays; one photo part per message.
    func attachPhoto(back: Data, front: Data?, to entry: Entry) throws {
        PhotoFiles.remove([entry.photoBackFileName, entry.photoFrontFileName])
        let (backName, frontName) = try storePhotos(id: UUID(), back: back, front: front)
        entry.photoBackFileName = backName
        entry.photoFrontFileName = frontName
        entry.photoLabels = ""
        // A spoken message keeps its title and tags: the transcript leads, so the photo changes nothing the model read.
        if entry.transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { reopenModelFields(entry) }
        changed()
    }

    /// A recording joins an existing message, such as the photo you snapped while you started talking.
    func attachVoice(audioFileName: String, duration: TimeInterval, transcript: String, to entry: Entry) {
        entry.audioFileName = audioFileName
        entry.duration = duration
        entry.transcript = transcript
        reopenModelFields(entry)   // the transcript now leads, so what the model said from the photo alone is stale
        changed()
    }

    private func storePhotos(id: UUID, back: Data, front: Data?) throws -> (String, String) {
        let backName = try PhotoFiles.store(back, as: "\(id.uuidString)-back.jpg")
        let frontName = try front.map { try PhotoFiles.store($0, as: "\(id.uuidString)-front.jpg") } ?? ""
        return (backName, frontName)
    }

    /// Model-set fields become fillable again; fields the user set are never reopened.
    private func reopenModelFields(_ entry: Entry) {
        if entry.titleSource == .model { entry.titleSourceValue = Source.none.rawValue }
        if entry.tagsSource == .model { entry.tagsSourceValue = Source.none.rawValue }
    }

    // MARK: Enrich

    /// Title and tags from the model, for whichever of them the user hasn't set. Throws if the model fails;
    /// the entry is untouched in that case and `enrichPending` will try again later.
    func enrich(_ entry: Entry) async throws {
        describePhotoIfNeeded(entry)
        guard entry.needsEnrichment else { return }
        let tagging = try await tagger.tag(transcript: entry.enrichmentText)
        if entry.titleSource != .user {
            entry.title = tagging.title
            entry.titleSourceValue = Source.model.rawValue
        }
        if entry.tagsSource != .user {
            entry.tagValues = tagging.tags.map(\.rawValue)
            entry.tagsSourceValue = Source.model.rawValue
        }
        try? context.save()
        await reconcileDots()
    }

    /// Scene labels for the photo part, computed once (on the CPU, so safe whatever the app's state).
    private func describePhotoIfNeeded(_ entry: Entry) {
        guard entry.hasPhoto, entry.photoLabels.isEmpty else { return }
        entry.photoLabels = PhotoFiles.labels(for: [entry.photoBackURL] + [entry.photoFrontURL].compactMap { $0 })
        try? context.save()
    }

    /// Entries saved while the model was cold or failed. Run after the tagger warms.
    func enrichPending() async {
        let all = (try? context.fetch(FetchDescriptor<Entry>(sortBy: [SortDescriptor(\.createdAt, order: .reverse)]))) ?? []
        for entry in all where entry.needsEnrichment || (entry.hasPhoto && entry.photoLabels.isEmpty) {
            try? await enrich(entry)
        }
    }

    /// Settings → re-tag: runs the model again over every title and tag *it* set. What the user set is never touched.
    func retagModelFields() async {
        for entry in (try? context.fetch(FetchDescriptor<Entry>())) ?? [] { reopenModelFields(entry) }
        try? context.save()
        await enrichPending()
    }

    // MARK: Edit (the user always wins)

    func setTitle(_ title: String, on entry: Entry) {
        entry.title = title
        entry.titleSourceValue = Source.user.rawValue
        changed()
    }

    func setTags(_ tags: [Quadrant], on entry: Entry) {
        entry.tagValues = Array(NSOrderedSet(array: tags).array.compactMap { $0 as? Quadrant }.prefix(2)).map(\.rawValue)
        entry.tagsSourceValue = Source.user.rawValue
        changed()
    }

    func setTranscript(_ text: String, on entry: Entry) {
        entry.transcript = text
        changed()
    }

    func setVisibility(_ visibility: Visibility, on entry: Entry) {
        entry.visibility = visibility
        changed()
    }

    /// Takes every part with it: audio, both photos and the dot.
    func delete(_ entry: Entry) {
        if entry.hasVoice { try? FileManager.default.removeItem(at: entry.audioURL) }
        PhotoFiles.remove([entry.photoBackFileName, entry.photoFrontFileName])
        context.delete(entry)
        changed()
    }

    private func changed() {
        try? context.save()
        Task { await reconcileDots() }
    }

    // MARK: Dots

    /// The only thing that leaves the device: id, time and tags of shared entries. Everything else stays.
    func reconcileDots() async {
        let all = (try? context.fetch(FetchDescriptor<Entry>())) ?? []
        let desired = all.filter { $0.isShared && !$0.tags.isEmpty }
            .map { Dot(id: $0.id.uuidString, createdAt: $0.createdAt, tags: $0.tags) }
        await dots.reconcile(desired: desired)
    }
}

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

    /// Saved immediately; no network, no model.
    @discardableResult
    func save(id: UUID = UUID(), audioFileName: String, duration: TimeInterval, transcript: String,
              visibility: Visibility, createdAt: Date = Date()) -> Entry {
        let entry = Entry(id: id, createdAt: createdAt, visibility: visibility, duration: duration,
                          audioFileName: audioFileName, transcript: transcript)
        context.insert(entry)
        try? context.save()
        return entry
    }

    // MARK: Enrich

    /// Title and tags from the model, for whichever of them the user hasn't set. Throws if the model fails;
    /// the entry is untouched in that case and `enrichPending` will try again later.
    func enrich(_ entry: Entry) async throws {
        guard entry.needsEnrichment else { return }
        let tagging = try await tagger.tag(transcript: entry.transcript)
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

    /// Entries saved while the model was cold or failed. Run after the tagger warms.
    func enrichPending() async {
        let all = (try? context.fetch(FetchDescriptor<Entry>(sortBy: [SortDescriptor(\.createdAt, order: .reverse)]))) ?? []
        for entry in all where entry.needsEnrichment { try? await enrich(entry) }
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

    func delete(_ entry: Entry) {
        try? FileManager.default.removeItem(at: entry.audioURL)
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

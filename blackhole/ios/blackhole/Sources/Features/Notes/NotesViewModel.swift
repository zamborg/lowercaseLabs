import Foundation
import Observation

@MainActor
@Observable
final class NotesViewModel {
    var notes: [NoteRecord] = []
    var searchResults: [SearchNoteResult] = []
    var searchQuery = ""
    var nextCursor: String?
    var isLoading = false
    var errorMessage: String?

    private let notesRepository: NotesRepository
    private let searchRepository: SearchRepository
    private let navigator: AppNavigator

    init(notesRepository: NotesRepository, searchRepository: SearchRepository, navigator: AppNavigator) {
        self.notesRepository = notesRepository
        self.searchRepository = searchRepository
        self.navigator = navigator
    }

    var isShowingSearchResults: Bool {
        !searchQuery.isEmpty
    }

    func refresh() async {
        guard !isLoading else { return }
        isLoading = true
        defer { isLoading = false }

        do {
            let page = try await notesRepository.listNotes(cursor: nil, limit: 20)
            notes = page.items
            nextCursor = page.nextCursor
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func search() async {
        guard !searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            searchResults = []
            return
        }
        do {
            let page = try await searchRepository.searchNotes(query: searchQuery, cursor: nil, limit: 20)
            searchResults = page.items
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func applyExternalSearchIfNeeded() async {
        guard let query = navigator.consumePendingSearchQuery() else {
            return
        }
        searchQuery = query
        await search()
    }

    func save(note: NoteRecord) async {
        do {
            let updated = try await notesRepository.updateNote(
                id: note.id,
                request: NoteUpdateRequest(
                    title: note.title,
                    bodyMarkdown: note.bodyMarkdown,
                    tags: note.tags,
                    archived: note.archivedAt != nil
                )
            )
            if let index = notes.firstIndex(where: { $0.id == updated.id }) {
                notes[index] = updated
            }
            if let index = searchResults.firstIndex(where: { $0.id == updated.id }) {
                searchResults[index] = SearchNoteResult(
                    id: updated.id,
                    title: updated.title,
                    snippet: updated.bodyMarkdown,
                    tags: updated.tags,
                    score: searchResults[index].score,
                    updatedAt: updated.updatedAt
                )
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}


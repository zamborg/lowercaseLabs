import SwiftUI

struct NotesView: View {
    @Bindable var model: NotesViewModel
    @Bindable var navigator: AppNavigator

    var body: some View {
        NavigationStack {
            ZStack {
                AppTheme.background
                    .ignoresSafeArea()

                VStack(spacing: 16) {
                    HStack(spacing: 12) {
                        TextField("Search notes", text: $model.searchQuery)
                            .textFieldStyle(.roundedBorder)
                        Button {
                            navigator.startVoiceSearch()
                        } label: {
                            Image(systemName: "mic.circle.fill")
                                .font(.system(size: 32))
                                .foregroundStyle(AppTheme.accentStrong)
                        }
                        Button("Go") {
                            Task { await model.search() }
                        }
                    }
                    .padding(.horizontal, 20)

                    List {
                        if model.isShowingSearchResults {
                            ForEach(model.searchResults) { result in
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(result.title)
                                        .font(.headline)
                                    Text(result.snippet)
                                        .font(.subheadline)
                                        .foregroundStyle(.secondary)
                                    Text(result.tags.joined(separator: " • "))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                .listRowBackground(Color.white.opacity(0.74))
                            }
                        } else {
                            ForEach(model.notes) { note in
                                NavigationLink {
                                    NoteDetailView(note: note) { updated in
                                        await model.save(note: updated)
                                    }
                                } label: {
                                    VStack(alignment: .leading, spacing: 6) {
                                        Text(note.title)
                                            .font(.headline)
                                        Text(note.bodyMarkdown)
                                            .font(.subheadline)
                                            .foregroundStyle(.secondary)
                                            .lineLimit(3)
                                        Text(note.tags.joined(separator: " • "))
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    .padding(.vertical, 4)
                                }
                                .listRowBackground(Color.white.opacity(0.74))
                            }
                        }
                    }
                    .scrollContentBackground(.hidden)
                    .background(Color.clear)
                }
            }
            .navigationTitle("Notes")
            .task {
                await model.refresh()
                await model.applyExternalSearchIfNeeded()
            }
        }
    }
}

struct NoteDetailView: View {
    @Environment(\.dismiss) private var dismiss
    @State var note: NoteRecord
    let onSave: @MainActor (NoteRecord) async -> Void

    var body: some View {
        Form {
            TextField("Title", text: $note.title)
            TextField("Tags (comma separated)", text: Binding(
                get: { note.tags.joined(separator: ", ") },
                set: { note.tags = $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) } }
            ))
            TextEditor(text: $note.bodyMarkdown)
                .frame(minHeight: 220)
        }
        .navigationTitle("Note")
        .toolbar {
            Button("Save") {
                Task {
                    await onSave(note)
                    dismiss()
                }
            }
        }
    }
}


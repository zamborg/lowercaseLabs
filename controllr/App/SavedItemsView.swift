import SwiftData
import SwiftUI

struct SavedItemsView: View {
    @Query(sort: \SavedItem.updatedAt, order: .reverse) private var items: [SavedItem]
    @State private var query = ""
    @State private var adding = false

    private var visibleItems: [SavedItem] {
        let search = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !search.isEmpty else { return items }
        return items.filter {
            $0.title.localizedStandardContains(search) || $0.text.localizedStandardContains(search)
                || $0.kind.title.localizedStandardContains(search)
        }
    }

    var body: some View {
        NavigationStack {
            List {
                ForEach(SavedItemKind.allCases) { kind in
                    let group = visibleItems.filter { $0.kind == kind }
                    if !group.isEmpty {
                        Section(kind.title == "Memory" ? "Memory" : "\(kind.title)s") {
                            ForEach(group) { item in
                                NavigationLink {
                                    SavedItemEditor(item: item)
                                } label: {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(item.title)
                                        Text(item.text).font(.subheadline).foregroundStyle(.secondary).lineLimit(2)
                                    }
                                }
                            }
                        }
                    }
                }
            }
            .overlay {
                if visibleItems.isEmpty {
                    if query.isEmpty {
                        ContentUnavailableView("Nothing saved yet", systemImage: "bookmark",
                                               description: Text("Add a memory, skill, or note."))
                    } else {
                        ContentUnavailableView.search(text: query)
                    }
                }
            }
            .searchable(text: $query, prompt: "Search saved items")
            .navigationTitle("Saved")
            .toolbar {
                Button("Add saved item", systemImage: "plus") { adding = true }
            }
            .sheet(isPresented: $adding) {
                NavigationStack { SavedItemEditor() }
            }
        }
    }
}

private struct SavedItemEditor: View {
    let item: SavedItem?
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var context
    @State private var kind: SavedItemKind
    @State private var title: String
    @State private var text: String
    @State private var showError = false
    @State private var confirmDelete = false

    init(item: SavedItem? = nil) {
        self.item = item
        _kind = State(initialValue: item?.kind ?? .memory)
        _title = State(initialValue: item?.title ?? "")
        _text = State(initialValue: item?.text ?? "")
    }

    var body: some View {
        Form {
            Picker("Kind", selection: $kind) {
                ForEach(SavedItemKind.allCases) { Text($0.title).tag($0) }
            }
            TextField("Title", text: $title)
            Section(kind == .skill ? "Instructions" : "Text") {
                TextEditor(text: $text)
                    .accessibilityLabel(kind == .skill ? "Instructions" : "Text")
                    .frame(minHeight: 150)
            }
            if item != nil {
                Button("Delete", role: .destructive) { confirmDelete = true }
            }
        }
        .navigationTitle(item == nil ? "New saved item" : "Edit saved item")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if item == nil {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            }
            ToolbarItem(placement: .confirmationAction) {
                Button("Save") {
                    do {
                        _ = try SavedItems.shared.save(kind: kind, title: title, text: text, updating: item)
                        dismiss()
                    } catch { showError = true }
                }
                .disabled(title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .alert("Couldn't save your change", isPresented: $showError) { Button("OK", role: .cancel) {} }
        .confirmationDialog("Delete this item?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete item", role: .destructive) {
                guard let item else { return }
                context.delete(item)
                do {
                    try context.save()
                    dismiss()
                } catch {
                    context.rollback()
                    showError = true
                }
            }
        }
    }
}

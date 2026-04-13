import SwiftUI

@MainActor
final class FeedViewModel: ObservableObject {
    enum Filter: String, CaseIterable { case all = "All", notes = "Notes", todos = "Todos" }

    @Published var items: [Item] = []
    @Published var isLoading = false
    @Published var alertMessage: String?
    @Published var filter: Filter = .all

    var displayedItems: [Item] {
        switch filter {
        case .all: return items
        case .notes: return items.filter { $0.type == .note }
        case .todos: return items.filter { $0.type == .todo }
        }
    }

    func load() async {
        isLoading = true
        do { items = try await APIClient.shared.listItems() }
        catch { alertMessage = error.localizedDescription }
        isLoading = false
    }

    func toggleCompleted(_ item: Item) async {
        do {
            let updated = try await APIClient.shared.updateItem(id: item.id, completed: !item.completed)
            if let idx = items.firstIndex(where: { $0.id == item.id }) { items[idx] = updated }
        } catch { alertMessage = error.localizedDescription }
    }

    func delete(_ item: Item) async {
        do {
            try await APIClient.shared.deleteItem(id: item.id)
            items.removeAll { $0.id == item.id }
        } catch { alertMessage = error.localizedDescription }
    }
}

struct FeedView: View {
    @StateObject private var viewModel = FeedViewModel()

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                content
            }
            .navigationTitle("Feed")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.black, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Picker("Filter", selection: $viewModel.filter) {
                        ForEach(FeedViewModel.Filter.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.menu)
                    .tint(.white)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button { Task { await viewModel.load() } } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .tint(.white)
                    .disabled(viewModel.isLoading)
                }
            }
        }
        .preferredColorScheme(.dark)
        .task { await viewModel.load() }
        .alert("Error", isPresented: Binding(
            get: { viewModel.alertMessage != nil },
            set: { if !$0 { viewModel.alertMessage = nil } }
        )) {
            Button("OK") { viewModel.alertMessage = nil }
        } message: {
            Text(viewModel.alertMessage ?? "")
        }
    }

    @ViewBuilder
    private var content: some View {
        if viewModel.isLoading && viewModel.items.isEmpty {
            ProgressView().tint(.white)
        } else if viewModel.displayedItems.isEmpty {
            Text(viewModel.items.isEmpty ? "Nothing here yet." : "No \(viewModel.filter.rawValue.lowercased()).")
                .font(.system(.subheadline, design: .rounded))
                .foregroundStyle(.white.opacity(0.4))
        } else {
            List {
                ForEach(viewModel.displayedItems) { item in
                    ItemRow(item: item) {
                        Task { await viewModel.toggleCompleted(item) }
                    }
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                    .listRowInsets(EdgeInsets(top: 4, leading: 16, bottom: 4, trailing: 16))
                    .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                        Button(role: .destructive) { Task { await viewModel.delete(item) } } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                }
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
        }
    }
}

private struct ItemRow: View {
    let item: Item
    let onToggle: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if item.type == .todo {
                Button(action: onToggle) {
                    Image(systemName: item.completed ? "checkmark.circle.fill" : "circle")
                        .font(.system(size: 22))
                        .foregroundStyle(item.completed ? .yellow : .white.opacity(0.45))
                }
                .buttonStyle(.plain)
            } else {
                Image(systemName: "note.text")
                    .font(.system(size: 16))
                    .foregroundStyle(.white.opacity(0.35))
                    .padding(.top, 3)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(item.title)
                    .font(.system(.subheadline, design: .rounded, weight: .semibold))
                    .foregroundStyle(item.completed ? .white.opacity(0.35) : .white)
                    .strikethrough(item.completed)

                Text(item.previewText)
                    .font(.system(.caption, design: .serif))
                    .foregroundStyle(.white.opacity(0.5))
                    .lineLimit(2)

                HStack(spacing: 8) {
                    if let due = item.dueDateFormatted {
                        Label(due, systemImage: "calendar")
                            .font(.system(.caption2, design: .rounded, weight: .medium))
                            .foregroundStyle(.yellow.opacity(0.8))
                    }
                    Text(item.timestampLabel)
                        .font(.system(.caption2, design: .rounded))
                        .foregroundStyle(.white.opacity(0.28))
                }
            }

            Spacer()
        }
        .padding(14)
        .background(Color.white.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(.white.opacity(0.08), lineWidth: 1))
    }
}

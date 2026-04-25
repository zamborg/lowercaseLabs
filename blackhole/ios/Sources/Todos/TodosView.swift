import SwiftUI

@MainActor
final class TodosViewModel: ObservableObject {
    @Published var todos: [Item] = []
    @Published var isLoading = false
    @Published var alertMessage: String?

    var openTodos: [Item] {
        todos
            .filter { $0.type == .todo && !$0.completed }
            .sorted { lhs, rhs in
                switch (lhs.dueDateParsed, rhs.dueDateParsed) {
                case (.some(let l), .some(let r)): return l < r
                case (.some, .none): return true
                case (.none, .some): return false
                case (.none, .none): return lhs.createdAtDate > rhs.createdAtDate
                }
            }
    }

    var doneTodos: [Item] {
        todos
            .filter { $0.type == .todo && $0.completed }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    func load() async {
        if todos.isEmpty {
            todos = await APIClient.shared.cachedItems()
        }

        isLoading = todos.isEmpty
        do {
            todos = try await APIClient.shared.listItems()
        } catch {
            if todos.isEmpty {
                alertMessage = error.localizedDescription
            }
        }
        isLoading = false
    }

    func toggleCompleted(_ item: Item) async {
        do {
            let updated = try await APIClient.shared.updateItem(id: item.id, completed: !item.completed)
            if let idx = todos.firstIndex(where: { $0.id == item.id }) { todos[idx] = updated }
        } catch { alertMessage = error.localizedDescription }
    }

    func delete(_ item: Item) async {
        do {
            try await APIClient.shared.deleteItem(id: item.id)
            todos.removeAll { $0.id == item.id }
        } catch { alertMessage = error.localizedDescription }
    }
}

struct TodosView: View {
    @StateObject private var viewModel = TodosViewModel()
    @State private var selectedItem: Item?

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                content
            }
            .navigationTitle("Todos")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.black, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbar {
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
        .sheet(item: $selectedItem) { item in
            ItemDetailView(item: item) { updated in
                if let idx = viewModel.todos.firstIndex(where: { $0.id == updated.id }) {
                    viewModel.todos[idx] = updated
                }
            }
        }
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
        if viewModel.isLoading && viewModel.todos.isEmpty {
            ProgressView().tint(.white)
        } else if viewModel.openTodos.isEmpty && viewModel.doneTodos.isEmpty {
            VStack(spacing: 6) {
                Text("No todos yet.")
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundStyle(.white.opacity(0.4))
                Text("Dictate into the Void.")
                    .font(.system(.caption, design: .rounded))
                    .foregroundStyle(.white.opacity(0.25))
            }
        } else {
            List {
                if !viewModel.openTodos.isEmpty {
                    Section {
                        ForEach(viewModel.openTodos) { item in
                            todoRow(item)
                        }
                    } header: {
                        todoSectionHeader("Open", count: viewModel.openTodos.count)
                    }
                }

                if !viewModel.doneTodos.isEmpty {
                    Section {
                        ForEach(viewModel.doneTodos) { item in
                            todoRow(item)
                        }
                    } header: {
                        todoSectionHeader("Done", count: viewModel.doneTodos.count)
                    }
                }
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
            .refreshable { await viewModel.load() }
        }
    }

    private func todoRow(_ item: Item) -> some View {
        Button { selectedItem = item } label: {
            TodoRow(item: item) {
                Task { await viewModel.toggleCompleted(item) }
            }
        }
        .buttonStyle(.plain)
        .listRowBackground(Color.clear)
        .listRowSeparator(.hidden)
        .listRowInsets(EdgeInsets(top: 4, leading: 16, bottom: 4, trailing: 16))
        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
            Button(role: .destructive) { Task { await viewModel.delete(item) } } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    private func todoSectionHeader(_ title: String, count: Int) -> some View {
        HStack(spacing: 4) {
            Text(title.uppercased())
            Text("·")
            Text("\(count)")
        }
        .font(.system(.caption2, design: .rounded, weight: .semibold))
        .foregroundStyle(.white.opacity(0.4))
        .textCase(nil)
        .listRowInsets(EdgeInsets(top: 12, leading: 0, bottom: 4, trailing: 0))
    }
}

private struct TodoRow: View {
    let item: Item
    let onToggle: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Button(action: onToggle) {
                Image(systemName: item.completed ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 22))
                    .foregroundStyle(
                        item.completed ? .white.opacity(0.3)
                        : item.isOverdue ? .red.opacity(0.7)
                        : .white.opacity(0.45)
                    )
            }
            .buttonStyle(.plain)

            VStack(alignment: .leading, spacing: 4) {
                Text(item.title)
                    .font(.system(.subheadline, design: .rounded, weight: .semibold))
                    .foregroundStyle(item.completed ? .white.opacity(0.3) : .white)
                    .strikethrough(item.completed)
                    .lineLimit(2)

                if !item.tags.isEmpty {
                    HStack(spacing: 4) {
                        ForEach(item.tags.prefix(3), id: \.self) { tag in
                            Text(tag)
                                .font(.system(.caption2, design: .rounded))
                                .foregroundStyle(.white.opacity(0.4))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.white.opacity(0.07))
                                .clipShape(Capsule())
                        }
                    }
                }

                if let due = item.dueDateFormatted {
                    Label(due, systemImage: "calendar")
                        .font(.system(.caption2, design: .rounded, weight: .medium))
                        .foregroundStyle(
                            item.completed ? .white.opacity(0.2)
                            : item.isOverdue ? .red.opacity(0.85)
                            : .yellow.opacity(0.8)
                        )
                }
            }

            Spacer()

            Image(systemName: "chevron.right")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.white.opacity(0.2))
        }
        .padding(14)
        .background(item.isOverdue && !item.completed ? Color.red.opacity(0.05) : Color.white.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(item.isOverdue && !item.completed ? .red.opacity(0.2) : .white.opacity(0.08), lineWidth: 1)
        )
    }
}

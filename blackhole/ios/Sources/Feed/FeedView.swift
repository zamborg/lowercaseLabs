import SwiftUI

@MainActor
final class FeedViewModel: ObservableObject {
    @Published var items: [Item] = []
    @Published var searchResults: [Item] = []
    @Published var searchQuery = ""
    @Published var isLoading = false
    @Published var isSearching = false
    @Published var hasSearched = false
    @Published var alertMessage: String?
    @Published private(set) var isSearchListening = false
    @Published private(set) var isSearchDictationBusy = false
    @Published private(set) var isSearchRecordingLocked = false

    private let audioCapture = AudioCapturePipeline()
    private let engineCoordinator = DictationEngineCoordinator()
    private var currentSession: DictationSession?
    private var isSearchHoldActive = false
    private var pendingSearchStopAfterStart = false
    private var pendingSearchLockAfterStart = false

    var displayedItems: [Item] {
        isSearchActive ? searchResults : items
    }

    var isSearchActive: Bool {
        hasSearched && !searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func load() async {
        if items.isEmpty {
            items = await APIClient.shared.cachedItems()
        }

        isLoading = items.isEmpty
        do {
            items = try await APIClient.shared.listItems()
        } catch {
            if items.isEmpty {
                alertMessage = error.localizedDescription
            }
        }
        isLoading = false
    }

    func toggleCompleted(_ item: Item) async {
        do {
            let updated = try await APIClient.shared.updateItem(id: item.id, completed: !item.completed)
            replace(updated)
        } catch { alertMessage = error.localizedDescription }
    }

    func delete(_ item: Item) async {
        do {
            try await APIClient.shared.deleteItem(id: item.id)
            items.removeAll { $0.id == item.id }
            searchResults.removeAll { $0.id == item.id }
        } catch { alertMessage = error.localizedDescription }
    }

    func search() async {
        let query = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else {
            clearSearch()
            return
        }

        isSearching = true
        do {
            searchResults = try await APIClient.shared.search(query: query)
            hasSearched = true
        } catch {
            searchResults = localSearch(query: query)
            hasSearched = true
            if searchResults.isEmpty {
                alertMessage = error.localizedDescription
            }
        }
        isSearching = false
    }

    func clearSearch() {
        searchQuery = ""
        searchResults = []
        hasSearched = false
    }

    func searchPressDidStart() {
        if isSearchListening, isSearchRecordingLocked {
            Task { [weak self] in await self?.stopSearchDictation() }
            return
        }
        guard !isSearchListening, !isSearchDictationBusy else { return }
        isSearchHoldActive = true
        pendingSearchStopAfterStart = false
        pendingSearchLockAfterStart = false
        Task { [weak self] in await self?.startSearchDictation() }
    }

    func searchPressDidEndInside() {
        isSearchHoldActive = false
        if isSearchDictationBusy {
            pendingSearchStopAfterStart = true
        } else if isSearchListening, !isSearchRecordingLocked {
            Task { [weak self] in await self?.stopSearchDictation() }
        }
    }

    func searchPressDidEndOutside() {
        isSearchHoldActive = false
        if isSearchDictationBusy {
            pendingSearchLockAfterStart = true
        } else if isSearchListening {
            isSearchRecordingLocked = true
        }
    }

    private func startSearchDictation() async {
        isSearchDictationBusy = true
        let hasMic = await audioCapture.requestPermission()
        guard hasMic else {
            alertMessage = DictationError.microphonePermissionDenied.localizedDescription
            resetSearchDictationState()
            return
        }

        do {
            let engine = try await engineCoordinator.engine(for: .apple)
            let session = try engine.makeSession(locale: .current)
            session.onUpdate = { [weak self] update in
                Task { @MainActor [weak self] in self?.searchQuery = update.text }
            }
            try audioCapture.start { [weak session] buffer, when in
                try? session?.append(buffer, at: when)
            }
            currentSession = session
            isSearchListening = true
            isSearchDictationBusy = false

            if pendingSearchLockAfterStart {
                isSearchRecordingLocked = true
            } else if pendingSearchStopAfterStart || !isSearchHoldActive {
                await stopSearchDictation()
            }
        } catch {
            alertMessage = error.localizedDescription
            currentSession?.cancel()
            currentSession = nil
            audioCapture.stop()
            resetSearchDictationState()
        }
    }

    private func stopSearchDictation() async {
        guard isSearchListening, let session = currentSession else { return }
        isSearchDictationBusy = true
        isSearchListening = false
        isSearchRecordingLocked = false
        audioCapture.stop()
        currentSession = nil

        do {
            searchQuery = try await session.finish().trimmingCharacters(in: .whitespacesAndNewlines)
            resetSearchDictationState(keepQuery: true)
            await search()
        } catch {
            alertMessage = error.localizedDescription
            resetSearchDictationState(keepQuery: true)
        }
    }

    private func replace(_ item: Item) {
        if let idx = items.firstIndex(where: { $0.id == item.id }) { items[idx] = item }
        if let idx = searchResults.firstIndex(where: { $0.id == item.id }) { searchResults[idx] = item }
    }

    func epicTitle(for item: Item) -> String? {
        guard let epicId = item.epicId else { return nil }
        return items.first(where: { $0.id == epicId && $0.type == .epic })?.title
    }

    private func localSearch(query: String) -> [Item] {
        let normalized = query.lowercased()
        return items.filter { item in
            [
                item.title,
                item.content,
                item.tags.joined(separator: " "),
                item.type.displayName,
                epicTitle(for: item) ?? "",
            ]
            .joined(separator: " ")
            .lowercased()
            .contains(normalized)
        }
    }

    private func resetSearchDictationState(keepQuery: Bool = true) {
        isSearchListening = false
        isSearchDictationBusy = false
        isSearchRecordingLocked = false
        isSearchHoldActive = false
        pendingSearchStopAfterStart = false
        pendingSearchLockAfterStart = false
        if !keepQuery { clearSearch() }
    }
}

struct FeedView: View {
    @StateObject private var viewModel = FeedViewModel()
    @State private var selectedItem: Item?
    @FocusState private var isSearchFocused: Bool

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                VStack(spacing: 0) {
                    searchBar
                        .padding(16)
                    Divider().overlay(.white.opacity(0.1))
                    content
                }
            }
            .navigationTitle("Feed")
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
            ItemDetailView(item: item, epicTitle: viewModel.epicTitle(for: item)) { updated in
                if let idx = viewModel.items.firstIndex(where: { $0.id == updated.id }) {
                    viewModel.items[idx] = updated
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

    private var searchBar: some View {
        HStack(spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.white.opacity(0.4))
                    .font(.system(size: 16))

                TextField("Search feed…", text: $viewModel.searchQuery)
                    .font(.system(.body, design: .rounded))
                    .foregroundStyle(.white)
                    .tint(.white)
                    .focused($isSearchFocused)
                    .submitLabel(.search)
                    .onSubmit { Task { await viewModel.search() } }
                    .onChange(of: viewModel.searchQuery) { _, newValue in
                        if newValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            viewModel.clearSearch()
                        }
                    }

                if !viewModel.searchQuery.isEmpty {
                    Button { viewModel.clearSearch() } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.white.opacity(0.4))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Color.white.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

            FeedSearchMicButton(
                isActive: viewModel.isSearchListening,
                isLocked: viewModel.isSearchRecordingLocked,
                isDisabled: viewModel.isSearchDictationBusy || viewModel.isSearching,
                onPressStart: { viewModel.searchPressDidStart() },
                onPressEndInside: { viewModel.searchPressDidEndInside() },
                onPressEndOutside: { viewModel.searchPressDidEndOutside() }
            )
        }
    }

    @ViewBuilder
    private var content: some View {
        if (viewModel.isLoading && viewModel.items.isEmpty) || viewModel.isSearching {
            ProgressView().tint(.white)
        } else if viewModel.displayedItems.isEmpty {
            VStack(spacing: 6) {
                Text(viewModel.isSearchActive ? "No results." : "Nothing here yet.")
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundStyle(.white.opacity(0.4))
                if !viewModel.isSearchActive {
                    Text("Dictate into the Void.")
                        .font(.system(.caption, design: .rounded))
                        .foregroundStyle(.white.opacity(0.25))
                }
            }
        } else {
            List {
                ForEach(viewModel.displayedItems) { item in
                    Button { selectedItem = item } label: {
                        ItemRow(item: item, epicTitle: viewModel.epicTitle(for: item)) {
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
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
            .refreshable { await viewModel.load() }
        }
    }
}

struct NotesView: View {
    var body: some View {
        TypeItemsView(
            itemType: .note,
            title: "Notes",
            emptyTitle: "No notes yet.",
            emptySubtitle: "Captured notes will show up here."
        )
    }
}

struct EpicsView: View {
    var body: some View {
        TypeItemsView(
            itemType: .epic,
            title: "Epics",
            emptyTitle: "No epics yet.",
            emptySubtitle: "Ask the Void to create an epic."
        )
    }
}

@MainActor
final class TypeItemsViewModel: ObservableObject {
    @Published var items: [Item] = []
    @Published var isLoading = false
    @Published var alertMessage: String?

    let itemType: Item.ItemType

    init(itemType: Item.ItemType) {
        self.itemType = itemType
    }

    var filteredItems: [Item] { items.filter { $0.type == itemType } }

    var epics: [Item] {
        items
            .filter { $0.type == .epic }
            .sorted { $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending }
    }

    var epicGroups: [EpicItemGroup] {
        let candidates = filteredItems
        let grouped = Dictionary(grouping: candidates) { item in
            item.epicId ?? EpicItemGroup.unassignedID
        }

        var groups = epics.compactMap { epic -> EpicItemGroup? in
            guard let epicItems = grouped[epic.id], !epicItems.isEmpty else { return nil }
            return EpicItemGroup(
                id: epic.id,
                title: epic.title,
                epic: epic,
                items: sortedItems(epicItems)
            )
        }

        if let unassigned = grouped[EpicItemGroup.unassignedID], !unassigned.isEmpty {
            groups.append(EpicItemGroup(
                id: EpicItemGroup.unassignedID,
                title: "No Epic",
                epic: nil,
                items: sortedItems(unassigned)
            ))
        }

        return groups
    }

    func epicTitle(for item: Item) -> String? {
        guard let epicId = item.epicId else { return nil }
        return epics.first(where: { $0.id == epicId })?.title
    }

    func epicStats(for epic: Item) -> EpicStats {
        let related = items.filter { $0.epicId == epic.id }
        let todos = related.filter { $0.type == .todo }
        return EpicStats(
            notes: related.filter { $0.type == .note }.count,
            todos: todos.count,
            openTodos: todos.filter { !$0.completed }.count
        )
    }

    private func sortedItems(_ source: [Item]) -> [Item] {
        if itemType == .todo {
            return source.sorted { lhs, rhs in
                switch (lhs.dueDateParsed, rhs.dueDateParsed) {
                case (.some(let l), .some(let r)): return l < r
                case (.some, .none): return true
                case (.none, .some): return false
                case (.none, .none): return lhs.createdAtDate > rhs.createdAtDate
                }
            }
        }
        return source.sorted { $0.createdAtDate > $1.createdAtDate }
    }

    func load() async {
        if items.isEmpty {
            items = await APIClient.shared.cachedItems()
        }

        isLoading = items.isEmpty
        do {
            items = try await APIClient.shared.listItems()
        } catch {
            if items.isEmpty {
                alertMessage = error.localizedDescription
            }
        }
        isLoading = false
    }

    func delete(_ item: Item) async {
        do {
            try await APIClient.shared.deleteItem(id: item.id)
            items.removeAll { $0.id == item.id }
        } catch { alertMessage = error.localizedDescription }
    }
}

enum ItemGroupingMode: String, CaseIterable, Identifiable {
    case recent
    case epic

    var id: String { rawValue }

    var title: String {
        switch self {
        case .recent: return "Recent"
        case .epic: return "Epic"
        }
    }
}

struct EpicItemGroup: Identifiable {
    static let unassignedID = "__no_epic__"

    let id: String
    let title: String
    let epic: Item?
    let items: [Item]
}

struct EpicStats {
    let notes: Int
    let todos: Int
    let openTodos: Int
}

private struct TypeItemsView: View {
    @StateObject private var viewModel: TypeItemsViewModel
    @State private var selectedItem: Item?
    @State private var groupingMode: ItemGroupingMode = .recent
    let title: String
    let emptyTitle: String
    let emptySubtitle: String

    init(itemType: Item.ItemType, title: String, emptyTitle: String, emptySubtitle: String) {
        _viewModel = StateObject(wrappedValue: TypeItemsViewModel(itemType: itemType))
        self.title = title
        self.emptyTitle = emptyTitle
        self.emptySubtitle = emptySubtitle
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                content
            }
            .navigationTitle(title)
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
            ItemDetailView(item: item, epicTitle: viewModel.epicTitle(for: item)) { updated in
                if let idx = viewModel.items.firstIndex(where: { $0.id == updated.id }) {
                    viewModel.items[idx] = updated
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
        if viewModel.isLoading && viewModel.items.isEmpty {
            ProgressView().tint(.white)
        } else if viewModel.filteredItems.isEmpty {
            VStack(spacing: 6) {
                Text(emptyTitle)
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundStyle(.white.opacity(0.4))
                Text(emptySubtitle)
                    .font(.system(.caption, design: .rounded))
                    .foregroundStyle(.white.opacity(0.25))
            }
        } else {
            VStack(spacing: 0) {
                if canGroupByEpic {
                    groupingPicker
                        .padding(.horizontal, 16)
                        .padding(.top, 12)
                        .padding(.bottom, 6)
                }

                List {
                    if viewModel.itemType == .epic {
                        ForEach(viewModel.filteredItems) { item in
                            Button { selectedItem = item } label: {
                                EpicRow(item: item, stats: viewModel.epicStats(for: item))
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
                    } else if groupingMode == .epic {
                        ForEach(viewModel.epicGroups) { group in
                            Section {
                                ForEach(group.items) { item in
                                    itemRow(item)
                                }
                            } header: {
                                groupHeader(group.title, count: group.items.count)
                            }
                        }
                    } else {
                        ForEach(viewModel.filteredItems) { item in
                            itemRow(item)
                        }
                    }
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
                .refreshable { await viewModel.load() }
            }
        }
    }

    private var canGroupByEpic: Bool {
        viewModel.itemType == .note || viewModel.itemType == .todo
    }

    private var groupingPicker: some View {
        Picker("Grouping", selection: $groupingMode) {
            ForEach(ItemGroupingMode.allCases) { mode in
                Text(mode.title).tag(mode)
            }
        }
        .pickerStyle(.segmented)
    }

    private func itemRow(_ item: Item) -> some View {
        Button { selectedItem = item } label: {
            ItemRow(item: item, epicTitle: viewModel.epicTitle(for: item)) {}
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

    private func groupHeader(_ title: String, count: Int) -> some View {
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

struct ItemDetailView: View {
    @State private var displayItem: Item
    let epicTitle: String?
    let onItemUpdated: ((Item) -> Void)?
    @State private var isEditing = false
    @Environment(\.dismiss) private var dismiss

    init(item: Item, epicTitle: String? = nil, onItemUpdated: ((Item) -> Void)? = nil) {
        _displayItem = State(initialValue: item)
        self.epicTitle = epicTitle
        self.onItemUpdated = onItemUpdated
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        // Metadata row
                        HStack(spacing: 10) {
                            Image(systemName: displayItem.type.systemImage)
                                .font(.system(size: 14))
                                .foregroundStyle(.white.opacity(0.4))
                            Text(displayItem.type.displayName)
                                .font(.system(.caption, design: .rounded, weight: .semibold))
                                .foregroundStyle(.white.opacity(0.4))
                            Spacer()
                            Text(displayItem.timestampLabel)
                                .font(.system(.caption, design: .rounded))
                                .foregroundStyle(.white.opacity(0.3))
                        }

                        // Due date (todos only)
                        if let due = displayItem.dueDateFormatted {
                            Label(due, systemImage: "calendar")
                                .font(.system(.subheadline, design: .rounded, weight: .medium))
                                .foregroundStyle(displayItem.isOverdue ? .red.opacity(0.85) : .yellow.opacity(0.85))
                        }

                        if let epicTitle {
                            Label(epicTitle, systemImage: "square.stack.3d.up")
                                .font(.system(.subheadline, design: .rounded, weight: .medium))
                                .foregroundStyle(.cyan.opacity(0.8))
                        }

                        // Tags
                        if !displayItem.tags.isEmpty {
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 6) {
                                    ForEach(displayItem.tags, id: \.self) { tag in
                                        Text(tag)
                                            .font(.system(.caption2, design: .rounded, weight: .semibold))
                                            .foregroundStyle(.white.opacity(0.6))
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 4)
                                            .background(Color.white.opacity(0.08))
                                            .clipShape(Capsule())
                                    }
                                }
                            }
                        }

                        Divider().overlay(Color.white.opacity(0.1))

                        // Full content
                        Text(displayItem.content)
                            .font(.system(.body, design: .serif))
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .padding(20)
                }
            }
            .navigationTitle(displayItem.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.black, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Edit") { isEditing = true }
                        .tint(.white)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                        .tint(.white)
                }
            }
        }
        .preferredColorScheme(.dark)
        .sheet(isPresented: $isEditing) {
            EditItemView(item: displayItem) { updated in
                displayItem = updated
                onItemUpdated?(updated)
            }
        }
    }
}

private struct FeedSearchMicButton: View {
    let isActive: Bool
    let isLocked: Bool
    let isDisabled: Bool
    let onPressStart: () -> Void
    let onPressEndInside: () -> Void
    let onPressEndOutside: () -> Void

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(isActive ? Color(white: 0.22) : Color.white.opacity(0.1))
            Image(systemName: symbolName)
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(.white)
            PressAndHoldCaptureView(
                isDisabled: isDisabled,
                onPressStart: onPressStart,
                onPressEndInside: onPressEndInside,
                onPressEndOutside: onPressEndOutside
            )
        }
        .frame(width: 44, height: 44)
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(.white.opacity(0.15), lineWidth: 1))
    }

    private var symbolName: String {
        if isLocked { return "stop.fill" }
        return isActive ? "waveform" : "mic.fill"
    }
}

private struct EpicRow: View {
    let item: Item
    let stats: EpicStats

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: item.type.systemImage)
                    .font(.system(size: 17))
                    .foregroundStyle(.cyan.opacity(0.7))
                    .padding(.top, 2)

                VStack(alignment: .leading, spacing: 5) {
                    Text(item.title)
                        .font(.system(.subheadline, design: .rounded, weight: .semibold))
                        .foregroundStyle(.white)
                    Text(item.previewText)
                        .font(.system(.caption, design: .serif))
                        .foregroundStyle(.white.opacity(0.5))
                        .lineLimit(2)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.2))
                    .padding(.top, 4)
            }

            HStack(spacing: 8) {
                statPill("\(stats.notes)", label: "notes")
                statPill("\(stats.todos)", label: "todos")
                if stats.openTodos > 0 {
                    statPill("\(stats.openTodos)", label: "open")
                }
            }
        }
        .padding(14)
        .background(Color.white.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(.cyan.opacity(0.14), lineWidth: 1))
    }

    private func statPill(_ value: String, label: String) -> some View {
        HStack(spacing: 3) {
            Text(value)
                .fontWeight(.semibold)
            Text(label)
        }
        .font(.system(.caption2, design: .rounded))
        .foregroundStyle(.white.opacity(0.55))
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.white.opacity(0.07))
        .clipShape(Capsule())
    }
}

private struct ItemRow: View {
    let item: Item
    var epicTitle: String? = nil
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
                Image(systemName: item.type.systemImage)
                    .font(.system(size: 16))
                    .foregroundStyle(iconColor)
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

                if let epicTitle {
                    Label(epicTitle, systemImage: "square.stack.3d.up")
                        .font(.system(.caption2, design: .rounded, weight: .semibold))
                        .foregroundStyle(.cyan.opacity(0.75))
                        .lineLimit(1)
                }

                HStack(spacing: 8) {
                    if let due = item.dueDateFormatted {
                        Label(due, systemImage: "calendar")
                            .font(.system(.caption2, design: .rounded, weight: .medium))
                            .foregroundStyle(item.isOverdue ? .red.opacity(0.85) : .yellow.opacity(0.8))
                    }
                    Text(item.timestampLabel)
                        .font(.system(.caption2, design: .rounded))
                        .foregroundStyle(.white.opacity(0.28))
                }
            }

            Spacer()

            Image(systemName: "chevron.right")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.white.opacity(0.2))
        }
        .padding(14)
        .background(Color.white.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(.white.opacity(0.08), lineWidth: 1))
    }

    private var iconColor: Color {
        switch item.type {
        case .note: return .white.opacity(0.35)
        case .todo: return .yellow
        case .epic: return .cyan.opacity(0.65)
        }
    }
}

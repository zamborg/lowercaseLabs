import SwiftUI
import SwiftData
import ControllrCore

struct ThreadsView: View {
    @Query(sort: \ChatThread.updatedAt, order: .reverse) private var threads: [ChatThread]
    @Environment(\.openURL) private var openURL
    @State private var shortcutUnavailable = false

    var body: some View {
        NavigationStack {
            List(threads) { thread in
                NavigationLink {
                    ThreadView(thread: thread)
                } label: {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(thread.title).lineLimit(2)
                        Text(thread.updatedAt, format: .dateTime.month().day().hour().minute())
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .overlay {
                if threads.isEmpty {
                    ContentUnavailableView("No threads yet", systemImage: "bubble.left.and.bubble.right",
                                           description: Text("Run the Controllr shortcut to start talking."))
                }
            }
            .navigationTitle("Threads")
            .toolbar {
                Button("Start by voice", systemImage: "mic") {
                    openURL(RouterShortcut.url()) { shortcutUnavailable = !$0 }
                }
            }
            .alert("Shortcut unavailable", isPresented: $shortcutUnavailable) {
                Button("OK", role: .cancel) {}
            } message: {
                Text("Install \(RouterShortcut.name) in Shortcuts.")
            }
        }
    }
}

private struct ThreadView: View {
    let thread: ChatThread
    @Environment(\.openURL) private var openURL
    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss
    @State private var shortcutUnavailable = false
    @State private var confirmDelete = false
    @State private var deleteFailed = false

    var body: some View {
        ScrollViewReader { scroll in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(thread.orderedEntries) { entry in
                        entryView(entry)
                    }
                    Color.clear.frame(height: 1).id("end")
                }
                .padding()
            }
            .onAppear { scroll.scrollTo("end", anchor: .bottom) }
            .onChange(of: thread.entries.count) { _, _ in
                withAnimation { scroll.scrollTo("end", anchor: .bottom) }
            }
        }
        .navigationTitle(thread.title)
        .navigationBarTitleDisplayMode(.inline)
        .safeAreaInset(edge: .bottom) {
            Button("Continue by voice", systemImage: "mic") {
                openURL(RouterShortcut.url(threadID: thread.id)) { shortcutUnavailable = !$0 }
            }
            .buttonStyle(.borderedProminent)
            .padding(10)
            .frame(maxWidth: .infinity)
            .background(.bar)
        }
        .toolbar {
            Button("Delete thread", systemImage: "trash", role: .destructive) { confirmDelete = true }
        }
        .confirmationDialog("Delete this thread?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete thread", role: .destructive) {
                guard !Conversations.shared.isBusy(thread) else { deleteFailed = true; return }
                context.delete(thread)
                do {
                    try context.save()
                    dismiss()
                } catch {
                    context.rollback()
                    deleteFailed = true
                }
            }
        }
        .alert("Shortcut unavailable", isPresented: $shortcutUnavailable) {
            Button("OK", role: .cancel) {}
        } message: {
            Text("Install \(RouterShortcut.name) in Shortcuts.")
        }
        .alert("Couldn't delete this thread", isPresented: $deleteFailed) {
            Button("OK", role: .cancel) {}
        }
    }

    @ViewBuilder
    private func entryView(_ entry: ChatEntry) -> some View {
        switch entry.message {
        case .user(let text):
            Text(text).textSelection(.enabled)
                .padding(12)
                .background(Color.accentColor.opacity(0.12), in: RoundedRectangle(cornerRadius: 14))
                .frame(maxWidth: .infinity, alignment: .trailing)
        case .assistant(let text), .followUp(_, _, let text):
            VStack(alignment: .leading, spacing: 4) {
                Text(text).textSelection(.enabled)
                if entry.actionID != nil {
                    Text(entry.actionStatus ?? "Awaiting shortcut result")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(12)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 14))
        case .tool(let call, _, let text):
            if let text {
                Text(text).textSelection(.enabled)
                    .padding(12)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 14))
            }
            if call.name == "search_saved_items" {
                Label("Searched saved items", systemImage: "magnifyingglass")
                    .font(.caption).foregroundStyle(.secondary)
            }
        case nil:
            Text("Message unavailable").foregroundStyle(.secondary)
        }
    }
}

enum RouterShortcut: String, CaseIterable {
    case legacy = "Controllr Voice 1.6"
    case modular = "Controllr Voice 2"
    case selector = "Controllr Voice 2.1"

    static let preferenceKey = "voiceShortcut"
    static var name: String {
        let saved = UserDefaults.standard.string(forKey: preferenceKey)
        return (saved.flatMap(Self.init(rawValue:)) ?? .legacy).rawValue
    }

    static func url(threadID: UUID? = nil) -> URL {
        var url = URLComponents()
        url.scheme = "shortcuts"
        url.host = "run-shortcut"
        url.queryItems = [URLQueryItem(name: "name", value: name)]
        if let threadID {
            url.queryItems?.append(contentsOf: [
                URLQueryItem(name: "input", value: "text"),
                URLQueryItem(name: "text", value: threadID.uuidString),
            ])
        }
        return url.url!
    }
}

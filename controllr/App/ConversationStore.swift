import Foundation
import SwiftData
import ControllrCore

@Model
final class ChatThread {
    @Attribute(.unique) var id: UUID
    var title: String
    var updatedAt: Date
    @Relationship(deleteRule: .cascade, inverse: \ChatEntry.thread) var entries: [ChatEntry] = []

    init(title: String) {
        self.id = UUID()
        self.title = String(title.prefix(80))
        self.updatedAt = .now
    }

    var orderedEntries: [ChatEntry] { entries.sorted { $0.position < $1.position } }
}

@Model
final class ChatEntry {
    var thread: ChatThread?
    var position: Int
    var payload: Data
    var actionID: String?
    var actionStatus: String?

    init(message: ConversationMessage, position: Int, actionID: String? = nil) throws {
        self.payload = try JSONEncoder().encode(message)
        self.position = position
        self.actionID = actionID
    }

    var message: ConversationMessage? { try? JSONDecoder().decode(ConversationMessage.self, from: payload) }

    var historyMessage: ConversationMessage? {
        guard actionID != nil, case .assistant(let summary) = message else { return message }
        return .assistant("\(summary) [\(actionStatus ?? "No execution receipt")]")
    }
}

@MainActor
final class Conversations {
    static let shared: Conversations = {
        let context = Store.container.mainContext
        return Conversations(context: context, route: routeThroughEngine)
    }()

    private static func routeThroughEngine(_ text: String, _ history: [ConversationMessage]) async -> RouteResult {
        await Engine.run(text, messages: history, source: "shortcut", execute: true)
    }

    private var activeThreads: Set<UUID> = []
    private let context: ModelContext
    private let route: @MainActor (String, [ConversationMessage]) async -> RouteResult

    init(context: ModelContext, route: @escaping @MainActor (String, [ConversationMessage]) async -> RouteResult) {
        self.context = context
        self.route = route
    }

    func turn(_ spoken: String, threadID: String?) async -> Decision {
        let transcript = spoken.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !transcript.isEmpty else { return .speak(.error, "I didn't hear anything.") }

        do {
            let thread: ChatThread
            if let threadID, !threadID.isEmpty {
                guard let found = try find(threadID) else {
                    return .speak(.error, "I couldn't find that thread.")
                }
                thread = found
            } else {
                thread = ChatThread(title: transcript)
                context.insert(thread)
            }
            guard activeThreads.insert(thread.id).inserted else {
                return .speak(.error, "That thread is already working on a request.")
            }
            defer { activeThreads.remove(thread.id) }

            try append(.user(transcript), to: thread)
            try save()
            let history = thread.orderedEntries.suffix(48).compactMap(\.historyMessage).drop {
                if case .user = $0 { return false }
                return true
            }

            let normalized = transcript.lowercased().trimmingCharacters(in: .punctuationCharacters)
            var decision: Decision
            if ["cancel", "stop", "never mind", "nevermind"].contains(normalized) {
                decision = .speak(.respond, "Okay.")
            } else {
                let result = await route(transcript, Array(history))
                for message in result.toolMessages { try append(message, to: thread) }
                decision = result.decision
            }

            if decision.action == .run { decision.actionID = UUID().uuidString }
            decision.threadID = thread.id.uuidString
            try append(.assistant(decision.summary), to: thread, actionID: decision.actionID)
            try save()
            return decision
        } catch {
            return .speak(.error, "I couldn't save this conversation.")
        }
    }

    func recordAction(threadID: String, actionID: String, completed: Bool) throws {
        guard let thread = try find(threadID),
              let entry = thread.entries.first(where: { $0.actionID == actionID }),
              entry.actionStatus == nil else { return }
        entry.actionStatus = completed ? "Shortcut step completed" : "Cancelled"
        thread.updatedAt = .now
        try save()
    }

    func isBusy(_ thread: ChatThread) -> Bool { activeThreads.contains(thread.id) }

    private func find(_ threadID: String) throws -> ChatThread? {
        guard let id = UUID(uuidString: threadID) else { return nil }
        var query = FetchDescriptor<ChatThread>(predicate: #Predicate { $0.id == id })
        query.fetchLimit = 1
        return try context.fetch(query).first
    }

    private func append(_ message: ConversationMessage, to thread: ChatThread, actionID: String? = nil) throws {
        let position = (thread.entries.map(\.position).max() ?? -1) + 1
        let entry = try ChatEntry(message: message, position: position, actionID: actionID)
        context.insert(entry)
        thread.entries.append(entry)
        thread.updatedAt = .now
    }

    private func save() throws {
        do {
            try context.save()
        } catch {
            context.rollback()
            throw error
        }
    }
}

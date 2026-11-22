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
    var actionPayload: Data?

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

private struct PreparedAction: Codable {
    let call: ToolCall
    let shortcut: String
    let routerVersion: Int?
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
    private let policy: () -> ModulePolicy
    private let executor: any ModuleExecutor

    init(context: ModelContext, policy: @escaping () -> ModulePolicy = { ModuleSettings.policy },
         executor: any ModuleExecutor = NativeExecutor(),
         route: @escaping @MainActor (String, [ConversationMessage]) async -> RouteResult) {
        self.context = context
        self.policy = policy
        self.executor = executor
        self.route = route
    }

    func turn(_ spoken: String, threadID: String?, routerVersion: Int = 1) async -> Decision {
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
            let normalized = transcript.lowercased().replacingOccurrences(of: "’", with: "'")
                .trimmingCharacters(in: .punctuationCharacters)
            if ["cancel", "stop", "never mind", "nevermind", "stop listening", "that's all", "end conversation"].contains(normalized) {
                return try speak(.respond, "Okay.", in: thread)
            }
            return try await decide(transcript, in: thread, routerVersion: routerVersion)
        } catch {
            return .speak(.error, "I couldn't save this conversation.")
        }
    }

    private func decide(_ source: String, in thread: ChatThread, routerVersion: Int) async throws -> Decision {
        let history = thread.orderedEntries.suffix(48).compactMap(\.historyMessage).drop {
            if case .user = $0 { return false }
            return true
        }
        let result = await route(source, Array(history))
        for message in result.toolMessages { try append(message, to: thread) }

        var decision = result.decision
        if decision.action == .run {
            if let call = result.toolCall, let module = Registry.v1.module(id: call.name) {
                guard policy().isEnabled(module) else {
                    return try speak(.error, "That action is turned off in Modules.", in: thread)
                }
                if routerVersion < 3 && module.id == "find_recent_messages" {
                    return try speak(.error, "Run Controllr Voice 1.5.1 for lookup by sender or conversation.", in: thread)
                }
                if routerVersion == 1 && !supportsLegacyShortcut(decision, module: module) {
                    return try speak(.error, "Run Controllr Voice 1.5 to use this action.", in: thread)
                }
            }
            decision.actionID = UUID().uuidString
        }
        decision.threadID = thread.id.uuidString
        let entry = try append(result.assistantMessage ?? .assistant(decision.summary), to: thread, actionID: decision.actionID)
        if routerVersion >= 2, decision.action == .run, let call = result.toolCall {
            entry.actionPayload = try JSONEncoder().encode(
                PreparedAction(call: call, shortcut: decision.shortcut, routerVersion: routerVersion)
            )
        }
        try save()
        return decision
    }

    func authorizeAction(threadID: String, actionID: String, approved: Bool) -> String {
        do {
            guard let (_, entry, action, module) = try preparedAction(threadID: threadID, actionID: actionID),
                  entry.actionStatus == nil else { return "That action is no longer pending." }
            guard policy().isEnabled(module) else { return "That action is turned off in Modules." }
            guard !policy().requiresConfirmation(module) || approved else { return "That action needs your confirmation." }
            // Only run the original validated action once, even if Shortcuts retries this callback.
            guard action.shortcut != Decision.nativeActionID || module.operation == .change else {
                return "That action is unavailable."
            }
            entry.actionStatus = "Started"
            try save()
            return "ready"
        } catch {
            return "I couldn't authorize that action."
        }
    }

    func completeAction(threadID: String, actionID: String, output: String?) async -> Decision {
        do {
            guard let (thread, entry, action, module) = try preparedAction(threadID: threadID, actionID: actionID),
                  entry.actionStatus == "Started", activeThreads.insert(thread.id).inserted else {
                return .speak(.error, "That action is no longer pending.")
            }
            defer { activeThreads.remove(thread.id) }

            guard policy().isEnabled(module) else {
                entry.actionStatus = "Cancelled"
                return try speak(.error, "That action is turned off in Modules.", in: thread)
            }
            if action.shortcut == Decision.nativeActionID {
                // Consume authorization before an awaited write.
                entry.actionStatus = "Executing"
                try save()
                do {
                    guard case .spoken(let text) = try await executor.execute(module, arguments: action.call.arguments.objectValue ?? [:]) else {
                        throw ExecutionError.failed("unexpected native outcome")
                    }
                    entry.actionStatus = "Completed"
                    return try speak(.done, text, in: thread)
                } catch ExecutionError.clarify(let question) {
                    entry.actionStatus = "Needs details"
                    return try speak(.clarify, question, in: thread)
                } catch ExecutionError.unavailable(let reason) {
                    entry.actionStatus = "Unavailable"
                    return try speak(.error, reason, in: thread)
                } catch {
                    entry.actionStatus = "Failed"
                    return try speak(.error, "That action didn't work.", in: thread)
                }
            }

            if module.operation == .read {
                guard let output = output ?? (module.id == "find_recent_messages" ? "" : nil) else {
                    entry.actionStatus = "Failed"
                    return try speak(.error, "The shortcut didn't return a result.", in: thread)
                }
                entry.actionStatus = "Completed"
                var text = output.trimmingCharacters(in: .whitespacesAndNewlines)
                if module.id == "find_messages",
                   let query = action.call.arguments.objectValue?["query"]?.stringValue,
                   !text.localizedCaseInsensitiveContains(query.trimmingCharacters(in: .whitespacesAndNewlines)) {
                    text = ""
                }
                if module.id == "find_recent_messages" {
                    let parts = text.split(separator: "\n", maxSplits: 1)
                    let expected = action.call.arguments.objectValue?["name"]?.stringValue?
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if parts.count == 2, let expected,
                       parts[0].localizedCaseInsensitiveCompare(expected) == .orderedSame {
                        text = String(parts[1])
                    } else {
                        text = "No matching message found among the 120 most recent messages."
                    }
                }
                try append(.tool(call: action.call, output: text.isEmpty ? "No results." : String(text.prefix(8000))), to: thread)
                try save()
                return try await decide("Shortcut result for \(module.id)", in: thread, routerVersion: action.routerVersion ?? 2)
            }

            entry.actionStatus = "Shortcut step completed"
            thread.updatedAt = .now
            try save()
            // A handoff might still need user interaction in another app.
            return .speak(.done, "")
        } catch {
            return .speak(.error, "I couldn't finish that action.")
        }
    }

    private func speak(_ action: DecisionAction, _ text: String, in thread: ChatThread) throws -> Decision {
        var decision = Decision.speak(action, text)
        decision.threadID = thread.id.uuidString
        try append(.assistant(text), to: thread)
        try save()
        return decision
    }

    private func preparedAction(threadID: String, actionID: String) throws -> (ChatThread, ChatEntry, PreparedAction, Module)? {
        guard let thread = try find(threadID), let entry = thread.orderedEntries.last,
              entry.actionID == actionID, let data = entry.actionPayload else { return nil }
        let action = try JSONDecoder().decode(PreparedAction.self, from: data)
        guard let module = Registry.v1.module(id: action.call.name) else { return nil }
        return (thread, entry, action, module)
    }

    private func supportsLegacyShortcut(_ decision: Decision, module: Module) -> Bool {
        if [Decision.openURLID, Decision.openURLConfirmID].contains(decision.shortcut) { return true }
        let legacy = ["send_message", "media_control", "start_timer", "set_alarm", "set_focus", "flashlight"]
        return legacy.contains(decision.shortcut) && decision.confirm == module.requiresConfirmation
    }

    func recordAction(threadID: String, actionID: String, completed: Bool) throws {
        guard let thread = try find(threadID),
              let entry = thread.entries.first(where: { $0.actionID == actionID }),
              entry.actionStatus == nil, entry.actionPayload == nil || !completed else { return }
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

    @discardableResult
    private func append(_ message: ConversationMessage, to thread: ChatThread, actionID: String? = nil) throws -> ChatEntry {
        let position = (thread.entries.map(\.position).max() ?? -1) + 1
        let entry = try ChatEntry(message: message, position: position, actionID: actionID)
        context.insert(entry)
        thread.entries.append(entry)
        thread.updatedAt = .now
        return entry
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

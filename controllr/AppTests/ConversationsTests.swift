import ControllrCore
import SwiftData
import XCTest
@testable import Controllr

@MainActor
final class ConversationsTests: XCTestCase {
    func testSuccessiveFollowUpsAndActionCancellationPersistInOneThread() async throws {
        let storeURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: storeURL, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: storeURL) }
        let url = storeURL.appendingPathComponent("test.store")
        var requests: [[ConversationMessage]] = []
        let sms = DecisionFixture.sendMessage
        let recipientCall = ToolCall(name: "clarify", arguments: .object(["question": .string("Who should I text?")]), id: "call_recipient")
        let bodyCall = ToolCall(name: "clarify", arguments: .object(["question": .string("What should I say?")]), id: "call_body")
        let recipientQuestion = ConversationMessage.followUp(call: recipientCall, assistantText: nil, spoken: "Who should I text?")
        let bodyQuestion = ConversationMessage.followUp(call: bodyCall, assistantText: nil, spoken: "What should I say?")

        let next: Decision = try await withStore(url: url) { context in
            let conversations = Conversations(context: context) { _, history in
                requests.append(history)
                switch requests.count {
                case 1: return Self.result(.speak(.clarify, "Who should I text?"), assistantMessage: recipientQuestion)
                case 2: return Self.result(.speak(.clarify, "What should I say?"), assistantMessage: bodyQuestion)
                default: return Self.result(sms)
                }
            }

            let firstQuestion = await conversations.turn("Send a text", threadID: nil)
            XCTAssertTrue(firstQuestion.listensAfterSpeaking)
            let id = try XCTUnwrap(firstQuestion.threadID)
            let secondQuestion = await conversations.turn("Sam", threadID: id)
            XCTAssertEqual(secondQuestion.threadID, id)
            XCTAssertTrue(secondQuestion.listensAfterSpeaking)
            let decision = await conversations.turn("Running late.", threadID: id)
            XCTAssertEqual(decision.threadID, id)
            XCTAssertEqual(decision.action, .run)
            XCTAssertFalse(decision.listensAfterSpeaking)
            XCTAssertTrue(decision.confirm)

            let actionID = try XCTUnwrap(decision.actionID)
            try conversations.recordAction(threadID: id, actionID: actionID, completed: false)
            try conversations.recordAction(threadID: id, actionID: actionID, completed: true)
            return decision
        }

        XCTAssertEqual(requests, [
            [.user("Send a text")],
            [.user("Send a text"), recipientQuestion, .user("Sam")],
            [.user("Send a text"), recipientQuestion, .user("Sam"), bodyQuestion, .user("Running late.")],
        ])
        try await withStore(url: url) { context in
            let threads = try context.fetch(FetchDescriptor<ChatThread>())
            let thread = try XCTUnwrap(threads.first)
            XCTAssertEqual(threads.count, 1)
            XCTAssertEqual(thread.id.uuidString, next.threadID)
            XCTAssertEqual(thread.orderedEntries.count, 6)
            XCTAssertEqual(thread.orderedEntries[1].message, recipientQuestion)
            XCTAssertEqual(thread.orderedEntries[3].message, bodyQuestion)
            XCTAssertEqual(thread.orderedEntries.last?.actionStatus, "Cancelled")

            let conversations = Conversations(context: context) { _, history in
                XCTAssertEqual(history.last, .user("What did you do?"))
                XCTAssertTrue(history.contains(.assistant("Text Sam: Running late. [Cancelled]")))
                return Self.result(.speak(.respond, "You cancelled the text."))
            }
            let reply = await conversations.turn("What did you do?", threadID: thread.id.uuidString)
            XCTAssertEqual(reply.summary, "You cancelled the text.")
            XCTAssertEqual(thread.orderedEntries.count, 8)
        }
    }

    func testBlankUnknownAndCancelledTurnsNeverReachProvider() async throws {
        try await withStore { context in
            let conversations = Conversations(context: context) { _, _ in
                XCTFail("This turn should never reach the provider")
                return Self.result(.speak(.error, "unexpected"))
            }

            let empty = await conversations.turn("   ", threadID: nil)
            let unknown = await conversations.turn("Continue", threadID: UUID().uuidString)
            XCTAssertEqual(empty.action, .error)
            XCTAssertEqual(unknown.action, .error)
            XCTAssertEqual(try context.fetchCount(FetchDescriptor<ChatThread>()), 0)

            let stopped = await conversations.turn("Cancel.", threadID: nil)
            XCTAssertEqual(stopped.action, .respond)
            XCTAssertEqual(stopped.summary, "Okay.")
            XCTAssertFalse(stopped.listensAfterSpeaking)

            let ended = await conversations.turn("That’s all.", threadID: stopped.threadID)
            XCTAssertFalse(ended.listensAfterSpeaking)
            XCTAssertEqual(ended.summary, "Okay.")
        }
    }

    func testSavedMemorySkillAndNotePersistAndRemainQueryable() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("test.store")
        try await withStore(url: url) { context in
            let state = SavedItems(context: context)
            _ = try state.save(kind: .memory, title: "Gym hours", text: "The gym closes at eight.")
            _ = try state.save(kind: .memory, title: "Gym Hours", text: "The gym closes at nine.", matchByTitle: true)
            _ = try state.save(kind: .skill, title: "Meeting recap", text: "Put decisions first.")
            _ = try state.save(kind: .note, title: "Bike parts", text: "Get a 10 mm wrench.")
        }
        try await withStore(url: url) { context in
            let state = SavedItems(context: context)
            let gym = try state.search(kind: .memory, query: "my gym hours")
            let recap = try state.search(kind: .skill, query: "meeting recap")
            let bike = try state.search(kind: nil, query: "wrench")
            XCTAssertEqual(try context.fetchCount(FetchDescriptor<SavedItem>()), 3)
            XCTAssertTrue(gym.contains("closes at nine"))
            XCTAssertTrue(recap.contains("Put decisions first"))
            XCTAssertTrue(bike.contains("Bike parts"))
            XCTAssertEqual(try state.search(kind: .note, query: "the gym"), "[]")
        }
    }

    func testPreparedNativeActionRequiresApprovalAndRunsOnlyOnce() async throws {
        try await withStore { context in
            let call = ToolCall(name: "edit_calendar_event", arguments: .object([
                "title": .string("Test event"), "new_title": .string("Updated event")
            ]))
            var action = Decision.speak(.run, "Edit Test event")
            action.shortcut = Decision.nativeActionID
            action.confirm = true
            let executor = CountingExecutor()
            let conversations = Conversations(context: context, executor: executor) { _, _ in
                Self.result(action, toolCall: call)
            }

            let proposed = await conversations.turn("Rename Test event", threadID: nil, routerVersion: 2)
            let thread = try XCTUnwrap(proposed.threadID)
            let id = try XCTUnwrap(proposed.actionID)
            let premature = await conversations.completeAction(threadID: thread, actionID: id, output: nil)
            XCTAssertEqual(premature.action, .error)
            XCTAssertNotEqual(conversations.authorizeAction(threadID: thread, actionID: id, approved: false), "ready")
            XCTAssertEqual(conversations.authorizeAction(threadID: thread, actionID: id, approved: true), "ready")
            let done = await conversations.completeAction(threadID: thread, actionID: id, output: nil)
            XCTAssertEqual(done.summary, "Updated event.")
            let replay = await conversations.completeAction(threadID: thread, actionID: id, output: nil)
            XCTAssertEqual(replay.action, .error)
            let count = await executor.executions
            XCTAssertEqual(count, 1)
        }
    }

    func testShortcutReadResumesSameTurnAndKeepsListening() async throws {
        try await withStore { context in
            let call = ToolCall(name: "read_clipboard", arguments: .object([:]), id: "read_1")
            var action = Decision.speak(.run, "Read clipboard text")
            action.shortcut = call.name
            let conversations = Conversations(context: context) { _, history in
                if history == [.user("Read my clipboard")] {
                    return Self.result(action, toolCall: call)
                }
                XCTAssertEqual(history.last, .tool(call: call, output: "A test sentence."))
                XCTAssertEqual(history.compactMap { item -> String? in
                    if case .user(let text) = item { return text }
                    return nil
                }, ["Read my clipboard"])
                return Self.result(.speak(.clarify, "Should I summarize it?"))
            }

            let proposed = await conversations.turn("Read my clipboard", threadID: nil, routerVersion: 2)
            let thread = try XCTUnwrap(proposed.threadID)
            let id = try XCTUnwrap(proposed.actionID)
            XCTAssertEqual(conversations.authorizeAction(threadID: thread, actionID: id, approved: false), "ready")
            let next = await conversations.completeAction(threadID: thread, actionID: id, output: "A test sentence.")
            XCTAssertEqual(next.threadID, thread)
            XCTAssertTrue(next.listensAfterSpeaking)
        }
    }

    func testMessagesResultMustMatchRequestedScopeBeforeItReachesModel() async throws {
        try await withStore { context in
            let call = ToolCall(name: "find_recent_messages", arguments: .object([
                "scope": .string("conversation"), "name": .string("Weekend Hike")
            ]), id: "read_messages")
            var action = Decision.speak(.run, "Find the recent Weekend Hike message")
            action.shortcut = call.name
            let conversations = Conversations(context: context) { _, history in
                if history == [.user("Latest Weekend Hike message")] {
                    return Self.result(action, toolCall: call)
                }
                XCTAssertEqual(history.last, .tool(call: call, output: "No matching message found among the 120 most recent messages."))
                return Self.result(.speak(.respond, "I couldn't find a recent message."))
            }
            let proposal = await conversations.turn("Latest Weekend Hike message", threadID: nil, routerVersion: 3)
            let thread = try XCTUnwrap(proposal.threadID)
            let id = try XCTUnwrap(proposal.actionID)
            XCTAssertEqual(conversations.authorizeAction(threadID: thread, actionID: id, approved: false), "ready")
            let reply = await conversations.completeAction(threadID: thread, actionID: id, output: "Other Thread\nAn unrelated message.")
            XCTAssertEqual(reply.action, .respond)
        }
    }

    private func withStore<T: Sendable>(url: URL? = nil, body: @MainActor (ModelContext) async throws -> T) async throws -> T {
        let schema = Schema([ChatThread.self, ChatEntry.self, SavedItem.self, RunRecord.self])
        let config = url.map { ModelConfiguration(url: $0) } ?? ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: schema, configurations: config)
        return try await body(container.mainContext)
    }

    private static func result(_ decision: Decision, toolCall: ToolCall? = nil,
                               assistantMessage: ConversationMessage? = nil) -> RouteResult {
        RouteResult(decision: decision, toolCall: toolCall, latencyMs: 0, provider: "test", model: "test", error: nil,
                    assistantMessage: assistantMessage)
    }
}

private actor CountingExecutor: ModuleExecutor {
    private(set) var executions = 0

    func execute(_ module: Module, arguments: [String: JSONValue]) async throws -> ExecutionOutcome {
        executions += 1
        return .spoken("Updated event.")
    }
}

private enum DecisionFixture {
    static var sendMessage: Decision {
        var decision = Decision.speak(.run, "Text Sam: Running late.")
        decision.shortcut = "send_message"
        decision.input = ["body": .string("Running late."), "recipient": .string("+15550101010")]
        decision.confirm = true
        return decision
    }
}

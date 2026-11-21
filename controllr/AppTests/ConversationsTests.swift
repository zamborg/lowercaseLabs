import ControllrCore
import SwiftData
import XCTest
@testable import Controllr

@MainActor
final class ConversationsTests: XCTestCase {
    func testFollowUpAndActionCancellationPersistInOneThread() async throws {
        let storeURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: storeURL, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: storeURL) }
        let url = storeURL.appendingPathComponent("test.store")
        var requests: [[ConversationMessage]] = []
        let sms = DecisionFixture.sendMessage

        let next: Decision = try await withStore(url: url) { context in
            let conversations = Conversations(context: context) { _, history in
                requests.append(history)
                let decision = requests.count == 1 ? Decision.speak(.clarify, "What should I say?") : sms
                return Self.result(decision)
            }

            let question = await conversations.turn("Text Sam", threadID: nil)
            XCTAssertEqual(question.action, .clarify)
            let id = try XCTUnwrap(question.threadID)
            let decision = await conversations.turn("Running late.", threadID: id)
            XCTAssertEqual(decision.threadID, id)
            XCTAssertEqual(decision.action, .run)
            XCTAssertTrue(decision.confirm)

            let actionID = try XCTUnwrap(decision.actionID)
            try conversations.recordAction(threadID: id, actionID: actionID, completed: false)
            try conversations.recordAction(threadID: id, actionID: actionID, completed: true)
            return decision
        }

        XCTAssertEqual(requests, [
            [.user("Text Sam")],
            [.user("Text Sam"), .assistant("What should I say?"), .user("Running late.")],
        ])
        try await withStore(url: url) { context in
            let threads = try context.fetch(FetchDescriptor<ChatThread>())
            let thread = try XCTUnwrap(threads.first)
            XCTAssertEqual(threads.count, 1)
            XCTAssertEqual(thread.id.uuidString, next.threadID)
            XCTAssertEqual(thread.orderedEntries.count, 4)
            XCTAssertEqual(thread.orderedEntries.last?.actionStatus, "Cancelled")

            let conversations = Conversations(context: context) { _, history in
                XCTAssertEqual(history.last, .user("What did you do?"))
                XCTAssertTrue(history.contains(.assistant("Text Sam: Running late. [Cancelled]")))
                return Self.result(.speak(.respond, "You cancelled the text."))
            }
            let reply = await conversations.turn("What did you do?", threadID: thread.id.uuidString)
            XCTAssertEqual(reply.summary, "You cancelled the text.")
            XCTAssertEqual(thread.orderedEntries.count, 6)
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

    private func withStore<T: Sendable>(url: URL? = nil, body: @MainActor (ModelContext) async throws -> T) async throws -> T {
        let schema = Schema([ChatThread.self, ChatEntry.self, SavedItem.self, RunRecord.self])
        let config = url.map { ModelConfiguration(url: $0) } ?? ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: schema, configurations: config)
        return try await body(container.mainContext)
    }

    private static func result(_ decision: Decision) -> RouteResult {
        RouteResult(decision: decision, toolCall: nil, latencyMs: 0, provider: "test", model: "test", error: nil)
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

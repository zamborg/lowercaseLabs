import Contacts
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

    func testTextRecipientChoiceResumesDraftAfterReopeningStore() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("test.store")
        let call = ToolCall(name: "send_message", arguments: .object([
            "recipient": .string("Sam Lee"), "body": .string("Running late."), "recipient_prompt": .string("menu")
        ]))
        let choice = Self.textRecipientChoice
        let id: String = try await withStore(url: url) { context in
            let conversations = Conversations(context: context, hasContactsAccess: { true }) { _, _ in
                Self.result(.speak(.clarify, choice.question), toolCall: call, recipientChoice: choice)
            }
            let first = await conversations.turn("Text Sam Lee that I'm running late", threadID: nil, routerVersion: 5)
            XCTAssertEqual(first.options, choice.menuOptions)
            XCTAssertFalse(first.listensAfterSpeaking)
            XCTAssertNil(first.actionID)
            return try XCTUnwrap(first.threadID)
        }
        try await withStore(url: url) { context in
            let conversations = Conversations(context: context, policy: { .init() }, hasContactsAccess: { true }) { _, _ in
                XCTFail("A reply to the recipient prompt should use the saved draft")
                return Self.result(.speak(.error, "unexpected"))
            }
            let proposed = await conversations.turn(choice.menuOptions[1], threadID: id, routerVersion: 5, selectedOption: true)
            XCTAssertEqual(proposed.threadID, id)
            XCTAssertEqual(proposed.shortcut, "send_message")
            XCTAssertEqual(proposed.input["recipient"], .string("+12025550131"))
            XCTAssertEqual(proposed.input["body"], .string("Running late."))
            XCTAssertTrue(proposed.summary.contains("0131"))
            XCTAssertTrue(proposed.confirm)
            let action = try XCTUnwrap(proposed.actionID)
            XCTAssertEqual(conversations.authorizeAction(threadID: id, actionID: action, approved: false),
                           "That action needs your confirmation.")
            try conversations.recordAction(threadID: id, actionID: action, completed: false)
            let thread = try XCTUnwrap(context.fetch(FetchDescriptor<ChatThread>()).first)
            XCTAssertNil(thread.orderedEntries[1].actionPayload)
            XCTAssertEqual(thread.orderedEntries.last?.actionStatus, "Cancelled")
        }
    }

    func testSelectedAnswerResumesTheOriginalFollowUpAndOlderRoutersSpeakOptions() async throws {
        try await withStore { context in
            let args: JSONValue = .object(["question": .string("What should happen?"), "options": .array([.string("Pause"), .string("Stop")])])
            let call = ToolCall(name: "clarify", arguments: args, id: "follow_up")
            var question = Decision.speak(.clarify, "What should happen?")
            question.options = ["Pause", "Stop"]
            let followUp = ConversationMessage.followUp(call: call, assistantText: nil, spoken: question.spokenQuestion)
            let conversations = Conversations(context: context) { _, history in
                if history.last == .user("Stop") {
                    XCTAssertEqual(history.dropLast().last, followUp)
                    return Self.result(.speak(.respond, "You chose Stop."))
                }
                return Self.result(question, toolCall: call, assistantMessage: followUp)
            }
            let old = await conversations.turn("Give me a choice", threadID: nil, routerVersion: 4)
            XCTAssertTrue(old.listensAfterSpeaking)
            XCTAssertTrue(old.summary.contains("Pause, Stop"))

            let offered = await conversations.turn("Give me a choice", threadID: nil, routerVersion: 5)
            XCTAssertEqual(offered.options, ["Pause", "Stop"])
            let selected = await conversations.turn("Stop", threadID: offered.threadID, routerVersion: 5, selectedOption: true)
            XCTAssertEqual(selected.threadID, offered.threadID)
            XCTAssertEqual(selected.summary, "You chose Stop.")
        }
    }

    func testAmbiguousTextRecipientStopsAfterOneRetryAndCanBeCancelled() async throws {
        try await withStore { context in
            let call = ToolCall(name: "send_message", arguments: .object([
                "recipient": .string("Sam Lee"), "body": .string("Running late.")
            ]))
            let choice = Self.textRecipientChoice
            var requests = 0
            let conversations = Conversations(context: context, hasContactsAccess: { true }) { _, _ in
                requests += 1
                return Self.result(.speak(.clarify, choice.question), toolCall: call, recipientChoice: choice)
            }
            let first = await conversations.turn("Text Sam Lee that I'm running late", threadID: nil, routerVersion: 3)
            let retry = await conversations.turn("Sam Lee", threadID: first.threadID, routerVersion: 3)
            XCTAssertTrue(retry.listensAfterSpeaking)
            XCTAssertNil(retry.actionID)
            let stop = await conversations.turn("Sam Lee", threadID: first.threadID, routerVersion: 3)
            XCTAssertFalse(stop.listensAfterSpeaking)
            XCTAssertNil(stop.actionID)

            let next = await conversations.turn("Text Sam Lee that I'm running late", threadID: nil, routerVersion: 3)
            let cancelled = await conversations.turn("Cancel", threadID: next.threadID, routerVersion: 3)
            XCTAssertEqual(cancelled.summary, "Okay.")
            XCTAssertFalse(cancelled.listensAfterSpeaking)
            XCTAssertNil(cancelled.actionID)
            XCTAssertEqual(requests, 2)
        }
    }

    func testContactsKeepsDistinctNumbersOnAndAcrossDuplicateCards() {
        func contact(_ name: String, _ phones: [(String, String)]) -> CNMutableContact {
            let card = CNMutableContact()
            card.givenName = name
            card.familyName = "Lee"
            card.phoneNumbers = phones.map { label, number in
                CNLabeledValue(label: label, value: CNPhoneNumber(stringValue: number))
            }
            return card
        }
        let sam = contact("Sam", [(CNLabelPhoneNumberMobile, "+1 (202) 555-0100"), (CNLabelWork, "+12025550131")])
        let duplicate = contact("Sam", [(CNLabelPhoneNumberMobile, "+12025550100"), (CNLabelHome, "+12025550132")])
        let fuzzy = contact("Samuel", [(CNLabelPhoneNumberMobile, "+12025550133")])
        let recipients = Controllr.Contacts.textRecipients(from: [sam, duplicate, fuzzy], matching: "Sam Lee")
        XCTAssertEqual(recipients.map(\.number), ["+1 (202) 555-0100", "+12025550131", "+12025550132"])
        XCTAssertEqual(recipients[1].label, CNLabeledValue<CNPhoneNumber>.localizedString(forLabel: CNLabelWork))
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

    func testModularRouterStagesAnArbitraryTimerAndLegacyRouterKeepsClockPresets() async throws {
        try await withStore { context in
            let call = ToolCall(name: "start_timer", arguments: .object(["seconds": .number(83)]))
            var timer = Decision.speak(.run, "Start timer")
            timer.shortcut = call.name
            timer.input = ["seconds_total": .string("83")]
            let executor = CountingExecutor()
            let conversations = Conversations(context: context, policy: { .init() }, executor: executor,
                                              hasTimerAccess: { true }) { _, _ in
                Self.result(timer, toolCall: call)
            }

            let legacy = await conversations.turn("Set a timer for 83 seconds", threadID: nil, routerVersion: 3)
            XCTAssertEqual(legacy.action, .clarify)
            XCTAssertNil(legacy.actionID)

            let proposed = await conversations.turn("Set a timer for 83 seconds", threadID: nil, routerVersion: 4)
            XCTAssertEqual(proposed.shortcut, Decision.nativeActionID)
            let id = try XCTUnwrap(proposed.actionID)
            let thread = try XCTUnwrap(proposed.threadID)
            XCTAssertEqual(conversations.authorizeAction(threadID: thread, actionID: id, approved: false), "ready")
            let completion = await conversations.completeAction(threadID: thread, actionID: id, output: nil)
            XCTAssertEqual(completion.action, .done)
            let repeated = await conversations.completeAction(threadID: thread, actionID: id, output: nil)
            XCTAssertEqual(repeated.action, .error)
            let executions = await executor.executions
            XCTAssertEqual(executions, 1)
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

    private static let textRecipientChoice = RecipientChoice(recipients: [
        TextRecipient(name: "Sam Lee", label: "mobile", number: "+12025550100"),
        TextRecipient(name: "Sam Lee", label: "work", number: "+12025550131"),
    ])

    private static func result(_ decision: Decision, toolCall: ToolCall? = nil,
                               assistantMessage: ConversationMessage? = nil, recipientChoice: RecipientChoice? = nil) -> RouteResult {
        RouteResult(decision: decision, toolCall: toolCall, latencyMs: 0, provider: "test", model: "test", error: nil,
                    assistantMessage: assistantMessage, recipientChoice: recipientChoice)
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

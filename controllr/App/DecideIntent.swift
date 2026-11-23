import AppIntents

struct DecideIntent: AppIntent {
    static let title: LocalizedStringResource = "Decide"
    static let description = IntentDescription("Adds one spoken turn to a Controllr thread and returns the next step.")
    static let openAppWhenRun = false

    @Parameter(title: "Transcript") var transcript: String?
    @Parameter(title: "Thread ID") var threadID: String?

    @Parameter(title: "Router Version") var routerVersion: Int?
    @Parameter(title: "Reply Mode") var replyMode: String?

    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let version = routerVersion ?? 1
        let decision = await Conversations.shared.turn(transcript ?? "", threadID: threadID, routerVersion: version,
                                                       selectedOption: version >= 5 && replyMode == "selected")
        return .result(value: decision.json)
    }
}

struct RecordActionIntent: AppIntent {
    static let title: LocalizedStringResource = "Record Controllr Action"
    static let description = IntentDescription("Records whether the router completed or cancelled a proposed action.")
    static let openAppWhenRun = false

    @Parameter(title: "Thread ID") var threadID: String
    @Parameter(title: "Action ID") var actionID: String
    @Parameter(title: "Completed") var completed: Bool

    func perform() async throws -> some IntentResult {
        try await Conversations.shared.recordAction(threadID: threadID, actionID: actionID, completed: completed)
        return .result()
    }
}

struct AuthorizeActionIntent: AppIntent {
    static let title: LocalizedStringResource = "Authorize Controllr Action"
    static let openAppWhenRun = false

    @Parameter(title: "Thread ID") var threadID: String
    @Parameter(title: "Action ID") var actionID: String
    @Parameter(title: "Approval") var approval: String

    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let authorization = await Conversations.shared.authorizeAction(
            threadID: threadID, actionID: actionID, approved: approval == "approved"
        )
        return .result(value: authorization)
    }
}

struct CompleteActionIntent: AppIntent {
    static let title: LocalizedStringResource = "Complete Controllr Action"
    static let openAppWhenRun = false

    @Parameter(title: "Thread ID") var threadID: String
    @Parameter(title: "Action ID") var actionID: String
    @Parameter(title: "Output") var output: String?

    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let decision = await Conversations.shared.completeAction(threadID: threadID, actionID: actionID, output: output)
        return .result(value: decision.json)
    }
}

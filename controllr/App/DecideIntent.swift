import AppIntents

struct DecideIntent: AppIntent {
    static let title: LocalizedStringResource = "Decide"
    static let description = IntentDescription("Adds one spoken turn to a Controllr thread and returns the next step.")
    static let openAppWhenRun = false

    @Parameter(title: "Transcript") var transcript: String?
    @Parameter(title: "Thread ID") var threadID: String?

    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let decision = await Conversations.shared.turn(transcript ?? "", threadID: threadID)
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

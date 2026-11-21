import AppIntents
import ControllrCore
import Foundation

/// The router shortcut's brain: transcript in, flat decision JSON out (see DECISIONS.md).
struct DecideIntent: AppIntent {
    static let title: LocalizedStringResource = "Decide"
    static let description = IntentDescription("Asks your chosen AI provider which controllr module to run for a request.")
    static let openAppWhenRun = false

    @Parameter(title: "Transcript", requestValueDialog: "What do you want to do?") var transcript: String

    @Parameter(title: "Answer") var answer: String?

    /// Up to two clarify rounds: Siri/Shortcuts speaks the question, listens, and we route again with the history.
    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        var context: [Exchange] = []
        var current = transcript
        var r = await Engine.run(current, context: context, source: "intent", execute: true)
        while r.decision.action == .clarify, context.count < 2 {
            let question = r.decision.summary
            let reply = try await $answer.requestValue(IntentDialog(stringLiteral: question))
            context.append(Exchange(user: current, question: question))
            current = reply
            r = await Engine.run(current, context: context, source: "intent", execute: true)
        }
        return .result(value: r.decision.json)
    }
}

struct ControllrShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: DecideIntent(),
            phrases: ["\(.applicationName)", "Ask \(.applicationName)", "Hey \(.applicationName)"],
            shortTitle: "controllr",
            systemImageName: "mic"
        )
    }
}

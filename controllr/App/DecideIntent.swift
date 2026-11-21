import AppIntents
import Foundation

/// The router shortcut's brain: transcript in, flat decision JSON out (see DECISIONS.md).
struct DecideIntent: AppIntent {
    static let title: LocalizedStringResource = "Decide"
    static let description = IntentDescription("Asks your chosen AI provider which controllr module to run for a request.")
    static let openAppWhenRun = false

    @Parameter(title: "Transcript", requestValueDialog: "What do you want to do?") var transcript: String

    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let r = await Engine.run(transcript, source: "intent", execute: true)
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

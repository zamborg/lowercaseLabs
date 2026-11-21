import AppIntents
import Foundation

/// Phase 0 spike. Contract JSON: flat; ALWAYS has `shortcut` and `input` (see DECISIONS.md: router has no If).
/// Spoken results run the "Controllr · Speak" module.
private func speakDecision(_ action: String, _ text: String) -> String {
    contractJSON(["action": action, "shortcut": "Controllr · Speak", "input": ["speak": text], "confirm": false, "summary": text])
}
private func contractJSON(_ dict: [String: Any]) -> String {
    let data = (try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])) ?? Data("{}".utf8)
    return String(decoding: data, as: UTF8.self)
}

/// Variant A: hard-coded decision. Proves Run Shortcut by variable name + no app switch.
struct DecideIntent: AppIntent {
    static let title: LocalizedStringResource = "Decide"
    static let description = IntentDescription("Spike: returns a hard-coded decision for the transcript.")
    static let openAppWhenRun = false

    @Parameter(title: "Transcript") var transcript: String

    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        .result(value: contractJSON([
            "action": "run",
            "shortcut": "Controllr · Spike",
            "input": ["echo": transcript],
            "confirm": false,
            "summary": "Spike module",
        ]))
    }
}

/// Variant B: reads the Keychain and makes a real HTTPS call (OpenAI list-models).
/// Proves background Keychain access + network inside an intent, and reports latency.
struct DecideNetworkIntent: AppIntent {
    static let title: LocalizedStringResource = "Decide (network spike)"
    static let description = IntentDescription("Spike: Keychain read + HTTPS call from an intent.")
    static let openAppWhenRun = false

    @Parameter(title: "Transcript") var transcript: String

    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let start = Date()
        guard let key = Keychain.get("openai"), !key.isEmpty else {
            return .result(value: speakDecision("error", "No API key in keychain."))
        }
        var req = URLRequest(url: URL(string: "https://api.openai.com/v1/models")!)
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        req.timeoutInterval = 10
        do {
            let (_, resp) = try await URLSession.shared.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
            let ms = Int(Date().timeIntervalSince(start) * 1000)
            return .result(value: speakDecision("respond", "Network ok, status \(code), \(ms) milliseconds."))
        } catch {
            return .result(value: speakDecision("error", "Network failed."))
        }
    }
}

struct ControllrShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: DecideIntent(),
            phrases: ["Decide with \(.applicationName)"],
            shortTitle: "Decide",
            systemImageName: "mic"
        )
    }
}

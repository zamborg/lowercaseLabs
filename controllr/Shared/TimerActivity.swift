import AlarmKit
import AppIntents
import Foundation

@available(iOS 26, *)
struct TimerMetadata: AlarmMetadata {}

@available(iOS 26, *)
struct StopTimerIntent: LiveActivityIntent {
    static let title: LocalizedStringResource = "Stop Controllr timer"

    @Parameter(title: "Timer ID") var timerID: String

    init() {}
    init(id: UUID) { timerID = id.uuidString }

    func perform() async throws -> some IntentResult {
        if let id = UUID(uuidString: timerID) {
            try AlarmManager.shared.stop(id: id)
        }
        return .result()
    }
}

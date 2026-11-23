import AlarmKit
import ControllrCore
import SwiftUI

enum PhoneTimers {
    static var available: Bool {
        if #available(iOS 26, *) { return true }
        return false
    }

    static var authorized: Bool {
        guard #available(iOS 26, *) else { return false }
        return AlarmManager.shared.authorizationState == .authorized
    }

    static var accessLabel: String {
        guard #available(iOS 26, *) else { return "Unavailable" }
        switch AlarmManager.shared.authorizationState {
        case .authorized: return "Allowed"
        case .denied: return "Denied"
        case .notDetermined: return "Not requested"
        @unknown default: return "Unavailable"
        }
    }

    static func request() async -> Bool {
        guard #available(iOS 26, *) else { return false }
        return (try? await AlarmManager.shared.requestAuthorization()) == .authorized
    }

    static func duration(from arguments: [String: JSONValue]) throws -> Int {
        guard !arguments.isEmpty else { throw ExecutionError.clarify("How long should I set it for?") }
        guard let minutes = Int(exactly: arguments["minutes"]?.numberValue ?? 0),
              let seconds = Int(exactly: arguments["seconds"]?.numberValue ?? 0),
              (0...120).contains(minutes), (0...7200).contains(seconds),
              (1...7200).contains(minutes * 60 + seconds) else {
            throw ExecutionError.clarify("What duration between one second and two hours should I use?")
        }
        return minutes * 60 + seconds
    }

    @MainActor
    static func start(seconds: Int) async throws -> String {
        guard #available(iOS 26, *), authorized else {
            throw ExecutionError.unavailable("Open Controllr and allow Timers in Setup.")
        }
        let id = UUID()
        let title: LocalizedStringResource = "Controllr timer"
        let alert: AlarmPresentation.Alert
        if #available(iOS 26.1, *) {
            alert = .init(title: title)
        } else {
            alert = .init(title: title, stopButton: .init(text: "Stop", textColor: .white, systemImageName: "stop.fill"))
        }
        let attributes = AlarmAttributes(
            presentation: AlarmPresentation(alert: alert, countdown: .init(title: title)),
            metadata: TimerMetadata(),
            tintColor: .orange
        )
        let configuration = AlarmManager.AlarmConfiguration.timer(
            duration: TimeInterval(seconds), attributes: attributes, stopIntent: StopTimerIntent(id: id)
        )
        _ = try await AlarmManager.shared.schedule(id: id, configuration: configuration)
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.minute, .second]
        formatter.unitsStyle = .full
        let duration = formatter.string(from: TimeInterval(seconds)) ?? "\(seconds) seconds"
        return "Started a timer for \(duration)."
    }
}

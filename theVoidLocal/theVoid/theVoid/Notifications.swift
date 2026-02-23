import Foundation
import UserNotifications

enum ReminderWeekday: Int, CaseIterable, Identifiable {
    case monday = 2
    case tuesday = 3
    case wednesday = 4
    case thursday = 5
    case friday = 6
    case saturday = 7
    case sunday = 1

    var id: Int { rawValue }

    var shortTitle: String {
        switch self {
        case .monday: return "Mon"
        case .tuesday: return "Tue"
        case .wednesday: return "Wed"
        case .thursday: return "Thu"
        case .friday: return "Fri"
        case .saturday: return "Sat"
        case .sunday: return "Sun"
        }
    }

    var title: String {
        switch self {
        case .monday: return "Monday"
        case .tuesday: return "Tuesday"
        case .wednesday: return "Wednesday"
        case .thursday: return "Thursday"
        case .friday: return "Friday"
        case .saturday: return "Saturday"
        case .sunday: return "Sunday"
        }
    }

    static var ordered: [ReminderWeekday] {
        [.monday, .tuesday, .wednesday, .thursday, .friday, .saturday, .sunday]
    }
}

// MARK: - Notification scheduler

enum NotificationScheduler {
    struct WeeklyReminder: Hashable {
        let weekday: ReminderWeekday
        let time: Date
    }

    private static let legacyIdentifier = "void.daily.checkin"
    private static let weeklyIdentifierPrefix = "void.weekly.checkin."

    private static var weeklyIdentifiers: [String] {
        ReminderWeekday.ordered.map { "\(weeklyIdentifierPrefix)\($0.rawValue)" }
    }

    static func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, _ in
                continuation.resume(returning: granted)
            }
        }
    }

    static func scheduleDaily(at date: Date) async throws {
        try await scheduleWeekly(
            reminders: ReminderWeekday.ordered.map { WeeklyReminder(weekday: $0, time: date) }
        )
    }

    static func scheduleWeekly(reminders: [WeeklyReminder]) async throws {
        let center = UNUserNotificationCenter.current()
        center.removePendingNotificationRequests(withIdentifiers: [legacyIdentifier] + weeklyIdentifiers)

        for reminder in reminders {
            let content = UNMutableNotificationContent()
            content.title = "Step Into the Void"
            content.body = "Your one intentional check-in is ready."
            content.sound = .default

            let hm = Calendar.current.dateComponents([.hour, .minute], from: reminder.time)
            var components = DateComponents()
            components.weekday = reminder.weekday.rawValue
            components.hour = hm.hour
            components.minute = hm.minute

            let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: true)
            let identifier = "\(weeklyIdentifierPrefix)\(reminder.weekday.rawValue)"
            let request = UNNotificationRequest(identifier: identifier, content: content, trigger: trigger)

            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                center.add(request) { error in
                    if let error {
                        continuation.resume(throwing: error)
                    } else {
                        continuation.resume(returning: ())
                    }
                }
            }
        }
    }

    static func authorizationStatus() async -> UNAuthorizationStatus {
        await withCheckedContinuation { continuation in
            UNUserNotificationCenter.current().getNotificationSettings { settings in
                continuation.resume(returning: settings.authorizationStatus)
            }
        }
    }
}

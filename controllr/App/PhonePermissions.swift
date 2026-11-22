import Contacts
import ControllrCore
import EventKit

extension ModulePermission {
    var title: String {
        switch self {
        case .calendar: "Calendar"
        case .contacts: "Contacts"
        case .reminders: "Reminders"
        case .shortcuts: "Shortcuts"
        }
    }

    var accessLabel: String {
        switch self {
        case .calendar: return Self.label(for: EKEventStore.authorizationStatus(for: .event))
        case .contacts:
            let status = CNContactStore.authorizationStatus(for: .contacts)
            if #available(iOS 18, *), status == .limited { return "Limited" }
            switch status {
            case .authorized: return "Allowed"
            case .denied: return "Denied"
            case .restricted: return "Restricted"
            default: return "Not requested"
            }
        case .reminders: return Self.label(for: EKEventStore.authorizationStatus(for: .reminder))
        case .shortcuts: return "Managed by Shortcuts"
        }
    }

    private static func label(for status: EKAuthorizationStatus) -> String {
        switch status {
        case .fullAccess: "Full access"
        case .writeOnly: "Write only"
        case .denied: "Denied"
        case .restricted: "Restricted"
        default: "Not requested"
        }
    }
}

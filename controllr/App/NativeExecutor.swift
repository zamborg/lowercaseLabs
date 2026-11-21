import Foundation
import Contacts
import EventKit
import ControllrCore

/// Native modules and contact resolution. Permissions are requested from the Setup screen, never here.
struct NativeExecutor: ModuleExecutor {
    func prepare(_ module: Module, arguments: [String: JSONValue]) async throws -> [String: JSONValue] {
        if module.id == "start_timer" {
            var out = arguments
            let m = Int(arguments["minutes"]?.numberValue ?? 0), sec = Int(arguments["seconds"]?.numberValue ?? 0)
            out["seconds_total"] = .string(String(max(1, m * 60 + sec)))
            return out
        }
        guard module.id == "send_message", let who = arguments["recipient"]?.stringValue else { return arguments }
        let c = try Contacts.resolve(who)
        var out = arguments
        out["recipient"] = .string(c.number)
        out["name"] = .string(c.name)
        return out
    }

    func execute(_ module: Module, arguments a: [String: JSONValue]) async throws -> ExecutionOutcome {
        switch module.id {
        case "call_contact":
            let c = try Contacts.resolve(a["recipient"]?.stringValue ?? "")
            let digits = c.number.filter { $0.isNumber || $0 == "+" }
            return .openURL("tel:\(digits)", summary: "Call \(c.name)")
        case "add_reminder":
            return .spoken(try await Reminders.add(title: a["title"]?.stringValue ?? "", due: a["due"]?.stringValue))
        case "directions":
            let dest = try Places.resolve(a["destination"]?.stringValue ?? "")
            let flag = ["drive": "d", "walk": "w", "transit": "r"][a["mode"]?.stringValue ?? "drive"] ?? "d"
            return .openURL("https://maps.apple.com/?daddr=\(Places.encode(dest))&dirflg=\(flag)", summary: "Directions to \(dest)")
        case "request_ride":
            let dest = try Places.resolve(a["destination"]?.stringValue ?? "")
            // Uber universal link; format unverified against current docs (see DECISIONS.md).
            return .openURL("https://m.uber.com/ul/?action=setPickup&pickup=my_location&dropoff%5Bformatted_address%5D=\(Places.encode(dest))",
                            summary: "Uber to \(dest)")
        default:
            throw ExecutionError.failed("no native handler for \(module.id)")
        }
    }
}

enum Places {
    static func resolve(_ spoken: String) throws -> String {
        let s = spoken.trimmingCharacters(in: .whitespaces)
        if s.lowercased() == "home" {
            let home = Settings.homeAddress
            guard !home.isEmpty else { throw ExecutionError.clarify("Set your home address in controllr first.") }
            return home
        }
        return s
    }
    static func encode(_ s: String) -> String {
        s.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed.subtracting(CharacterSet(charactersIn: "&=+#"))) ?? s
    }
}

enum Contacts {
    struct Match { let name: String; let number: String }

    static var authorized: Bool { CNContactStore.authorizationStatus(for: .contacts) == .authorized }

    static func resolve(_ spoken: String) throws -> Match {
        guard authorized else { throw ExecutionError.clarify("Open controllr and allow contacts access.") }
        let query = Settings.nicknames[spoken.lowercased()] ?? spoken
        let store = CNContactStore()
        let keys = [CNContactGivenNameKey, CNContactFamilyNameKey, CNContactPhoneNumbersKey] as [CNKeyDescriptor]
        var found: [CNContact]
        do { found = try store.unifiedContacts(matching: CNContact.predicateForContacts(matchingName: query), keysToFetch: keys) }
        catch { throw ExecutionError.failed("contacts lookup failed") }
        found = found.filter { !$0.phoneNumbers.isEmpty }
        func full(_ c: CNContact) -> String { "\(c.givenName) \(c.familyName)".trimmingCharacters(in: .whitespaces) }
        if found.count > 1, let exact = found.first(where: { full($0).lowercased() == query.lowercased() }), found.filter({ full($0).lowercased() == query.lowercased() }).count == 1 {
            found = [exact]
        }
        switch found.count {
        case 0: throw ExecutionError.clarify("I couldn't find \(spoken) in your contacts.")
        case 1:
            let c = found[0]
            let phones = c.phoneNumbers
            let mobile = phones.first { [CNLabelPhoneNumberMobile, CNLabelPhoneNumberiPhone].contains($0.label ?? "") } ?? phones[0]
            return Match(name: full(c), number: mobile.value.stringValue)
        default:
            throw ExecutionError.clarify("Which \(spoken)? I found \(found.prefix(3).map(full).joined(separator: ", ")).")
        }
    }

    static func request() async -> Bool { (try? await CNContactStore().requestAccess(for: .contacts)) ?? false }
}

enum Reminders {
    static var authorized: Bool { EKEventStore.authorizationStatus(for: .reminder) == .fullAccess }

    static func request() async -> Bool { (try? await EKEventStore().requestFullAccessToReminders()) ?? false }

    static func add(title: String, due: String?) async throws -> String {
        guard authorized else { throw ExecutionError.clarify("Open controllr and allow reminders access.") }
        let store = EKEventStore()
        let r = EKReminder(eventStore: store)
        r.title = title
        r.calendar = store.defaultCalendarForNewReminders()
        var when: Date?
        if let due, !due.isEmpty {
            let f = ISO8601DateFormatter()
            when = f.date(from: due) ?? { f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]; return f.date(from: due) }()
            if when == nil { // tolerate a missing zone offset
                let g = DateFormatter(); g.locale = Locale(identifier: "en_US_POSIX"); g.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"; when = g.date(from: due)
            }
        }
        if let when {
            r.dueDateComponents = Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: when)
            r.addAlarm(EKAlarm(absoluteDate: when))
        }
        do { try store.save(r, commit: true) } catch { throw ExecutionError.failed("reminder save failed") }
        if let when {
            return "Okay, I'll remind you to \(title) \(when.formatted(.relative(presentation: .named))) at \(when.formatted(date: .omitted, time: .shortened))."
        }
        return "Okay, reminder added: \(title)."
    }
}

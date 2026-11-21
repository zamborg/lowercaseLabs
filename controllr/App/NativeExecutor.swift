import Foundation
import Contacts
import EventKit
import ControllrCore

/// Native modules and contact resolution. Permissions are requested from the Setup screen, never here.
struct NativeExecutor: ModuleExecutor {
    func prepare(_ module: Module, arguments: [String: JSONValue]) async throws -> [String: JSONValue] {
        if module.id == "start_timer" {
            let minutes = Int(exactly: arguments["minutes"]?.numberValue ?? 0) ?? 0
            let seconds = Int(exactly: arguments["seconds"]?.numberValue ?? 0) ?? 0
            guard !arguments.isEmpty else { throw ExecutionError.clarify("How long should I set it for?") }
            guard (0...120).contains(minutes), (0...7200).contains(seconds),
                  module.supportedSeconds?.contains(minutes * 60 + seconds) == true else {
                throw ExecutionError.clarify("This shortcut can set common timer lengths, like 30 seconds or 5 minutes. What duration should I use?")
            }
            var prepared = arguments
            prepared["seconds_total"] = .string(String(minutes * 60 + seconds))
            return prepared
        }
        if module.id == "flashlight" {
            var prepared = arguments
            prepared["setting"] = .string(arguments["on"] == .bool(true) ? "On" : "Off")
            return prepared
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
        case "save_item":
            guard let kind = a["kind"]?.stringValue.flatMap(SavedItemKind.init(rawValue:)), kind != .note,
                  let title = a["title"]?.stringValue, let text = a["text"]?.stringValue else {
                throw ExecutionError.failed("invalid saved item")
            }
            let saved = try await SavedItems.shared.save(kind: kind, title: title, text: text, matchByTitle: true)
            return .spoken(kind == .memory ? "I'll remember that." : "Saved the skill \(saved).")
        case "search_saved_items":
            let kind = a["kind"]?.stringValue.flatMap(SavedItemKind.init(rawValue:))
            return .toolResult(try await SavedItems.shared.search(kind: kind, query: a["query"]?.stringValue ?? ""))
        case "create_note":
            guard let text = a["text"]?.stringValue else { throw ExecutionError.failed("missing note text") }
            let title = a["title"]?.stringValue ?? String((text.split(whereSeparator: \.isNewline).first ?? "Note").prefix(70))
            let saved = try await SavedItems.shared.save(kind: .note, title: title, text: text)
            return .spoken("Saved the note \(saved).")
        case "create_bear_note":
            // Bear's URL scheme; the router opens it. First line becomes the title.
            let text = a["text"]?.stringValue ?? ""
            let enc = Places.encode(text)
            return .openURL("bear://x-callback-url/create?text=\(enc)&open_note=no&show_window=no&new_window=no", summary: "Note: \(text)")
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
            guard !home.isEmpty else { throw ExecutionError.clarify("What address should I use for home?") }
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

    static var authorized: Bool {
        let status = CNContactStore.authorizationStatus(for: .contacts)
        if #available(iOS 18.0, *), status == .limited { return true }
        return status == .authorized
    }

    static func resolve(_ spoken: String) throws -> Match {
        guard authorized else { throw ExecutionError.unavailable("Open Controllr and allow Contacts access.") }
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
        case 0: throw ExecutionError.clarify("I couldn't find \(spoken) in your contacts. What name should I look for?")
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
        guard authorized else { throw ExecutionError.unavailable("Open Controllr and allow Reminders access.") }
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

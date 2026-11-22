import Foundation
import Contacts
import EventKit
import ControllrCore

/// Native modules and contact resolution. Permissions are requested from the Setup screen, never here.
struct NativeExecutor: ModuleExecutor {
    private func checkEnabled(_ module: Module) throws {
        guard ModuleSettings.policy.isEnabled(module) else {
            throw ExecutionError.unavailable("That action is turned off in Modules.")
        }
    }

    func prepare(_ module: Module, arguments: [String: JSONValue]) async throws -> [String: JSONValue] {
        try checkEnabled(module)
        if ["find_messages", "find_recent_messages"].contains(module.id) {
            guard #available(iOS 27, *) else {
                throw ExecutionError.unavailable("Messages lookup in Controllr requires iOS 27.")
            }
            if module.id == "find_messages" {
                let query = arguments["query"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                guard query.count >= 3 else { throw ExecutionError.clarify("What phrase should I look for in Messages?") }
                return ["query": .string(query)]
            }
            let name = arguments["name"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            var prepared = arguments
            prepared["name"] = .string(name)
            return prepared
        }
        if module.id == "set_brightness" || module.id == "set_volume" {
            guard let level = arguments["percent"]?.numberValue, (0...100).contains(level) else {
                throw ExecutionError.clarify("What percentage between zero and one hundred?")
            }
            var prepared = arguments
            prepared["level"] = .number(level / 100)
            return prepared
        }
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
        let recipients = try Contacts.textRecipients(who)
        guard let recipient = recipients.first else {
            throw ExecutionError.clarify("I couldn't find a phone number for \(who). What name should I look for?")
        }
        if recipients.count > 1 {
            throw ExecutionError.chooseTextRecipient(RecipientChoice(recipients: recipients))
        }
        var out = arguments
        out["recipient"] = .string(recipient.number)
        out["name"] = .string(recipient.name)
        return out
    }

    func execute(_ module: Module, arguments a: [String: JSONValue]) async throws -> ExecutionOutcome {
        try checkEnabled(module)
        switch module.id {
        case "call_contact":
            let c = try Contacts.resolve(a["recipient"]?.stringValue ?? "")
            let digits = c.number.filter { $0.isNumber || $0 == "+" }
            return .openURL("tel:\(digits)", summary: "Call \(c.name)")
        case "add_reminder":
            return .spoken(try await Reminders.add(title: a["title"]?.stringValue ?? "", due: a["due"]?.stringValue))
        case "list_reminders":
            return .toolResult(try await Reminders.list())
        case "complete_reminder", "reschedule_reminder":
            return .spoken(try await Reminders.change(a, complete: module.id == "complete_reminder"))
        case "calendar_agenda", "calendar_availability":
            return .toolResult(try PhoneCalendar.agenda(from: a["start"]?.stringValue ?? "", to: a["end"]?.stringValue,
                                                       availability: module.id == "calendar_availability"))
        case "add_calendar_event":
            return .spoken(try PhoneCalendar.add(title: a["title"]?.stringValue ?? "", start: a["start"]?.stringValue ?? "",
                                                 end: a["end"]?.stringValue, location: a["location"]?.stringValue))
        case "edit_calendar_event":
            return .spoken(try PhoneCalendar.edit(a))
        case "request_lyft":
            let destination = try Places.resolve(a["destination"]?.stringValue ?? "")
            guard let shortcut = UserDefaults.standard.string(forKey: "lyftShortcut"), !shortcut.isEmpty else {
                throw ExecutionError.unavailable("Set your Lyft shortcut name in Modules.")
            }
            var url = URLComponents()
            url.scheme = "shortcuts"
            url.host = "run-shortcut"
            url.queryItems = [URLQueryItem(name: "name", value: shortcut), URLQueryItem(name: "input", value: "text"),
                              URLQueryItem(name: "text", value: destination)]
            guard let link = url.string else { throw ExecutionError.failed("invalid shortcut name") }
            return .openURL(link, summary: "Lyft to \(destination)")
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
        let keys = [CNContactGivenNameKey, CNContactFamilyNameKey, CNContactPhoneNumbersKey] as [CNKeyDescriptor]
        let (query, contacts) = try lookup(spoken, keys: keys)
        var found = contacts.filter { !$0.phoneNumbers.isEmpty }
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

    static func textRecipients(_ spoken: String) throws -> [TextRecipient] {
        let keys = [CNContactFormatter.descriptorForRequiredKeys(for: .fullName), CNContactPhoneNumbersKey as CNKeyDescriptor]
        let (query, contacts) = try lookup(spoken, keys: keys)
        return textRecipients(from: contacts, matching: query)
    }

    static func textRecipients(from contacts: [CNContact], matching query: String) -> [TextRecipient] {
        let candidates = contacts.filter { !$0.phoneNumbers.isEmpty }.map { contact in
            (name: CNContactFormatter.string(from: contact, style: .fullName) ?? query, phones: contact.phoneNumbers)
        }
        let exact = candidates.filter { $0.name.localizedCaseInsensitiveCompare(query) == .orderedSame }
        return TextRecipient.unique((exact.isEmpty ? candidates : exact).flatMap { candidate in
            candidate.phones.map { phone in
                TextRecipient(name: candidate.name, label: phone.label.map(CNLabeledValue<CNPhoneNumber>.localizedString(forLabel:)),
                              number: phone.value.stringValue)
            }
        })
    }

    private static func lookup(_ spoken: String, keys: [CNKeyDescriptor]) throws -> (String, [CNContact]) {
        guard authorized else { throw ExecutionError.unavailable("Open Controllr and allow Contacts access.") }
        let name = spoken.trimmingCharacters(in: .whitespacesAndNewlines)
        let query = Settings.nicknames[name.lowercased()] ?? name
        let predicate = CNContact.predicateForContacts(matchingName: query)
        do { return (query, try CNContactStore().unifiedContacts(matching: predicate, keysToFetch: keys)) }
        catch { throw ExecutionError.failed("contacts lookup failed") }
    }

    static func request() async -> Bool { (try? await CNContactStore().requestAccess(for: .contacts)) ?? false }
}

enum Reminders {
    static var authorized: Bool { EKEventStore.authorizationStatus(for: .reminder) == .fullAccess }

    static func request() async -> Bool { (try? await EKEventStore().requestFullAccessToReminders()) ?? false }

    static var writableLists: [CalendarChoice] {
        guard authorized else { return [] }
        return PhoneCalendar.calendarChoices(EKEventStore().calendars(for: .reminder))
    }

    @MainActor
    private static func incomplete(_ store: EKEventStore) async -> [EKReminder] {
        let predicate = store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: nil)
        let ids = await withCheckedContinuation { continuation in
            store.fetchReminders(matching: predicate) { reminders in
                continuation.resume(returning: (reminders ?? []).map(\.calendarItemIdentifier))
            }
        }
        return ids.compactMap { store.calendarItem(withIdentifier: $0) as? EKReminder }
    }

    @MainActor
    static func list() async throws -> String {
        guard authorized else { throw ExecutionError.unavailable("Open Controllr and allow Reminders access.") }
        let store = EKEventStore()
        let reminders = await incomplete(store)
        let ordered = reminders.sorted {
            ($0.dueDateComponents?.date ?? .distantFuture) < ($1.dueDateComponents?.date ?? .distantFuture)
        }
        let rows = ordered.prefix(30).map { reminder in
            JSONValue.object(["id": .string(reminder.calendarItemIdentifier), "title": .string(reminder.title ?? "Untitled"),
                              "list": .string(reminder.calendar.title),
                              "due": reminder.dueDateComponents?.date.map { .string(ScheduleDate.format($0)) } ?? .null])
        }
        return JSONValue.object(["reminders": .array(Array(rows)), "total": .number(Double(reminders.count))]).jsonString()
    }

    @MainActor
    static func change(_ arguments: [String: JSONValue], complete: Bool) async throws -> String {
        guard authorized else { throw ExecutionError.unavailable("Open Controllr and allow Reminders access.") }
        let store = EKEventStore()
        let reminder: EKReminder
        if let id = arguments["reminder_id"]?.stringValue,
           let selected = store.calendarItem(withIdentifier: id) as? EKReminder, !selected.isCompleted {
            reminder = selected
        } else {
            guard let title = arguments["title"]?.stringValue, !title.isEmpty else {
                throw ExecutionError.clarify("Which reminder?")
            }
            let list = arguments["list"]?.stringValue
            let matches = await incomplete(store).filter { reminder in
                reminder.title?.localizedCaseInsensitiveCompare(title) == .orderedSame &&
                    (list.map { reminder.calendar.title.localizedCaseInsensitiveCompare($0) == .orderedSame } ?? true)
            }
            guard matches.count == 1, let found = matches.first else {
                let question = matches.isEmpty ? "I couldn't find \(title). Which reminder?" : "Which list is \(title) in?"
                throw ExecutionError.clarify(question)
            }
            reminder = found
        }
        if complete {
            reminder.isCompleted = true
        } else {
            guard let due = arguments["due"]?.stringValue else { throw ExecutionError.clarify("When should it be due?") }
            _ = try setDue(due, on: reminder)
        }
        try store.save(reminder, commit: true)
        return complete ? "Completed \(reminder.title ?? "the reminder")." : "Rescheduled \(reminder.title ?? "the reminder")."
    }

    static func add(title: String, due: String?) async throws -> String {
        guard authorized else { throw ExecutionError.unavailable("Open Controllr and allow Reminders access.") }
        let store = EKEventStore()
        let r = EKReminder(eventStore: store)
        r.title = title
        let preferred = UserDefaults.standard.string(forKey: "preferredReminderList")
        guard let list = preferred.flatMap(store.calendar(withIdentifier:)) ?? store.defaultCalendarForNewReminders(),
              list.allowsContentModifications else {
            throw ExecutionError.unavailable("I couldn't find a writable reminder list.")
        }
        r.calendar = list
        let when = try due.flatMap { $0.isEmpty ? nil : try setDue($0, on: r) }
        do { try store.save(r, commit: true) } catch { throw ExecutionError.failed("reminder save failed") }
        if let when {
            if due?.count == 10 {
                return "Okay, I'll remind you to \(title) \(when.formatted(.relative(presentation: .named)))."
            }
            return "Okay, I'll remind you to \(title) \(when.formatted(.relative(presentation: .named))) at \(when.formatted(date: .omitted, time: .shortened))."
        }
        return "Okay, reminder added: \(title)."
    }

    private static func setDue(_ due: String, on reminder: EKReminder) throws -> Date {
        let date = try ScheduleDate.parse(due)
        if due.count == 10 {
            reminder.dueDateComponents = Calendar.current.dateComponents([.year, .month, .day], from: date)
            reminder.alarms = nil
        } else {
            reminder.dueDateComponents = Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: date)
            reminder.alarms = [EKAlarm(absoluteDate: date)]
        }
        return date
    }
}

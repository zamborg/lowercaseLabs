import ControllrCore
import EventKit
import Foundation

struct CalendarChoice: Identifiable {
    let id: String
    let title: String
}

enum PhoneCalendar {
    static var authorized: Bool { EKEventStore.authorizationStatus(for: .event) == .fullAccess }

    static var writableCalendars: [CalendarChoice] {
        guard authorized else { return [] }
        return calendarChoices(EKEventStore().calendars(for: .event))
    }

    static func request() async -> Bool { (try? await EKEventStore().requestFullAccessToEvents()) ?? false }

    static func calendarChoices(_ calendars: [EKCalendar]) -> [CalendarChoice] {
        calendars.filter(\.allowsContentModifications).map { CalendarChoice(id: $0.calendarIdentifier, title: $0.title) }
    }

    private static func store() throws -> EKEventStore {
        guard authorized else { throw ExecutionError.unavailable("Open Controllr and allow Calendar access.") }
        return EKEventStore()
    }

    static func agenda(from start: String, to end: String?, availability: Bool) throws -> String {
        let store = try store()
        let beginning = try ScheduleDate.parse(start)
        let finish = try end.map(ScheduleDate.parse) ?? Calendar.current.date(byAdding: .day, value: 1, to: beginning)!
        guard finish > beginning, finish.timeIntervalSince(beginning) <= 32 * 86_400 else {
            throw ExecutionError.clarify("What shorter date range should I check?")
        }
        let events = store.events(matching: store.predicateForEvents(withStart: beginning, end: finish, calendars: nil))
            .filter { $0.status != .canceled && (!availability || $0.availability != .free) }
            .sorted { $0.startDate < $1.startDate }
        let rows = events.prefix(30).map { event in
            JSONValue.object([
                "id": .string(event.eventIdentifier),
                "title": .string(event.title ?? "Untitled"),
                "start": .string(ScheduleDate.format(event.startDate)),
                "end": .string(ScheduleDate.format(event.endDate)),
                "calendar": .string(event.calendar.title),
                "all_day": .bool(event.isAllDay),
            ])
        }
        return JSONValue.object([
            "events": .array(Array(rows)),
            "total": .number(Double(events.count)),
            "available": availability ? .bool(events.isEmpty) : .null,
        ]).jsonString()
    }

    static func add(title: String, start: String, end: String?, location: String?) throws -> String {
        let store = try store()
        let beginning = try ScheduleDate.parse(start)
        let finish = try end.map(ScheduleDate.parse) ?? beginning.addingTimeInterval(3600)
        guard finish > beginning else { throw ExecutionError.clarify("When should the event end?") }
        let event = EKEvent(eventStore: store)
        event.title = title
        event.location = location
        event.startDate = beginning
        event.endDate = finish
        let preferred = UserDefaults.standard.string(forKey: "preferredCalendar")
        guard let target = preferred.flatMap(store.calendar(withIdentifier:)) ?? store.defaultCalendarForNewEvents,
              target.allowsContentModifications else {
            throw ExecutionError.unavailable("I couldn't find a writable calendar.")
        }
        event.calendar = target
        try store.save(event, span: .thisEvent)
        return "Added \(title) to Calendar."
    }

    static func edit(_ arguments: [String: JSONValue]) throws -> String {
        let store = try store()
        let title = arguments["title"]?.stringValue ?? ""
        let event: EKEvent
        if let id = arguments["event_id"]?.stringValue, let selected = store.event(withIdentifier: id) {
            event = selected
        } else {
            guard !title.isEmpty else { throw ExecutionError.clarify("Which calendar event should I change?") }
            let day = try arguments["current_day"]?.stringValue.map(ScheduleDate.parse)
            let start = day.map { Calendar.current.startOfDay(for: $0) } ?? .now
            let end = Calendar.current.date(byAdding: .day, value: day == nil ? 30 : 1, to: start)!
            let matches = store.events(matching: store.predicateForEvents(withStart: start, end: end, calendars: nil))
                .filter { $0.status != .canceled && $0.title?.localizedCaseInsensitiveCompare(title) == .orderedSame }
            guard matches.count == 1, let match = matches.first else {
                if matches.isEmpty { throw ExecutionError.clarify("I couldn't find \(title). Which event and date should I use?") }
                let dates = matches.prefix(3).map { $0.startDate.formatted(date: .abbreviated, time: .shortened) }
                throw ExecutionError.clarify("Which \(title) event? I found \(dates.joined(separator: ", ")).")
            }
            event = match
        }
        guard event.calendar.allowsContentModifications else {
            throw ExecutionError.unavailable("That calendar is read only.")
        }
        let start = try arguments["new_start"]?.stringValue.map(ScheduleDate.parse)
        let end = try arguments["new_end"]?.stringValue.map(ScheduleDate.parse)
        let newTitle = arguments["new_title"]?.stringValue
        guard start != nil || end != nil || newTitle?.isEmpty == false else {
            throw ExecutionError.clarify("What should I change about this event?")
        }
        let oldDuration = event.endDate.timeIntervalSince(event.startDate)
        let finalStart = start ?? event.startDate!
        let finalEnd = end ?? (start == nil ? event.endDate! : finalStart.addingTimeInterval(oldDuration))
        guard finalEnd > finalStart else { throw ExecutionError.clarify("When should the event end?") }
        if let newTitle, !newTitle.isEmpty { event.title = newTitle }
        event.startDate = finalStart
        event.endDate = finalEnd
        try store.save(event, span: .thisEvent)
        return "Updated \(event.title ?? "the event") in Calendar."
    }
}

enum ScheduleDate {
    static func parse(_ text: String) throws -> Date {
        for options: ISO8601DateFormatter.Options in [.withInternetDateTime, [.withInternetDateTime, .withFractionalSeconds]] {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = options
            if let date = formatter.date(from: text) { return date }
        }
        for format in ["yyyy-MM-dd'T'HH:mm:ss", "yyyy-MM-dd'T'HH:mm", "yyyy-MM-dd"] {
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.calendar = Calendar(identifier: .gregorian)
            formatter.timeZone = .current
            formatter.dateFormat = format
            if let date = formatter.date(from: text) { return date }
        }
        throw ExecutionError.clarify("What date and time should I use?")
    }

    static func format(_ date: Date) -> String { date.formatted(.iso8601) }
}

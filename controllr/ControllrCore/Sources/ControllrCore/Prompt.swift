import Foundation

public struct RouterSettings: Sendable {
    public var homeAddress: String?
    /// Spoken nickname -> contact name, e.g. "mom" -> "Linda Smith".
    public var nicknames: [String: String]
    public init(homeAddress: String? = nil, nicknames: [String: String] = [:]) {
        self.homeAddress = homeAddress; self.nicknames = nicknames
    }
}

public enum Prompt {
    public static func system(now: Date = Date(), timeZone: TimeZone = .current,
                              locale: Locale = .current, settings: RouterSettings = .init()) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = timeZone
        f.dateFormat = "EEEE, yyyy-MM-dd HH:mm:ss xxx"
        var lines = [
            "You are the router for a voice assistant on the user's iPhone.",
            "Current date and time: \(f.string(from: now)) (\(timeZone.identifier)). Locale: \(locale.identifier).",
            "Pick exactly one tool. Prefer clarify over guessing when a recipient, destination or other required detail is missing or unclear.",
            "Do not clarify when the detail was given: a name, a relationship word (my wife, mom, dad), a place name or a landmark (the airport, home) is enough. The app resolves contacts and places itself, so pass the words through as spoken.",
            "Clarify only when a required detail is absent altogether (for example just 'text them', 'call', 'get me a ride').",
            "The transcript comes from dictation and may contain recognition errors; infer the intent.",
            "For answer, reply in one or two spoken sentences.",
            "Resolve relative times (tomorrow, in an hour) against the current date and time.",
        ]
        if let home = settings.homeAddress, !home.isEmpty { lines.append("The user's home address is saved; 'home' refers to it.") }
        if !settings.nicknames.isEmpty {
            let map = settings.nicknames.sorted { $0.key < $1.key }.map { "\($0.key) = \($0.value)" }.joined(separator: "; ")
            lines.append("Nicknames: \(map). Use the contact name, not the nickname.")
        }
        return lines.joined(separator: "\n")
    }
}

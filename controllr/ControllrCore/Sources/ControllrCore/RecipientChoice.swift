import Foundation

public struct TextRecipient: Codable, Equatable, Sendable {
    public let name: String
    public let label: String?
    public let number: String

    public init(name: String, label: String?, number: String) {
        self.name = name
        self.label = label
        self.number = number
    }

    public var description: String {
        let suffix = Self.digits(number).suffix(4).map(String.init).joined(separator: " ")
        let label = label.map { ", \($0)" } ?? ""
        return "\(name)\(label), ending in \(suffix)"
    }

    public var confirmationName: String {
        let suffix = Self.digits(number).suffix(4)
        let label = label.map { ", \($0)" } ?? ""
        return "\(name)\(label), ending in \(suffix)"
    }

    public static func unique(_ recipients: [Self]) -> [Self] {
        var seen = Set<String>()
        return recipients.filter { recipient in
            let digits = Self.digits(recipient.number)
            let prefix = recipient.number.trimmingCharacters(in: .whitespaces).hasPrefix("+") ? "+" : ""
            return !digits.isEmpty && seen.insert(prefix + digits).inserted
        }
    }

    fileprivate static func digits(_ value: String) -> String {
        value.compactMap { $0.wholeNumberValue.flatMap { (0...9).contains($0) ? String($0) : nil } }.joined()
    }
}

public struct RecipientChoice: Codable, Equatable, Sendable {
    public let recipients: [TextRecipient]
    private let attempts: Int

    private static let optionWords = [
        ["1", "one", "first"], ["2", "two", "second"], ["3", "three", "third"],
        ["4", "four", "fourth"], ["5", "five", "fifth"],
    ]
    private static let digitWords = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    public init(recipients: [TextRecipient]) {
        self.init(recipients: TextRecipient.unique(recipients), attempts: 0)
    }

    private init(recipients: [TextRecipient], attempts: Int) {
        self.recipients = recipients
        self.attempts = attempts
    }

    public var questionTitle: String {
        attempts == 0 ? "Which number should I text?" : "I couldn't match that choice. Which number should I text?"
    }

    public var menuOptions: [String] {
        recipients.prefix(Self.optionWords.count).enumerated().map { offset, recipient in
            "Option \(offset + 1), \(recipient.confirmationName)"
        }
    }

    public var question: String {
        let options = recipients.prefix(Self.optionWords.count).enumerated().map { offset, recipient in
            "Option \(offset + 1), \(recipient.description)."
        }.joined(separator: " ")
        let lead = attempts == 0 ? "Which number should I text?" : "I couldn't match that choice."
        return "\(lead) \(options) Say the option number or the last four digits."
    }

    public func retry() -> Self? {
        guard attempts == 0 else { return nil }
        return Self(recipients: recipients, attempts: attempts + 1)
    }

    public func recipient(for reply: String) -> TextRecipient? {
        if let index = menuOptions.firstIndex(of: reply) { return recipients[index] }
        var words = Self.words(reply)
        if words.first == "use" { words.removeFirst() }
        if words.first == "the" { words.removeFirst() }
        if words.last == "please" { words.removeLast() }
        var option = words
        if let first = option.first, ["option", "number"].contains(first) { option.removeFirst() }
        if option.count == 2 && option.last == "one" { option.removeLast() }
        if option.count == 1, let word = option.first,
           let index = Self.optionWords.firstIndex(where: { $0.contains(word) }), recipients.indices.contains(index) {
            return recipients[index]
        }

        if words.first == "number" { words.removeFirst() }
        for prefix in [["ending", "in"], ["ending", "with"], ["ends", "in"], ["last", "four"], ["last", "4"]] where words.starts(with: prefix) {
            words.removeFirst(prefix.count)
            if words.first == "digits" { words.removeFirst() }
            break
        }
        let digitGroups = words.compactMap { word -> String? in
            let digits = TextRecipient.digits(word)
            if !digits.isEmpty && digits.count == word.count { return digits }
            if word == "oh" { return "0" }
            return Self.digitWords.firstIndex(of: word).map(String.init)
        }
        if digitGroups.count == words.count {
            let digits = digitGroups.joined()
            if digits.count >= 4 {
                return uniqueMatch { recipient in
                    let number = TextRecipient.digits(recipient.number)
                    return digits.count == 4 ? number.hasSuffix(digits) : number == digits
                }
            }
        }

        if words.count > 1, let last = words.last, ["one", "number"].contains(last) { words.removeLast() }
        guard !words.isEmpty else { return nil }
        return uniqueMatch { recipient in
            let name = Self.words(recipient.name)
            let label = recipient.label.map(Self.words) ?? []
            return words == name || (!label.isEmpty && (words == label || words == name + label))
        }
    }

    private func uniqueMatch(_ matches: (TextRecipient) -> Bool) -> TextRecipient? {
        let matching = recipients.filter(matches)
        return matching.count == 1 ? matching.first : nil
    }

    private static func words(_ value: String) -> [String] {
        value.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
            .split { !$0.isLetter && !$0.isNumber }.map(String.init)
    }
}

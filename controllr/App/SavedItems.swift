import Foundation
import SwiftData
import ControllrCore

enum SavedItemKind: String, CaseIterable, Identifiable {
    case memory, skill, note

    var id: String { rawValue }
    var title: String { rawValue.capitalized }
}

@Model
final class SavedItem {
    var kindValue: String
    var title: String
    var text: String
    var updatedAt: Date

    init(kind: SavedItemKind, title: String, text: String) {
        self.kindValue = kind.rawValue
        self.title = title
        self.text = text
        self.updatedAt = .now
    }

    var kind: SavedItemKind { SavedItemKind(rawValue: kindValue) ?? .note }
}

@MainActor
struct SavedItems {
    static var shared: SavedItems { SavedItems(context: Store.container.mainContext) }
    let context: ModelContext

    func save(kind: SavedItemKind, title: String, text: String, updating item: SavedItem? = nil,
                     matchByTitle: Bool = false) throws -> String {
        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty, !text.isEmpty else {
            throw ExecutionError.unavailable("Give the item a title and some text.")
        }
        let match = matchByTitle
            ? try entries(kind: kind).first { $0.title.compare(title, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame }
            : nil
        if let existing = item ?? match {
            existing.kindValue = kind.rawValue
            existing.title = title
            existing.text = text
            existing.updatedAt = .now
        } else {
            context.insert(SavedItem(kind: kind, title: title, text: text))
        }
        do {
            try context.save()
        } catch {
            context.rollback()
            throw error
        }
        return title
    }

    func search(kind: SavedItemKind?, query: String) throws -> String {
        let entries = try entries(kind: kind)
        let words = Self.words(query).filter { !["a", "an", "the", "my", "of", "about", "in", "for"].contains($0) }
        let matches: [SavedItem]
        if words.isEmpty {
            matches = Array(entries.prefix(5))
        } else {
            let ranked = entries.enumerated().compactMap { index, item -> (index: Int, score: Int, item: SavedItem)? in
                let title = Set(Self.words(item.title))
                let body = Set(Self.words(item.text))
                let score = words.reduce(0) { score, word in
                    score + (title.contains(word) ? 3 : body.contains(word) ? 1 : 0)
                }
                return score > 0 ? (index, score, item) : nil
            }
            matches = ranked.sorted { lhs, rhs in
                lhs.score == rhs.score ? lhs.index < rhs.index : lhs.score > rhs.score
            }.prefix(5).map(\.item)
        }
        let result = matches.map { item in
            JSONValue.object(["kind": .string(item.kind.rawValue), "title": .string(item.title),
                              "text": .string(item.text)])
        }
        return JSONValue.array(result).jsonString()
    }

    private func entries(kind: SavedItemKind?) throws -> [SavedItem] {
        var query = FetchDescriptor<SavedItem>(sortBy: [SortDescriptor(\.updatedAt, order: .reverse)])
        if let kind {
            let value = kind.rawValue
            query.predicate = #Predicate { $0.kindValue == value }
        }
        return try context.fetch(query)
    }

    private static func words(_ text: String) -> [String] {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
            .split { !$0.isLetter && !$0.isNumber }.map(String.init)
    }
}

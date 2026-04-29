import Foundation

struct Item: Identifiable, Codable, Hashable {
    let id: String
    let content: String
    let title: String
    let type: ItemType
    let epicId: String?
    let dueDate: String?
    var completed: Bool
    let tags: [String]
    let createdAt: String
    var updatedAt: String

    enum ItemType: String, Codable {
        case note
        case todo
        case epic

        var displayName: String {
            switch self {
            case .note: return "Note"
            case .todo: return "To-do"
            case .epic: return "Epic"
            }
        }

        var systemImage: String {
            switch self {
            case .note: return "note.text"
            case .todo: return "checkmark.circle"
            case .epic: return "square.stack.3d.up"
            }
        }
    }

    enum CodingKeys: String, CodingKey {
        case id, content, title, type, completed, tags
        case epicId = "epic_id"
        case dueDate = "due_date"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }

    var createdAtDate: Date {
        ISO8601DateFormatter().date(from: createdAt) ?? .distantPast
    }

    var dueDateFormatted: String? {
        guard let dueDate, let date = ISO8601DateFormatter().date(from: dueDate) else { return dueDate }
        let fmt = DateFormatter()
        fmt.dateStyle = .medium
        fmt.timeStyle = .short
        return fmt.string(from: date)
    }

    var timestampLabel: String {
        let fmt = DateFormatter()
        fmt.dateStyle = .medium
        fmt.timeStyle = .short
        return fmt.string(from: createdAtDate)
    }

    var previewText: String {
        let s = content.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        return s.count <= 120 ? s : String(s.prefix(120)) + "…"
    }

    var dueDateParsed: Date? {
        guard let dueDate else { return nil }
        return ISO8601DateFormatter().date(from: dueDate)
    }

    var isOverdue: Bool {
        guard type == .todo, !completed, let date = dueDateParsed else { return false }
        return date < Date()
    }
}

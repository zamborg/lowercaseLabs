import Foundation

enum JSONValue: Codable, Hashable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}

struct Item: Identifiable, Codable, Hashable {
    let id: String
    let content: String
    let title: String
    let type: ItemType
    let status: ItemStatus
    let priority: ItemPriority?
    let source: ItemSource
    let parentId: String?
    let epicId: String?
    let dueDate: String?
    let startTime: String?
    let endTime: String?
    let location: String?
    let url: String?
    let readStatus: ReadStatus?
    let email: String?
    let phone: String?
    let organization: String?
    let recurrenceRule: [String: JSONValue]?
    let metadata: [String: JSONValue]
    var completed: Bool
    let tags: [String]
    let createdAt: String
    var updatedAt: String

    enum ItemType: Hashable, Codable {
        case note
        case todo
        case event
        case epic
        case contact
        case resource
        case decision
        case journal
        case habit
        case table
        case unknown(String)

        init(from decoder: Decoder) throws {
            let rawValue = try decoder.singleValueContainer().decode(String.self)
            self = ItemType(rawValue: rawValue)
        }

        init(rawValue: String) {
            switch rawValue {
            case "note": self = .note
            case "todo": self = .todo
            case "event": self = .event
            case "epic": self = .epic
            case "contact": self = .contact
            case "resource": self = .resource
            case "decision": self = .decision
            case "journal": self = .journal
            case "habit": self = .habit
            case "table": self = .table
            default: self = .unknown(rawValue)
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            try container.encode(rawValue)
        }

        var rawValue: String {
            switch self {
            case .note: return "note"
            case .todo: return "todo"
            case .event: return "event"
            case .epic: return "epic"
            case .contact: return "contact"
            case .resource: return "resource"
            case .decision: return "decision"
            case .journal: return "journal"
            case .habit: return "habit"
            case .table: return "table"
            case .unknown(let value): return value
            }
        }

        var displayName: String {
            switch self {
            case .note: return "Note"
            case .todo: return "To-do"
            case .event: return "Event"
            case .epic: return "Epic"
            case .contact: return "Contact"
            case .resource: return "Resource"
            case .decision: return "Decision"
            case .journal: return "Journal"
            case .habit: return "Habit"
            case .table: return "Table"
            case .unknown(let value): return value.capitalized
            }
        }

        var systemImage: String {
            switch self {
            case .note: return "note.text"
            case .todo: return "checkmark.circle"
            case .event: return "calendar"
            case .epic: return "square.stack.3d.up"
            case .contact: return "person.crop.circle"
            case .resource: return "link"
            case .decision: return "checkmark.seal"
            case .journal: return "book.closed"
            case .habit: return "repeat"
            case .table: return "tablecells"
            case .unknown: return "questionmark.circle"
            }
        }
    }

    enum ItemStatus: String, Codable, Hashable {
        case open
        case inProgress = "in_progress"
        case done
        case cancelled
        case archived
    }

    enum ItemPriority: String, Codable, Hashable {
        case low
        case medium
        case high
        case urgent
    }

    enum ItemSource: String, Codable, Hashable {
        case voice
        case manual
        case agent
        case `import`
        case system
    }

    enum ReadStatus: String, Codable, Hashable {
        case unread
        case reading
        case read
        case skipped
    }

    enum CodingKeys: String, CodingKey {
        case id, content, title, type, status, priority, source, completed, tags
        case parentId = "parent_id"
        case epicId = "epic_id"
        case dueDate = "due_date"
        case startTime = "start_time"
        case endTime = "end_time"
        case location, url
        case readStatus = "read_status"
        case email, phone, organization
        case recurrenceRule = "recurrence_rule"
        case metadata
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        content = try container.decodeIfPresent(String.self, forKey: .content) ?? ""
        title = try container.decodeIfPresent(String.self, forKey: .title) ?? content
        type = try container.decodeIfPresent(ItemType.self, forKey: .type) ?? .note
        let legacyCompleted = try container.decodeIfPresent(Bool.self, forKey: .completed)
        status = try container.decodeIfPresent(ItemStatus.self, forKey: .status) ?? (legacyCompleted == true ? .done : .open)
        priority = try container.decodeIfPresent(ItemPriority.self, forKey: .priority)
        source = try container.decodeIfPresent(ItemSource.self, forKey: .source) ?? .voice
        parentId = try container.decodeIfPresent(String.self, forKey: .parentId)
        epicId = try container.decodeIfPresent(String.self, forKey: .epicId)
        dueDate = try container.decodeIfPresent(String.self, forKey: .dueDate)
        startTime = try container.decodeIfPresent(String.self, forKey: .startTime)
        endTime = try container.decodeIfPresent(String.self, forKey: .endTime)
        location = try container.decodeIfPresent(String.self, forKey: .location)
        url = try container.decodeIfPresent(String.self, forKey: .url)
        readStatus = try container.decodeIfPresent(ReadStatus.self, forKey: .readStatus)
        email = try container.decodeIfPresent(String.self, forKey: .email)
        phone = try container.decodeIfPresent(String.self, forKey: .phone)
        organization = try container.decodeIfPresent(String.self, forKey: .organization)
        recurrenceRule = try container.decodeIfPresent([String: JSONValue].self, forKey: .recurrenceRule)
        metadata = try container.decodeIfPresent([String: JSONValue].self, forKey: .metadata) ?? [:]
        completed = legacyCompleted ?? (status == .done)
        tags = try container.decodeIfPresent([String].self, forKey: .tags) ?? []
        createdAt = try container.decodeIfPresent(String.self, forKey: .createdAt) ?? ""
        updatedAt = try container.decodeIfPresent(String.self, forKey: .updatedAt) ?? createdAt
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

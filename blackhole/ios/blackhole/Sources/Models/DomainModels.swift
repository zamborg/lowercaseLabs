import Foundation

enum CaptureMode: String, Codable, CaseIterable, Sendable {
    case auto
    case note
    case todo
    case search
}

enum RouteTarget: String, Codable, Sendable {
    case note
    case todo
    case search
}

enum TodoPriority: String, Codable, CaseIterable, Sendable {
    case low
    case normal
    case high
}

enum TodoStatus: String, Codable, CaseIterable, Sendable {
    case open
    case completed
    case canceled
}

struct SessionUser: Codable, Equatable, Sendable {
    let id: String
    let displayName: String
    let email: String?
    let authProvider: String
    let createdAt: Date
}

struct StoredAuthSession: Codable, Equatable, Sendable {
    let accessToken: String
    let user: SessionUser
}

struct NoteRecord: Codable, Identifiable, Equatable, Sendable {
    let id: String
    let userId: String
    var title: String
    var bodyMarkdown: String
    var tags: [String]
    let sourceCaptureId: String?
    let createdAt: Date
    let updatedAt: Date
    let archivedAt: Date?
}

struct TodoRecord: Codable, Identifiable, Equatable, Sendable {
    let id: String
    let userId: String
    var title: String
    var descriptionMarkdown: String
    var priority: TodoPriority
    var status: TodoStatus
    var deadlineAt: Date?
    var tags: [String]
    let sourceCaptureId: String?
    let createdAt: Date
    let updatedAt: Date
    let completedAt: Date?
}

struct SearchNoteResult: Codable, Identifiable, Equatable, Sendable {
    let id: String
    let title: String
    let snippet: String
    let tags: [String]
    let score: Double
    let updatedAt: Date
}

struct NotePage: Codable, Sendable {
    let items: [NoteRecord]
    let nextCursor: String?
}

struct TodoPage: Codable, Sendable {
    let items: [TodoRecord]
    let nextCursor: String?
}

struct SearchPage: Codable, Sendable {
    let items: [SearchNoteResult]
    let nextCursor: String?
}

struct NoteDraftPayload: Codable, Sendable {
    let title: String
    let bodyMarkdown: String
    let tags: [String]
}

struct TodoDraftPayload: Codable, Sendable {
    let title: String
    let descriptionMarkdown: String
    let priority: TodoPriority
    let deadlineAt: Date?
    let tags: [String]
}

struct SearchDraftPayload: Codable, Sendable {
    let queryText: String
}

enum JSONValue: Codable, Equatable, Sendable {
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
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            throw DecodingError.typeMismatch(JSONValue.self, .init(codingPath: decoder.codingPath, debugDescription: "Unsupported JSON value"))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
}

extension Dictionary where Key == String, Value == JSONValue {
    func decodePayload<T: Decodable>(_ type: T.Type) throws -> T {
        let data = try JSONEncoder.blackhole.encode(self)
        return try JSONDecoder.blackhole.decode(T.self, from: data)
    }
}

struct TranscriptionFinalResult: Codable, Sendable {
    let sessionId: String
    let transcript: String
    let route: RouteTarget
    let routeConfidence: Double
    let routePayload: [String: JSONValue]
    let captureId: String?
}

struct TranscriptionSessionRecord: Codable, Sendable {
    let id: String
    let userId: String
    let mode: CaptureMode
    let status: String
    let audioObjectKey: String?
    let finalTranscript: String?
    let route: RouteTarget?
    let routeConfidence: Double?
    let routePayload: [String: JSONValue]?
    let captureId: String?
    let errorMessage: String?
    let createdAt: Date
    let updatedAt: Date

    func asFinalResult() -> TranscriptionFinalResult? {
        guard
            status == "completed",
            let route,
            let routeConfidence,
            let routePayload
        else {
            return nil
        }

        return TranscriptionFinalResult(
            sessionId: id,
            transcript: finalTranscript ?? "",
            route: route,
            routeConfidence: routeConfidence,
            routePayload: routePayload,
            captureId: captureId
        )
    }
}

enum TranscriptEvent: Sendable {
    case sessionCreated(String)
    case uploading
    case transcribing
    case partialTranscript(String)
    case finalResult(TranscriptionFinalResult)
    case failed(String)
}

struct APIMessage: Codable, Sendable {
    let message: String
}

struct AuthSessionResponse: Codable, Sendable {
    let accessToken: String
    let user: SessionUser
}

struct DevAuthRequest: Codable, Sendable {
    let displayName: String
    let email: String?
}

struct NoteCreateRequest: Codable, Sendable {
    let title: String
    let bodyMarkdown: String
    let tags: [String]
    let sourceCaptureId: String?
}

struct NoteUpdateRequest: Codable, Sendable {
    let title: String?
    let bodyMarkdown: String?
    let tags: [String]?
    let archived: Bool?
}

struct TodoCreateRequest: Codable, Sendable {
    let title: String
    let descriptionMarkdown: String
    let priority: TodoPriority
    let deadlineAt: Date?
    let tags: [String]
    let sourceCaptureId: String?
}

struct TodoUpdateRequest: Codable, Sendable {
    let title: String?
    let descriptionMarkdown: String?
    let priority: TodoPriority?
    let deadlineAt: Date?
    let tags: [String]?
    let status: TodoStatus?
}

struct TranscriptionSessionCreateResponse: Codable, Sendable {
    let transcriptionSessionId: String
}

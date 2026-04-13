import CryptoKit
import Foundation

// MARK: - DTOs

struct APIUserProfile: Codable {
    let id: String
    let displayName: String?
    let anonymousHandle: String
    let dailyCheckinTimeLocal: String
    let timezone: String
    let notificationEnabled: Bool
}

struct APIAuthSession: Codable {
    let accessToken: String
    let tokenType: String
    let user: APIUserProfile
}

struct APIEntryCreate: Codable {
    let entryId: String
    let uploadUrl: String
    let uploadMethod: String
    let objectKey: String
    let expiresAt: String
}

enum JSONValue: Codable, Hashable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([JSONValue])
    case object([String: JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported JSON value"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
}

struct APITranscript: Codable, Hashable {
    let text: String
    let providerMetadata: [String: JSONValue]?
}

struct APIInsight: Codable, Hashable {
    let moodScore: Double
    let moodTags: [String]
    let themes: [String]
    let signals: [String: Double]
    let safetyFlags: [String: JSONValue]?
}

enum HealthMetricType: String, Codable, Hashable, CaseIterable {
    case sleepHours
    case hrvSdnnMs
    case restingHeartRateBpm
    case stepsToday

    var displayName: String {
        switch self {
        case .sleepHours:
            return "Sleep"
        case .hrvSdnnMs:
            return "HRV"
        case .restingHeartRateBpm:
            return "Resting HR"
        case .stepsToday:
            return "Steps"
        }
    }

    var displayOrder: Int {
        switch self {
        case .sleepHours:
            return 0
        case .hrvSdnnMs:
            return 1
        case .restingHeartRateBpm:
            return 2
        case .stepsToday:
            return 3
        }
    }
}

struct HealthComponentSnapshot: Codable, Hashable {
    let type: HealthMetricType
    let rawValue: Double
    let unit: String
    let componentScore: Double
    let sampledAtISO8601: String?
    let isStale: Bool

    var scorePercent: Int {
        Int(max(0, min(100, componentScore)).rounded())
    }

    var formattedRawValue: String {
        switch type {
        case .sleepHours:
            return String(format: "%.1fh", rawValue)
        case .hrvSdnnMs:
            return String(format: "%.0f ms", rawValue)
        case .restingHeartRateBpm:
            return String(format: "%.0f bpm", rawValue)
        case .stepsToday:
            return String(format: "%.0f", rawValue)
        }
    }
}

struct EntryHealthSnapshot: Codable, Hashable {
    let capturedAtISO8601: String
    let readinessScore: Int?
    let confidence: Double
    let components: [HealthComponentSnapshot]
    let version: Int

    var sortedComponents: [HealthComponentSnapshot] {
        components.sorted { lhs, rhs in
            if lhs.type.displayOrder != rhs.type.displayOrder {
                return lhs.type.displayOrder < rhs.type.displayOrder
            }
            return lhs.type.rawValue < rhs.type.rawValue
        }
    }

    var confidencePercent: Int {
        Int((max(0, min(1, confidence)) * 100).rounded())
    }
}

struct APIEntry: Codable, Identifiable, Hashable {
    let id: String
    let localDate: String
    let durationSeconds: Int
    let status: String
    let createdAt: String
    let title: String
    let transcript: APITranscript?
    let insight: APIInsight?
    let healthSnapshot: EntryHealthSnapshot?

    var displayTitle: String {
        Self.sanitizeTitle(title)
    }

    var createdAtTimeLabel: String? {
        guard let date = parsedCreatedAt else { return nil }
        return DateFormatter.clock.string(from: date)
    }

    init(
        id: String,
        localDate: String,
        durationSeconds: Int,
        status: String,
        createdAt: String,
        transcript: APITranscript?,
        insight: APIInsight?,
        title: String = "Entry",
        healthSnapshot: EntryHealthSnapshot? = nil
    ) {
        self.id = id
        self.localDate = localDate
        self.durationSeconds = durationSeconds
        self.status = status
        self.createdAt = createdAt
        self.title = Self.sanitizeTitle(title)
        self.transcript = transcript
        self.insight = insight
        self.healthSnapshot = healthSnapshot
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case localDate
        case durationSeconds
        case status
        case createdAt
        case title
        case transcript
        case insight
        case healthSnapshot
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        localDate = try container.decode(String.self, forKey: .localDate)
        durationSeconds = try container.decode(Int.self, forKey: .durationSeconds)
        status = try container.decode(String.self, forKey: .status)
        createdAt = try container.decode(String.self, forKey: .createdAt)
        title = Self.sanitizeTitle(try container.decodeIfPresent(String.self, forKey: .title) ?? "Entry")
        transcript = try container.decodeIfPresent(APITranscript.self, forKey: .transcript)
        insight = try container.decodeIfPresent(APIInsight.self, forKey: .insight)
        healthSnapshot = try container.decodeIfPresent(EntryHealthSnapshot.self, forKey: .healthSnapshot)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(localDate, forKey: .localDate)
        try container.encode(durationSeconds, forKey: .durationSeconds)
        try container.encode(status, forKey: .status)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(Self.sanitizeTitle(title), forKey: .title)
        try container.encodeIfPresent(transcript, forKey: .transcript)
        try container.encodeIfPresent(insight, forKey: .insight)
        try container.encodeIfPresent(healthSnapshot, forKey: .healthSnapshot)
    }

    static func sanitizeTitle(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "Entry" }
        let collapsed = trimmed.replacingOccurrences(
            of: "\\s+",
            with: " ",
            options: .regularExpression
        )
        if collapsed.count <= 48 {
            return collapsed
        }
        return String(collapsed.prefix(48))
    }

    private var parsedCreatedAt: Date? {
        if let date = Self.iso8601WithFractional.date(from: createdAt) {
            return date
        }
        return Self.iso8601Basic.date(from: createdAt)
    }

    private static let iso8601WithFractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static let iso8601Basic: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()
}

struct APISocialDot: Codable, Identifiable, Hashable {
    var id: String {
        if let presenceId, !presenceId.isEmpty {
            return presenceId
        }
        var parts = [userId]
        if let localDate, !localDate.isEmpty {
            parts.append(localDate)
        }
        if let updatedAt, !updatedAt.isEmpty {
            parts.append(updatedAt)
        }
        return parts.joined(separator: "::")
    }

    let userId: String
    let dotColor: String
    let dotTags: [String]?
    let label: String?
    let isRevealed: Bool
    let hasEntry: Bool
    let presenceId: String?
    let localDate: String?
    let updatedAt: String?
}

struct APISocialDotsEnvelope: Codable {
    let localDate: String
    let dots: [APISocialDot]
}

struct APIInvite: Codable {
    let inviteToken: String
    let inviteUrl: String
    let expiresAt: String
}

struct APIMessage: Codable {
    let message: String
}

enum AppleNonce {
    static func random(length: Int = 32) -> String {
        let charset: [Character] = Array("0123456789ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz-._")
        var result = ""
        result.reserveCapacity(length)

        while result.count < length {
            let randomByte = UInt8.random(in: 0 ... 255)
            if randomByte < charset.count {
                result.append(charset[Int(randomByte)])
            }
        }
        return result
    }

    static func sha256(_ value: String) -> String {
        let inputData = Data(value.utf8)
        let hashed = SHA256.hash(data: inputData)
        return hashed.compactMap { String(format: "%02x", $0) }.joined()
    }
}

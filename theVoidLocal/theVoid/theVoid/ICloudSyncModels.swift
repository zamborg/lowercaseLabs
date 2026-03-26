import Foundation

enum ICloudSyncStatus: Equatable {
    case disabled
    case unavailable
    case idle
    case syncing
    case error(String)
}

struct StoredLocalEntry: Codable {
    let entry: APIEntry
    let audioFileName: String
    let wasSharedToSocial: Bool?
    let updatedAt: String
    let cloudRecordChangeTag: String?
    let isDeleted: Bool

    init(
        entry: APIEntry,
        audioFileName: String,
        wasSharedToSocial: Bool?,
        updatedAt: String,
        cloudRecordChangeTag: String? = nil,
        isDeleted: Bool = false
    ) {
        self.entry = entry
        self.audioFileName = audioFileName
        self.wasSharedToSocial = wasSharedToSocial
        self.updatedAt = updatedAt
        self.cloudRecordChangeTag = cloudRecordChangeTag
        self.isDeleted = isDeleted
    }

    private enum CodingKeys: String, CodingKey {
        case entry
        case audioFileName
        case wasSharedToSocial
        case updatedAt
        case cloudRecordChangeTag
        case isDeleted
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        entry = try container.decode(APIEntry.self, forKey: .entry)
        audioFileName = try container.decode(String.self, forKey: .audioFileName)
        wasSharedToSocial = try container.decodeIfPresent(Bool.self, forKey: .wasSharedToSocial)
        updatedAt = try container.decodeIfPresent(String.self, forKey: .updatedAt) ?? entry.createdAt
        cloudRecordChangeTag = try container.decodeIfPresent(String.self, forKey: .cloudRecordChangeTag)
        isDeleted = try container.decodeIfPresent(Bool.self, forKey: .isDeleted) ?? false
    }
}

struct StoredLocalDraft: Codable, Hashable {
    let draftID: String
    let createdAt: String
    let updatedAt: String
    let audioFileName: String
    let isDeleted: Bool
    let cloudRecordChangeTag: String?

    init(
        draftID: String,
        createdAt: String,
        updatedAt: String,
        audioFileName: String,
        isDeleted: Bool = false,
        cloudRecordChangeTag: String? = nil
    ) {
        self.draftID = draftID
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.audioFileName = audioFileName
        self.isDeleted = isDeleted
        self.cloudRecordChangeTag = cloudRecordChangeTag
    }

    private enum CodingKeys: String, CodingKey {
        case draftID
        case createdAt
        case updatedAt
        case audioFileName
        case isDeleted
        case cloudRecordChangeTag
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        draftID = try container.decode(String.self, forKey: .draftID)
        createdAt = try container.decodeIfPresent(String.self, forKey: .createdAt) ?? ICloudSyncTimestamp.nowString()
        updatedAt = try container.decodeIfPresent(String.self, forKey: .updatedAt) ?? createdAt
        audioFileName = try container.decode(String.self, forKey: .audioFileName)
        isDeleted = try container.decodeIfPresent(Bool.self, forKey: .isDeleted) ?? false
        cloudRecordChangeTag = try container.decodeIfPresent(String.self, forKey: .cloudRecordChangeTag)
    }
}

enum ICloudSyncOperationKind: String, Codable {
    case upsertEntry
    case deleteEntry
    case upsertAudio
    case deleteAudio
    case upsertDraft
    case deleteDraft
}

struct ICloudSyncOperation: Codable, Hashable {
    let id: String
    let kind: ICloudSyncOperationKind
    let userID: String
    let entryID: String?
    let draftID: String?
    let queuedAt: String

    init(
        id: String = UUID().uuidString,
        kind: ICloudSyncOperationKind,
        userID: String,
        entryID: String? = nil,
        draftID: String? = nil,
        queuedAt: String = ICloudSyncTimestamp.nowString()
    ) {
        self.id = id
        self.kind = kind
        self.userID = userID
        self.entryID = entryID
        self.draftID = draftID
        self.queuedAt = queuedAt
    }

    var dedupeKey: String {
        "\(kind.rawValue)::\(userID)::\(entryID ?? "")::\(draftID ?? "")"
    }
}

enum ICloudSyncTimestamp {
    private static let withFractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static let basic: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    static func nowString() -> String {
        string(from: Date())
    }

    static func string(from date: Date) -> String {
        withFractional.string(from: date)
    }

    static func date(from rawValue: String?) -> Date? {
        guard let rawValue, !rawValue.isEmpty else { return nil }
        if let parsed = withFractional.date(from: rawValue) {
            return parsed
        }
        return basic.date(from: rawValue)
    }
}

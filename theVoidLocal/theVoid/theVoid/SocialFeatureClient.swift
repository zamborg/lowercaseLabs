import CloudKit
import Foundation

struct SocialProfile: Equatable {
    let userID: String
    let displayName: String?
    let anonymousHandle: String

    var visibleLabel: String {
        if let displayName,
           !displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return displayName
        }
        return anonymousHandle
    }
}

struct SocialDotPublication: Equatable {
    let localDate: String
    let moodScore: Double
    let moodTags: [String]
    let dotColor: String?
    let hasEntry: Bool

    init(
        localDate: String,
        moodScore: Double,
        moodTags: [String],
        dotColor: String?,
        hasEntry: Bool = true
    ) {
        self.localDate = localDate
        self.moodScore = moodScore
        self.moodTags = moodTags
        self.dotColor = dotColor
        self.hasEntry = hasEntry
    }
}

protocol SocialFeatureClient {
    func fetchSocialDots(
        for profile: SocialProfile,
        history: Bool,
        limit: Int
    ) async throws -> APISocialDotsEnvelope

    func publishLocalDot(
        for profile: SocialProfile,
        publication: SocialDotPublication
    ) async throws

    func deleteLocalDot(
        for profile: SocialProfile,
        localDate: String
    ) async throws

    func createInvite(for profile: SocialProfile) async throws -> APIInvite

    func acceptInvite(from rawValue: String) async throws
}

final class CloudKitSocialFeatureClient: SocialFeatureClient {
    private enum Constants {
        static let containerID = "iCloud.com.lowercaseLabs.theVoid"
        static let zoneName = "VoidSocialZone"
        static let circleRecordType = "TVSocialCircle"
        static let dotRecordType = "TVSocialDot"
        static let shareURLDefaultsPrefix = "thevoid.social.cloudkit.shareURL."
    }

    private let container: CKContainer
    private let defaults: UserDefaults
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(
        container: CKContainer = CKContainer(identifier: Constants.containerID),
        defaults: UserDefaults = .standard
    ) {
        self.container = container
        self.defaults = defaults
    }

    func fetchSocialDots(
        for profile: SocialProfile,
        history _: Bool,
        limit: Int
    ) async throws -> APISocialDotsEnvelope {
        let zones = try await fetchAllRecordZones(in: container.sharedCloudDatabase)
        guard !zones.isEmpty else {
            return APISocialDotsEnvelope(localDate: DateFormatter.localDate.string(from: Date()), dots: [])
        }

        var dots: [APISocialDot] = []
        for zone in zones {
            let records = try await fetchDotRecords(in: container.sharedCloudDatabase, zoneID: zone.zoneID)
            dots.append(contentsOf: records.compactMap { parseDot($0, currentUserID: profile.userID) })
        }

        let sortedDots = dots.sorted(by: socialDotIsMoreRecent).prefix(max(1, limit))
        return APISocialDotsEnvelope(
            localDate: DateFormatter.localDate.string(from: Date()),
            dots: Array(sortedDots)
        )
    }

    func publishLocalDot(
        for profile: SocialProfile,
        publication: SocialDotPublication
    ) async throws {
        let root = try await ensurePrivateSocialRoot(for: profile)
        let recordID = CKRecord.ID(recordName: dotRecordName(userID: profile.userID, localDate: publication.localDate), zoneID: privateZoneID)
        let existing = try await fetchRecordIfExists(recordID: recordID, in: container.privateCloudDatabase)
        let record = existing ?? CKRecord(recordType: Constants.dotRecordType, recordID: recordID)
        record.parent = CKRecord.Reference(recordID: root.recordID, action: .none)
        writeDotFields(
            record,
            profile: profile,
            publication: publication,
            isDeleted: false,
            updatedAt: Date()
        )
        _ = try await saveRecord(record, in: container.privateCloudDatabase)
    }

    func deleteLocalDot(
        for profile: SocialProfile,
        localDate: String
    ) async throws {
        let root = try await ensurePrivateSocialRoot(for: profile)
        let recordID = CKRecord.ID(recordName: dotRecordName(userID: profile.userID, localDate: localDate), zoneID: privateZoneID)
        let existing = try await fetchRecordIfExists(recordID: recordID, in: container.privateCloudDatabase)
        let record = existing ?? CKRecord(recordType: Constants.dotRecordType, recordID: recordID)
        record.parent = CKRecord.Reference(recordID: root.recordID, action: .none)
        writeDotFields(
            record,
            profile: profile,
            publication: SocialDotPublication(
                localDate: localDate,
                moodScore: 0,
                moodTags: [],
                dotColor: "#2A2A2A",
                hasEntry: false
            ),
            isDeleted: true,
            updatedAt: Date()
        )
        _ = try await saveRecord(record, in: container.privateCloudDatabase)
    }

    func createInvite(for profile: SocialProfile) async throws -> APIInvite {
        if let cachedURLString = defaults.string(forKey: shareURLDefaultsKey(for: profile.userID)),
           let cachedURL = URL(string: cachedURLString) {
            return invite(from: cachedURL)
        }

        let root = try await ensurePrivateSocialRoot(for: profile)
        let share = CKShare(rootRecord: root)
        share[CKShare.SystemFieldKey.title] = "theVoid Social Circle" as CKRecordValue
        share.publicPermission = .readOnly

        let savedShare = try await saveRecords([root, share], in: container.privateCloudDatabase)
            .compactMap { $0 as? CKShare }
            .first

        guard let shareURL = savedShare?.url else {
            throw CloudKitSocialError.missingShareURL
        }

        defaults.set(shareURL.absoluteString, forKey: shareURLDefaultsKey(for: profile.userID))
        return invite(from: shareURL)
    }

    func acceptInvite(from rawValue: String) async throws {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed), !trimmed.isEmpty else {
            throw CloudKitSocialError.invalidShareURL
        }

        let metadata = try await fetchShareMetadata(for: url)
        try await acceptShare(metadata)
    }

    private var privateZoneID: CKRecordZone.ID {
        CKRecordZone.ID(zoneName: Constants.zoneName, ownerName: CKCurrentUserDefaultName)
    }

    private func ensurePrivateSocialRoot(for profile: SocialProfile) async throws -> CKRecord {
        try await ensureZoneExists(privateZoneID, in: container.privateCloudDatabase)
        let recordID = CKRecord.ID(recordName: rootRecordName(userID: profile.userID), zoneID: privateZoneID)
        let existing = try await fetchRecordIfExists(recordID: recordID, in: container.privateCloudDatabase)
        let record = existing ?? CKRecord(recordType: Constants.circleRecordType, recordID: recordID)
        record["userID"] = profile.userID as CKRecordValue
        record["displayName"] = profile.displayName as CKRecordValue?
        record["anonymousHandle"] = profile.anonymousHandle as CKRecordValue
        record["updatedAt"] = Date() as CKRecordValue
        if existing == nil {
            record["createdAt"] = Date() as CKRecordValue
        }
        return try await saveRecord(record, in: container.privateCloudDatabase)
    }

    private func writeDotFields(
        _ record: CKRecord,
        profile: SocialProfile,
        publication: SocialDotPublication,
        isDeleted: Bool,
        updatedAt: Date
    ) {
        record["userID"] = profile.userID as CKRecordValue
        record["displayName"] = profile.displayName as CKRecordValue?
        record["anonymousHandle"] = profile.anonymousHandle as CKRecordValue
        record["localDate"] = publication.localDate as CKRecordValue
        record["dotColor"] = (publication.dotColor ?? "#2A2A2A") as CKRecordValue
        record["dotTagsJSON"] = encodeJSONString(Array(publication.moodTags.prefix(8))) as CKRecordValue?
        record["hasEntry"] = publication.hasEntry as CKRecordValue
        record["isDeleted"] = isDeleted as CKRecordValue
        record["updatedAt"] = updatedAt as CKRecordValue
    }

    private func parseDot(_ record: CKRecord, currentUserID: String) -> APISocialDot? {
        guard (record["isDeleted"] as? Bool) != true,
              let userID = record["userID"] as? String,
              userID != currentUserID,
              let dotColor = record["dotColor"] as? String else {
            return nil
        }

        let displayName = record["displayName"] as? String
        let anonymousHandle = record["anonymousHandle"] as? String
        let localDate = record["localDate"] as? String
        let updatedAt = record["updatedAt"] as? Date
        let tags: [String] = decodeJSONString(record["dotTagsJSON"] as? String) ?? []
        let label: String?
        if let displayName,
           !displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            label = displayName
        } else {
            label = anonymousHandle
        }

        return APISocialDot(
            userId: userID,
            dotColor: dotColor,
            dotTags: tags,
            label: label,
            isRevealed: label != nil,
            hasEntry: (record["hasEntry"] as? Bool) ?? true,
            presenceId: record.recordID.recordName,
            localDate: localDate,
            updatedAt: updatedAt.map { ICloudSyncTimestamp.string(from: $0) }
        )
    }

    private func socialDotIsMoreRecent(_ lhs: APISocialDot, _ rhs: APISocialDot) -> Bool {
        let lhsUpdatedAt = ICloudSyncTimestamp.date(from: lhs.updatedAt) ?? .distantPast
        let rhsUpdatedAt = ICloudSyncTimestamp.date(from: rhs.updatedAt) ?? .distantPast
        if lhsUpdatedAt != rhsUpdatedAt {
            return lhsUpdatedAt > rhsUpdatedAt
        }

        let lhsLocalDate = DateFormatter.localDate.date(from: lhs.localDate ?? "") ?? .distantPast
        let rhsLocalDate = DateFormatter.localDate.date(from: rhs.localDate ?? "") ?? .distantPast
        if lhsLocalDate != rhsLocalDate {
            return lhsLocalDate > rhsLocalDate
        }

        return lhs.id < rhs.id
    }

    private func fetchDotRecords(in database: CKDatabase, zoneID: CKRecordZone.ID) async throws -> [CKRecord] {
        let query = CKQuery(recordType: Constants.dotRecordType, predicate: NSPredicate(value: true))
        return try await fetchAllRecords(query: query, zoneID: zoneID, in: database)
    }

    private func ensureZoneExists(_ zoneID: CKRecordZone.ID, in database: CKDatabase) async throws {
        let zone = CKRecordZone(zoneID: zoneID)
        try await withCheckedThrowingContinuation { continuation in
            let operation = CKModifyRecordZonesOperation(recordZonesToSave: [zone], recordZoneIDsToDelete: nil)
            operation.modifyRecordZonesCompletionBlock = { _, _, error in
                if let ckError = error as? CKError,
                   ckError.code == .serverRejectedRequest {
                    continuation.resume(returning: ())
                    return
                }
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            }
            database.add(operation)
        }
    }

    private func fetchAllRecordZones(in database: CKDatabase) async throws -> [CKRecordZone] {
        try await withCheckedThrowingContinuation { continuation in
            database.fetchAllRecordZones { zones, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: zones ?? [])
            }
        }
    }

    private func fetchAllRecords(
        query: CKQuery,
        zoneID: CKRecordZone.ID,
        in database: CKDatabase
    ) async throws -> [CKRecord] {
        var records: [CKRecord] = []
        var cursor: CKQueryOperation.Cursor?

        repeat {
            let page = try await fetchRecordsPage(query: cursor == nil ? query : nil, cursor: cursor, zoneID: zoneID, in: database)
            records.append(contentsOf: page.records)
            cursor = page.cursor
        } while cursor != nil

        return records
    }

    private func fetchRecordsPage(
        query: CKQuery?,
        cursor: CKQueryOperation.Cursor?,
        zoneID: CKRecordZone.ID,
        in database: CKDatabase
    ) async throws -> (records: [CKRecord], cursor: CKQueryOperation.Cursor?) {
        try await withCheckedThrowingContinuation { continuation in
            let operation: CKQueryOperation
            if let cursor {
                operation = CKQueryOperation(cursor: cursor)
            } else if let query {
                operation = CKQueryOperation(query: query)
                operation.zoneID = zoneID
            } else {
                continuation.resume(returning: ([], nil))
                return
            }

            operation.resultsLimit = 200
            var records: [CKRecord] = []
            let lock = NSLock()
            operation.recordFetchedBlock = { record in
                lock.lock()
                records.append(record)
                lock.unlock()
            }
            operation.queryCompletionBlock = { [self] nextCursor, error in
                if let error {
                    if isMissingRecordTypeError(error, recordType: Constants.dotRecordType) {
                        continuation.resume(returning: ([], nil))
                    } else {
                        continuation.resume(throwing: error)
                    }
                    return
                }
                continuation.resume(returning: (records, nextCursor))
            }
            database.add(operation)
        }
    }

    private func fetchRecordIfExists(recordID: CKRecord.ID, in database: CKDatabase) async throws -> CKRecord? {
        try await withCheckedThrowingContinuation { continuation in
            database.fetch(withRecordID: recordID) { record, error in
                if let ckError = error as? CKError, ckError.code == .unknownItem {
                    continuation.resume(returning: nil)
                    return
                }
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: record)
            }
        }
    }

    private func saveRecord(_ record: CKRecord, in database: CKDatabase) async throws -> CKRecord {
        let saved = try await saveRecords([record], in: database)
        guard let first = saved.first else {
            throw CloudKitSocialError.emptySaveResult
        }
        return first
    }

    private func saveRecords(_ records: [CKRecord], in database: CKDatabase) async throws -> [CKRecord] {
        try await withCheckedThrowingContinuation { continuation in
            let operation = CKModifyRecordsOperation(recordsToSave: records, recordIDsToDelete: nil)
            operation.savePolicy = .changedKeys
            operation.modifyRecordsCompletionBlock = { savedRecords, _, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: savedRecords ?? [])
            }
            database.add(operation)
        }
    }

    private func fetchShareMetadata(for url: URL) async throws -> CKShare.Metadata {
        try await withCheckedThrowingContinuation { continuation in
            var output: CKShare.Metadata?
            var outputError: Error?
            let operation = CKFetchShareMetadataOperation(shareURLs: [url])
            operation.shouldFetchRootRecord = true
            operation.perShareMetadataBlock = { _, metadata, error in
                output = metadata
                outputError = error
            }
            operation.fetchShareMetadataCompletionBlock = { error in
                if let error = outputError ?? error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let output else {
                    continuation.resume(throwing: CloudKitSocialError.invalidShareURL)
                    return
                }
                continuation.resume(returning: output)
            }
            operation.qualityOfService = .userInitiated
            container.add(operation)
        }
    }

    private func acceptShare(_ metadata: CKShare.Metadata) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let operation = CKAcceptSharesOperation(shareMetadatas: [metadata])
            operation.acceptSharesCompletionBlock = { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            }
            operation.qualityOfService = .userInitiated
            container.add(operation)
        }
    }

    private func invite(from url: URL) -> APIInvite {
        APIInvite(
            inviteToken: url.absoluteString,
            inviteUrl: url.absoluteString,
            expiresAt: "No expiration"
        )
    }

    private func shareURLDefaultsKey(for userID: String) -> String {
        "\(Constants.shareURLDefaultsPrefix)\(userID)"
    }

    private func rootRecordName(userID: String) -> String {
        "social-root::\(userID)"
    }

    private func dotRecordName(userID: String, localDate: String) -> String {
        "social-dot::\(userID)::\(localDate)"
    }

    private func encodeJSONString<T: Encodable>(_ value: T?) -> String? {
        guard let value else { return nil }
        guard let data = try? encoder.encode(value) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func decodeJSONString<T: Decodable>(_ rawValue: String?) -> T? {
        guard let rawValue,
              let data = rawValue.data(using: .utf8) else {
            return nil
        }
        return try? decoder.decode(T.self, from: data)
    }

    private func isMissingRecordTypeError(_ error: Error, recordType: String) -> Bool {
        if let ckError = error as? CKError {
            switch ckError.code {
            case .unknownItem, .invalidArguments:
                return containsMissingRecordTypeMessage(in: ckError.localizedDescription, recordType: recordType)
            case .partialFailure:
                if let nested = ckError.partialErrorsByItemID?.values {
                    for nestedError in nested where isMissingRecordTypeError(nestedError, recordType: recordType) {
                        return true
                    }
                }
            default:
                break
            }
        }

        return containsMissingRecordTypeMessage(in: error.localizedDescription, recordType: recordType)
    }

    private func containsMissingRecordTypeMessage(in message: String, recordType: String) -> Bool {
        let normalizedMessage = message.lowercased()
        let normalizedRecordType = recordType.lowercased()
        return normalizedMessage.contains("did not find record type")
            && normalizedMessage.contains(normalizedRecordType)
    }
}

private enum CloudKitSocialError: LocalizedError {
    case invalidShareURL
    case missingShareURL
    case emptySaveResult

    var errorDescription: String? {
        switch self {
        case .invalidShareURL:
            return "Paste a valid iCloud share link."
        case .missingShareURL:
            return "iCloud did not return a share link."
        case .emptySaveResult:
            return "iCloud did not confirm the saved social record."
        }
    }
}

import CloudKit
import CryptoKit
import Foundation

actor ICloudSyncService {
    private enum Constants {
        static let containerID = "iCloud.com.lowercaseLabs.theVoid"
        static let zoneName = "VoidJournalZone"

        static let entryRecordType = "TVJournalEntry"
        static let audioRecordType = "TVJournalAudio"
        static let draftRecordType = "TVDraft"
        static let ownerUserIDField = "ownerUserID"
    }

    private struct RemoteEntryRecord {
        let entryID: String
        let entry: APIEntry
        let wasSharedToSocial: Bool
        let isDeleted: Bool
        let updatedAt: Date
        let changeTag: String?
    }

    private struct RemoteAudioRecord {
        let entryID: String
        let isDeleted: Bool
        let updatedAt: Date
        let asset: CKAsset?
    }

    private struct RemoteDraftRecord {
        let draftID: String
        let createdAt: Date
        let updatedAt: Date
        let isDeleted: Bool
        let asset: CKAsset?
        let changeTag: String?
    }

    private let defaults: UserDefaults
    private let localJournalStore: LocalJournalStore
    private let draftStore: DraftStore
    private let stateStore = ICloudSyncStateStore()

    private let container: CKContainer
    private let database: CKDatabase
    private let zoneID = CKRecordZone.ID(zoneName: Constants.zoneName, ownerName: CKCurrentUserDefaultName)

    private var enabled = true
    private var isSyncing = false
    private nonisolated(unsafe) var statusHandler: (@MainActor (ICloudSyncStatus, Date?) -> Void)?

    init(localJournalStore: LocalJournalStore, draftStore: DraftStore, defaults: UserDefaults = .standard) {
        self.localJournalStore = localJournalStore
        self.draftStore = draftStore
        self.defaults = defaults
        let container = CKContainer(identifier: Constants.containerID)
        self.container = container
        self.database = container.privateCloudDatabase
    }

    func setStatusHandler(_ handler: @escaping @MainActor (ICloudSyncStatus, Date?) -> Void) {
        statusHandler = handler
    }

    func setEnabled(_ value: Bool, userID: String) async {
        enabled = value
        if !value {
            let lastSync = await stateStore.lastSyncAt()
            await publishStatus(.disabled, lastSync: lastSync)
            return
        }
        await syncNow(userID: userID, reason: "enable")
    }

    func queueEntryUpsert(userID: String, entryID: String) async {
        guard !userID.isEmpty, !entryID.isEmpty else { return }
        await stateStore.enqueue(ICloudSyncOperation(kind: .upsertEntry, userID: userID, entryID: entryID))
        await stateStore.enqueue(ICloudSyncOperation(kind: .upsertAudio, userID: userID, entryID: entryID))
    }

    func queueEntryDelete(userID: String, entryID: String) async {
        guard !userID.isEmpty, !entryID.isEmpty else { return }
        await stateStore.enqueue(ICloudSyncOperation(kind: .deleteEntry, userID: userID, entryID: entryID))
        await stateStore.enqueue(ICloudSyncOperation(kind: .deleteAudio, userID: userID, entryID: entryID))
    }

    func queueDraftUpsert(userID: String, draftID: String) async {
        guard !userID.isEmpty, !draftID.isEmpty else { return }
        await stateStore.enqueue(ICloudSyncOperation(kind: .upsertDraft, userID: userID, draftID: draftID))
    }

    func queueDraftDelete(userID: String, draftID: String) async {
        guard !userID.isEmpty, !draftID.isEmpty else { return }
        await stateStore.enqueue(ICloudSyncOperation(kind: .deleteDraft, userID: userID, draftID: draftID))
    }

    func queueBootstrapBackfill(userID: String) async {
        guard !userID.isEmpty else { return }
        for record in localJournalStore.storedEntries(for: userID) where !record.isDeleted {
            await queueEntryUpsert(userID: userID, entryID: record.entry.id)
        }
        for draft in draftStore.storedDrafts() where !draft.isDeleted {
            await queueDraftUpsert(userID: userID, draftID: draft.draftID)
        }
    }

    func syncNow(userID: String, reason _: String = "manual") async {
        guard enabled else {
            let lastSync = await stateStore.lastSyncAt()
            await publishStatus(.disabled, lastSync: lastSync)
            return
        }
        guard !userID.isEmpty else { return }
        guard !isSyncing else { return }

        isSyncing = true
        let previousSync = await stateStore.lastSyncAt()
        await publishStatus(.syncing, lastSync: previousSync)

        do {
            guard await isICloudAccountAvailable() else {
                await publishStatus(.unavailable, lastSync: previousSync)
                isSyncing = false
                return
            }

            try await ensureZoneExists()
            try await pullRemoteState(userID: userID)
            try await pushQueuedOperations(userID: userID)

            let now = Date()
            await stateStore.setLastSyncAt(now)
            await publishStatus(.idle, lastSync: now)
        } catch {
            await publishStatus(.error(error.localizedDescription), lastSync: previousSync)
        }

        isSyncing = false
    }

    private func pullRemoteState(userID: String) async throws {
        let entryRecords = try await fetchAllRecords(recordType: Constants.entryRecordType, userID: userID)
        for record in entryRecords {
            guard let remoteEntry = parseRemoteEntry(record) else { continue }
            let local = localJournalStore.storedEntry(for: userID, entryID: remoteEntry.entryID)
            let localUpdatedAt = ICloudSyncTimestamp.date(from: local?.updatedAt) ?? .distantPast

            if remoteEntry.isDeleted {
                if remoteEntry.updatedAt >= localUpdatedAt {
                    try? localJournalStore.deleteEntryForSync(for: userID, entryID: remoteEntry.entryID)
                }
                continue
            }

            if localUpdatedAt > remoteEntry.updatedAt {
                continue
            }

            let mergedAudioFileName = local?.audioFileName ?? "\(remoteEntry.entryID).m4a"
            let merged = StoredLocalEntry(
                entry: remoteEntry.entry,
                audioFileName: mergedAudioFileName,
                wasSharedToSocial: remoteEntry.wasSharedToSocial,
                updatedAt: ICloudSyncTimestamp.string(from: remoteEntry.updatedAt),
                cloudRecordChangeTag: remoteEntry.changeTag,
                isDeleted: false
            )
            try localJournalStore.upsertStoredEntry(for: userID, record: merged)
        }

        let audioRecords = try await fetchAllRecords(recordType: Constants.audioRecordType, userID: userID)
        for record in audioRecords {
            guard let remoteAudio = parseRemoteAudio(record) else { continue }
            let local = localJournalStore.storedEntry(for: userID, entryID: remoteAudio.entryID)
            let localUpdatedAt = ICloudSyncTimestamp.date(from: local?.updatedAt) ?? .distantPast

            if remoteAudio.isDeleted {
                if remoteAudio.updatedAt >= localUpdatedAt {
                    try? localJournalStore.deleteAudioForSync(for: userID, entryID: remoteAudio.entryID)
                }
                continue
            }

            if localUpdatedAt > remoteAudio.updatedAt {
                continue
            }

            guard let assetURL = remoteAudio.asset?.fileURL,
                  let audioData = try? Data(contentsOf: assetURL) else {
                continue
            }
            try? localJournalStore.saveAudioDataForSync(
                for: userID,
                entryID: remoteAudio.entryID,
                data: audioData
            )
        }

        let draftRecords = try await fetchAllRecords(recordType: Constants.draftRecordType, userID: userID)
        for record in draftRecords {
            guard let remoteDraft = parseRemoteDraft(record) else { continue }
            let local = draftStore.storedDraft(draftID: remoteDraft.draftID)
            let localUpdatedAt = ICloudSyncTimestamp.date(from: local?.updatedAt) ?? .distantPast

            if remoteDraft.isDeleted {
                if remoteDraft.updatedAt >= localUpdatedAt {
                    draftStore.deleteDraft(draftID: remoteDraft.draftID)
                }
                continue
            }

            if localUpdatedAt > remoteDraft.updatedAt {
                continue
            }

            guard let assetURL = remoteDraft.asset?.fileURL,
                  let draftData = try? Data(contentsOf: assetURL) else {
                continue
            }

            let merged = StoredLocalDraft(
                draftID: remoteDraft.draftID,
                createdAt: ICloudSyncTimestamp.string(from: remoteDraft.createdAt),
                updatedAt: ICloudSyncTimestamp.string(from: remoteDraft.updatedAt),
                audioFileName: "\(remoteDraft.draftID).m4a",
                isDeleted: false,
                cloudRecordChangeTag: remoteDraft.changeTag
            )
            try? draftStore.upsertDraftFromCloud(merged, audioData: draftData)
        }

        await stateStore.setLastFullReconcileAt(Date())
    }

    private func pushQueuedOperations(userID: String) async throws {
        let queued = await stateStore.queuedOperations(for: userID)
        guard !queued.isEmpty else { return }

        var completedIDs: Set<String> = []
        do {
            for operation in queued {
                try await process(operation)
                completedIDs.insert(operation.id)
            }
            await stateStore.removeOperations(withIDs: completedIDs)
        } catch {
            await stateStore.removeOperations(withIDs: completedIDs)
            throw error
        }
    }

    private func process(_ operation: ICloudSyncOperation) async throws {
        switch operation.kind {
        case .upsertEntry:
            guard let entryID = operation.entryID else { return }
            try await upsertEntry(userID: operation.userID, entryID: entryID)
        case .deleteEntry:
            guard let entryID = operation.entryID else { return }
            try await tombstoneEntry(userID: operation.userID, entryID: entryID, deletedAt: operation.queuedAt)
        case .upsertAudio:
            guard let entryID = operation.entryID else { return }
            try await upsertAudio(userID: operation.userID, entryID: entryID)
        case .deleteAudio:
            guard let entryID = operation.entryID else { return }
            try await tombstoneAudio(userID: operation.userID, entryID: entryID, deletedAt: operation.queuedAt)
        case .upsertDraft:
            guard let draftID = operation.draftID else { return }
            try await upsertDraft(userID: operation.userID, draftID: draftID)
        case .deleteDraft:
            guard let draftID = operation.draftID else { return }
            try await tombstoneDraft(userID: operation.userID, draftID: draftID, deletedAt: operation.queuedAt)
        }
    }

    private func upsertEntry(userID: String, entryID: String) async throws {
        guard let localRecord = localJournalStore.storedEntry(for: userID, entryID: entryID), !localRecord.isDeleted else {
            return
        }
        let localUpdatedAt = ICloudSyncTimestamp.date(from: localRecord.updatedAt) ?? Date()

        let recordID = CKRecord.ID(recordName: entryRecordName(userID: userID, entryID: entryID), zoneID: zoneID)
        let existing = try await fetchRecordIfExists(recordID: recordID)
        if let existing,
           let remoteUpdatedAt = existing["updatedAt"] as? Date,
           remoteUpdatedAt > localUpdatedAt {
            return
        }

        let record = existing ?? CKRecord(recordType: Constants.entryRecordType, recordID: recordID)
        record[Constants.ownerUserIDField] = userID as CKRecordValue
        record["entryID"] = entryID as CKRecordValue
        record["localDate"] = localRecord.entry.localDate as CKRecordValue
        record["createdAt"] = (ICloudSyncTimestamp.date(from: localRecord.entry.createdAt) ?? localUpdatedAt) as CKRecordValue
        record["updatedAt"] = localUpdatedAt as CKRecordValue
        record["durationSeconds"] = localRecord.entry.durationSeconds as CKRecordValue
        record["title"] = localRecord.entry.title as CKRecordValue
        record["status"] = localRecord.entry.status as CKRecordValue
        record["transcriptJSON"] = encodeJSONString(localRecord.entry.transcript) as CKRecordValue?
        record["insightJSON"] = encodeJSONString(localRecord.entry.insight) as CKRecordValue?
        record["healthSnapshotJSON"] = encodeJSONString(localRecord.entry.healthSnapshot) as CKRecordValue?
        record["wasSharedToSocial"] = (localRecord.wasSharedToSocial ?? false) as CKRecordValue
        record["isDeleted"] = false as CKRecordValue
        record["deletedAt"] = nil

        let saved = try await saveRecord(record)
        try? localJournalStore.updateCloudChangeTag(
            for: userID,
            entryID: entryID,
            changeTag: saved.recordChangeTag
        )
    }

    private func tombstoneEntry(userID: String, entryID: String, deletedAt: String) async throws {
        let deletionDate = ICloudSyncTimestamp.date(from: deletedAt) ?? Date()

        let recordID = CKRecord.ID(recordName: entryRecordName(userID: userID, entryID: entryID), zoneID: zoneID)
        let existing = try await fetchRecordIfExists(recordID: recordID)
        if let existing,
           let remoteUpdatedAt = existing["updatedAt"] as? Date,
           remoteUpdatedAt > deletionDate {
            return
        }

        let record = existing ?? CKRecord(recordType: Constants.entryRecordType, recordID: recordID)
        record[Constants.ownerUserIDField] = userID as CKRecordValue
        record["entryID"] = entryID as CKRecordValue
        record["updatedAt"] = deletionDate as CKRecordValue
        record["isDeleted"] = true as CKRecordValue
        record["deletedAt"] = deletionDate as CKRecordValue
        _ = try await saveRecord(record)
    }

    private func upsertAudio(userID: String, entryID: String) async throws {
        guard let localRecord = localJournalStore.storedEntry(for: userID, entryID: entryID), !localRecord.isDeleted else {
            return
        }
        guard let localAudioData = try? localJournalStore.audioData(for: userID, entryID: entryID) else {
            return
        }

        let localUpdatedAt = ICloudSyncTimestamp.date(from: localRecord.updatedAt) ?? Date()
        let recordID = CKRecord.ID(recordName: audioRecordName(userID: userID, entryID: entryID), zoneID: zoneID)
        let existing = try await fetchRecordIfExists(recordID: recordID)
        if let existing,
           let remoteUpdatedAt = existing["updatedAt"] as? Date,
           remoteUpdatedAt > localUpdatedAt {
            return
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("audio-upload-\(UUID().uuidString).m4a")
        try localAudioData.write(to: tempURL, options: [.atomic])
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let record = existing ?? CKRecord(recordType: Constants.audioRecordType, recordID: recordID)
        record[Constants.ownerUserIDField] = userID as CKRecordValue
        record["entryID"] = entryID as CKRecordValue
        record["updatedAt"] = localUpdatedAt as CKRecordValue
        record["audioAsset"] = CKAsset(fileURL: tempURL)
        record["contentHash"] = contentHash(for: localAudioData) as CKRecordValue
        record["isDeleted"] = false as CKRecordValue
        record["deletedAt"] = nil

        _ = try await saveRecord(record)
    }

    private func tombstoneAudio(userID: String, entryID: String, deletedAt: String) async throws {
        let deletionDate = ICloudSyncTimestamp.date(from: deletedAt) ?? Date()

        let recordID = CKRecord.ID(recordName: audioRecordName(userID: userID, entryID: entryID), zoneID: zoneID)
        let existing = try await fetchRecordIfExists(recordID: recordID)
        if let existing,
           let remoteUpdatedAt = existing["updatedAt"] as? Date,
           remoteUpdatedAt > deletionDate {
            return
        }

        let record = existing ?? CKRecord(recordType: Constants.audioRecordType, recordID: recordID)
        record[Constants.ownerUserIDField] = userID as CKRecordValue
        record["entryID"] = entryID as CKRecordValue
        record["updatedAt"] = deletionDate as CKRecordValue
        record["isDeleted"] = true as CKRecordValue
        record["deletedAt"] = deletionDate as CKRecordValue
        record["audioAsset"] = nil

        _ = try await saveRecord(record)
    }

    private func upsertDraft(userID: String, draftID: String) async throws {
        guard let localDraft = draftStore.storedDraft(draftID: draftID), !localDraft.isDeleted else {
            return
        }
        guard let localDraftData = try? draftStore.draftData(draftID: draftID) else {
            return
        }

        let localUpdatedAt = ICloudSyncTimestamp.date(from: localDraft.updatedAt) ?? Date()
        let recordID = CKRecord.ID(recordName: draftRecordName(userID: userID, draftID: draftID), zoneID: zoneID)
        let existing = try await fetchRecordIfExists(recordID: recordID)
        if let existing,
           let remoteUpdatedAt = existing["updatedAt"] as? Date,
           remoteUpdatedAt > localUpdatedAt {
            return
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("draft-upload-\(UUID().uuidString).m4a")
        try localDraftData.write(to: tempURL, options: [.atomic])
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let record = existing ?? CKRecord(recordType: Constants.draftRecordType, recordID: recordID)
        record[Constants.ownerUserIDField] = userID as CKRecordValue
        record["draftID"] = draftID as CKRecordValue
        record["createdAt"] = (ICloudSyncTimestamp.date(from: localDraft.createdAt) ?? localUpdatedAt) as CKRecordValue
        record["updatedAt"] = localUpdatedAt as CKRecordValue
        record["audioAsset"] = CKAsset(fileURL: tempURL)
        record["contentHash"] = contentHash(for: localDraftData) as CKRecordValue
        record["isDeleted"] = false as CKRecordValue
        record["deletedAt"] = nil

        let saved = try await saveRecord(record)
        draftStore.updateCloudChangeTag(draftID: draftID, changeTag: saved.recordChangeTag)
    }

    private func tombstoneDraft(userID: String, draftID: String, deletedAt: String) async throws {
        let deletionDate = ICloudSyncTimestamp.date(from: deletedAt) ?? Date()

        let recordID = CKRecord.ID(recordName: draftRecordName(userID: userID, draftID: draftID), zoneID: zoneID)
        let existing = try await fetchRecordIfExists(recordID: recordID)
        if let existing,
           let remoteUpdatedAt = existing["updatedAt"] as? Date,
           remoteUpdatedAt > deletionDate {
            return
        }

        let record = existing ?? CKRecord(recordType: Constants.draftRecordType, recordID: recordID)
        record[Constants.ownerUserIDField] = userID as CKRecordValue
        record["draftID"] = draftID as CKRecordValue
        record["updatedAt"] = deletionDate as CKRecordValue
        record["isDeleted"] = true as CKRecordValue
        record["deletedAt"] = deletionDate as CKRecordValue
        record["audioAsset"] = nil

        _ = try await saveRecord(record)
    }

    private func parseRemoteEntry(_ record: CKRecord) -> RemoteEntryRecord? {
        guard let entryID = record["entryID"] as? String,
              let updatedAtDate = record["updatedAt"] as? Date else {
            return nil
        }
        let isDeleted = (record["isDeleted"] as? Bool) ?? false

        let localDate = record["localDate"] as? String ?? DateFormatter.localDate.string(from: updatedAtDate)
        let createdAtDate = record["createdAt"] as? Date ?? updatedAtDate
        let title = record["title"] as? String ?? "Entry"
        let status = record["status"] as? String ?? "ready"
        let durationSeconds = (record["durationSeconds"] as? Int) ?? 60

        let transcript: APITranscript? = decodeJSONString(record["transcriptJSON"] as? String)
        let insight: APIInsight? = decodeJSONString(record["insightJSON"] as? String)
        let healthSnapshot: EntryHealthSnapshot? = decodeJSONString(record["healthSnapshotJSON"] as? String)

        let entry = APIEntry(
            id: entryID,
            localDate: localDate,
            durationSeconds: max(1, durationSeconds),
            status: status,
            createdAt: ICloudSyncTimestamp.string(from: createdAtDate),
            transcript: transcript,
            insight: insight,
            title: title,
            healthSnapshot: healthSnapshot
        )

        return RemoteEntryRecord(
            entryID: entryID,
            entry: entry,
            wasSharedToSocial: (record["wasSharedToSocial"] as? Bool) ?? false,
            isDeleted: isDeleted,
            updatedAt: updatedAtDate,
            changeTag: record.recordChangeTag
        )
    }

    private func parseRemoteAudio(_ record: CKRecord) -> RemoteAudioRecord? {
        guard let entryID = record["entryID"] as? String,
              let updatedAt = record["updatedAt"] as? Date else {
            return nil
        }
        return RemoteAudioRecord(
            entryID: entryID,
            isDeleted: (record["isDeleted"] as? Bool) ?? false,
            updatedAt: updatedAt,
            asset: record["audioAsset"] as? CKAsset
        )
    }

    private func parseRemoteDraft(_ record: CKRecord) -> RemoteDraftRecord? {
        guard let draftID = record["draftID"] as? String,
              let createdAt = record["createdAt"] as? Date,
              let updatedAt = record["updatedAt"] as? Date else {
            return nil
        }
        return RemoteDraftRecord(
            draftID: draftID,
            createdAt: createdAt,
            updatedAt: updatedAt,
            isDeleted: (record["isDeleted"] as? Bool) ?? false,
            asset: record["audioAsset"] as? CKAsset,
            changeTag: record.recordChangeTag
        )
    }

    private func isICloudAccountAvailable() async -> Bool {
        await withCheckedContinuation { continuation in
            container.accountStatus { status, _ in
                continuation.resume(returning: status == .available)
            }
        }
    }

    private func ensureZoneExists() async throws {
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

    private func fetchAllRecords(recordType: String, userID: String) async throws -> [CKRecord] {
        let predicate = NSPredicate(format: "%K == %@", Constants.ownerUserIDField, userID)
        let query = CKQuery(recordType: recordType, predicate: predicate)

        var allRecords: [CKRecord] = []
        var cursor: CKQueryOperation.Cursor?
        do {
            repeat {
                let page = try await fetchRecordsPage(query: cursor == nil ? query : nil, cursor: cursor)
                allRecords.append(contentsOf: page.records)
                cursor = page.cursor
            } while cursor != nil
        } catch {
            // First-run containers may not have deployed/created the custom record type yet.
            // Treat this as "no remote rows" so local upserts can proceed.
            if isMissingRecordTypeError(error, recordType: recordType) {
                return []
            }
            throw error
        }

        return allRecords
    }

    private func fetchRecordsPage(
        query: CKQuery?,
        cursor: CKQueryOperation.Cursor?
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

            operation.queryCompletionBlock = { nextCursor, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: (records, nextCursor))
            }

            database.add(operation)
        }
    }

    private func fetchRecordIfExists(recordID: CKRecord.ID) async throws -> CKRecord? {
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

    private func saveRecord(_ record: CKRecord) async throws -> CKRecord {
        try await withCheckedThrowingContinuation { continuation in
            database.save(record) { saved, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let saved else {
                    continuation.resume(throwing: NSError(domain: "ICloudSync", code: -1))
                    return
                }
                continuation.resume(returning: saved)
            }
        }
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

    private func publishStatus(_ status: ICloudSyncStatus, lastSync: Date?) async {
        guard let statusHandler else { return }
        await MainActor.run {
            statusHandler(status, lastSync)
        }
    }

    private func contentHash(for data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func encodeJSONString<T: Encodable>(_ value: T?) -> String? {
        guard let value else { return nil }
        guard let data = try? JSONEncoder().encode(value) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func decodeJSONString<T: Decodable>(_ rawValue: String?) -> T? {
        guard let rawValue,
              let data = rawValue.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(T.self, from: data)
    }

    private func entryRecordName(userID: String, entryID: String) -> String {
        "entry::\(userID)::\(entryID)"
    }

    private func audioRecordName(userID: String, entryID: String) -> String {
        "audio::\(userID)::\(entryID)"
    }

    private func draftRecordName(userID: String, draftID: String) -> String {
        "draft::\(userID)::\(draftID)"
    }
}

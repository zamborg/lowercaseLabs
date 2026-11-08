import CloudKit
import Foundation

struct FriendDot: Identifiable {
    let id: String
    let ownerID: String
    let createdAt: Date
    let tags: [Quadrant]
}

/// The whole of what a friend can ever receive: who (implicit), when, and which tags. There is no field for
/// anything else, so content cannot leak through here.
struct Dot: Equatable, Sendable {
    let id: String
    let createdAt: Date
    let tags: [Quadrant]
}

/// Dots: friends-only, tags only. One CloudKit zone-wide share holds the user's dots; friends are
/// read-only participants. No transcript, title or audio ever leaves the device/private iCloud.
@MainActor
@Observable
final class DotStore {
    static let shared = DotStore()
    nonisolated static let containerID = "iCloud.com.lowercaselabs.thevoid"
    /// Off unless the build carries the iCloud entitlement (Info.plist VoidICloud = true).
    nonisolated static let enabled = Bundle.main.object(forInfoDictionaryKey: "VoidICloud") as? Bool == true

    @ObservationIgnored private lazy var container = CKContainer(identifier: DotStore.containerID)
    private let zoneID = CKRecordZone.ID(zoneName: "Dots", ownerName: CKCurrentUserDefaultName)
    private var zoneReady = false

    var friendDots: [FriendDot] = []
    var status = ""

    /// Test edge (Mac harness only): an in-memory stand-in for the CloudKit zone, so flows can see exactly what
    /// would leave the device. Nil in the app.
    private(set) var fakeRemote: [String: Dot]?
    func useFakeCloud() { fakeRemote = [:] }

    // What this device believes is published, so a failed publish is retried and a withdrawn dot is removed.
    private var published: [String: [String]] {
        get { UserDefaults.standard.dictionary(forKey: "publishedDots") as? [String: [String]] ?? [:] }
        set { UserDefaults.standard.set(newValue, forKey: "publishedDots") }
    }

    // MARK: Publish

    /// Make the cloud match `desired`: publish what's missing or changed, withdraw what's no longer wanted.
    /// Idempotent and safe to call after anything. Does nothing at all while iCloud is off.
    func reconcile(desired: [Dot]) async {
        guard Self.enabled || fakeRemote != nil else { return }
        var ledger = fakeRemote == nil ? published : fakeRemote!.mapValues { $0.tags.map(\.rawValue) }
        let wanted = Dictionary(uniqueKeysWithValues: desired.map { ($0.id, $0) })
        let toPublish = desired.filter { ledger[$0.id] != $0.tags.map(\.rawValue) }
        let toWithdraw = ledger.keys.filter { wanted[$0] == nil }
        guard !toPublish.isEmpty || !toWithdraw.isEmpty else { return }

        if fakeRemote != nil {
            for dot in toPublish { fakeRemote?[dot.id] = dot }
            for id in toWithdraw { fakeRemote?[id] = nil }
            return
        }
        do {
            try await ensureZone()
            let records = toPublish.map { dot -> CKRecord in
                let record = CKRecord(recordType: "Dot", recordID: CKRecord.ID(recordName: dot.id, zoneID: zoneID))
                record["createdAt"] = dot.createdAt
                record["tags"] = dot.tags.map(\.rawValue)
                return record
            }
            let ids = toWithdraw.map { CKRecord.ID(recordName: $0, zoneID: zoneID) }
            _ = try await container.privateCloudDatabase.modifyRecords(saving: records, deleting: ids)
            for dot in toPublish { ledger[dot.id] = dot.tags.map(\.rawValue) }
            for id in toWithdraw { ledger[id] = nil }
            published = ledger
        } catch {
            status = "dots not synced: \(error.localizedDescription)"
        }
    }

    private func ensureZone() async throws {
        guard !zoneReady else { return }
        _ = try await container.privateCloudDatabase.modifyRecordZones(saving: [CKRecordZone(zoneID: zoneID)], deleting: [])
        zoneReady = true
    }

    // MARK: Invite

    /// The zone-wide share (created on first use). Read-only for participants, invite-only.
    func shareForInvite() async throws -> CKShare {
        guard Self.enabled else { throw DotError.iCloudOff }
        try await ensureZone()
        let shareID = CKRecord.ID(recordName: CKRecordNameZoneWideShare, zoneID: zoneID)
        if let existing = try? await container.privateCloudDatabase.record(for: shareID) as? CKShare {
            return existing
        }
        let share = CKShare(recordZoneID: zoneID)
        share[CKShare.SystemFieldKey.title] = "My Void dots"
        share.publicPermission = .none
        _ = try await container.privateCloudDatabase.modifyRecords(saving: [share], deleting: [])
        return share
    }

    var cloudContainer: CKContainer { container }

    // MARK: Accept

    func accept(_ metadata: CKShare.Metadata) async {
        guard Self.enabled else { return }
        do {
            _ = try await container.accept([metadata])
            // Friendship is mutual: after accepting, also invite back by sharing our own zone.
            status = "friend added"
            await refreshFriends()
        } catch {
            status = "accept failed: \(error.localizedDescription)"
        }
    }

    // MARK: Read friends' dots (pull on open)

    func refreshFriends() async {
        guard Self.enabled else { status = "iCloud sharing is off in this build"; return }
        let db = container.sharedCloudDatabase
        do {
            let zones = try await db.allRecordZones()
            var dots: [FriendDot] = []
            for zone in zones {
                let changes = try await db.recordZoneChanges(inZoneWith: zone.zoneID, since: nil)
                for (_, result) in changes.modificationResultsByID {
                    guard let record = try? result.get().record, record.recordType == "Dot",
                          let createdAt = record["createdAt"] as? Date,
                          let raw = record["tags"] as? [String] else { continue }
                    dots.append(FriendDot(
                        id: record.recordID.recordName, ownerID: zone.zoneID.ownerName,
                        createdAt: createdAt, tags: raw.compactMap(Quadrant.init(rawValue:))
                    ))
                }
            }
            friendDots = dots.sorted { $0.createdAt > $1.createdAt }
        } catch {
            status = "couldn't load friends' dots: \(error.localizedDescription)"
        }
    }
}

enum DotError: Error, LocalizedError {
    case iCloudOff
    var errorDescription: String? { "iCloud sharing is off in this build" }
}

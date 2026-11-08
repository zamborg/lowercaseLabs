import CloudKit
import Foundation

struct FriendDot: Identifiable {
    let id: String
    let ownerID: String
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

    private let container = CKContainer(identifier: DotStore.containerID)
    private let zoneID = CKRecordZone.ID(zoneName: "Dots", ownerName: CKCurrentUserDefaultName)
    private var zoneReady = false

    var friendDots: [FriendDot] = []
    var status = ""

    // MARK: Publish

    /// A dot is derived from a memo at save time: id, owner (implicit), createdAt, tags.
    func publish(memoID: UUID, tags: [Quadrant], createdAt: Date) async throws {
        try await ensureZone()
        let record = CKRecord(recordType: "Dot", recordID: CKRecord.ID(recordName: memoID.uuidString, zoneID: zoneID))
        record["createdAt"] = createdAt
        record["tags"] = tags.map(\.rawValue)
        _ = try await container.privateCloudDatabase.modifyRecords(saving: [record], deleting: [])
    }

    /// Deleting a memo deletes its dot.
    func unpublish(memoID: UUID) async {
        let id = CKRecord.ID(recordName: memoID.uuidString, zoneID: zoneID)
        _ = try? await container.privateCloudDatabase.modifyRecords(saving: [], deleting: [id])
    }

    private func ensureZone() async throws {
        guard !zoneReady else { return }
        _ = try await container.privateCloudDatabase.modifyRecordZones(saving: [CKRecordZone(zoneID: zoneID)], deleting: [])
        zoneReady = true
    }

    // MARK: Invite

    /// The zone-wide share (created on first use). Read-only for participants, invite-only.
    func shareForInvite() async throws -> CKShare {
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

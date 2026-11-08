import CloudKit
import Foundation

struct FriendDot: Identifiable {
    let id: String
    let ownerID: String
    let createdAt: Date
    let tags: [Quadrant]
}

struct Friend: Identifiable {
    let id: String          // the owner's zone owner name
    let name: String
    let dots: [FriendDot]   // newest first
    var sharesBack = true   // whether they can see our dots too
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
    nonisolated static let containerID = "iCloud.com.lowercaseLabs.void"
    /// Off unless the build carries the iCloud entitlement (Info.plist VoidICloud = true).
    nonisolated static let enabled: Bool = {
        let value = Bundle.main.object(forInfoDictionaryKey: "VoidICloud")
        return (value as? Bool) == true || (value as? String) == "true"
    }()

    @ObservationIgnored private lazy var container = CKContainer(identifier: DotStore.containerID)
    private let zoneID = CKRecordZone.ID(zoneName: "Dots", ownerName: CKCurrentUserDefaultName)
    private var zoneReady = false

    var friends: [Friend] = []
    var status = ""

    /// Test edge (Mac harness only): an in-memory stand-in for the CloudKit zone, so flows can see exactly what
    /// would leave the device. Nil in the app.
    private(set) var fakeRemote: [String: Dot]?
    func useFakeCloud() { fakeRemote = [:] }
    /// `-seed` edge: friends with plausible dots so the Social screen can be checked without iCloud.
    func seedFriends(_ friends: [Friend]) { self.friends = friends }

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

    /// Friend cap (D7): the design limit until a two-account run measures the real CloudKit one.
    nonisolated static let friendCap = 30

    /// The zone-wide share (created on first use). Read-only for participants, invite-only.
    func shareForInvite() async throws -> CKShare {
        guard Self.enabled else { throw DotError.iCloudOff }
        try await ensureZone()
        if let existing = try await existingShare() {
            if existing.participants.filter({ $0.acceptanceStatus != .removed }).count > Self.friendCap { throw DotError.friendCap }
            return existing
        }
        let share = CKShare(recordZoneID: zoneID)
        share[CKShare.SystemFieldKey.title] = "My Void dots"
        share.publicPermission = .none
        _ = try await container.privateCloudDatabase.modifyRecords(saving: [share], deleting: [])
        return share
    }

    private func existingShare() async throws -> CKShare? {
        let shareID = CKRecord.ID(recordName: CKRecordNameZoneWideShare, zoneID: zoneID)
        do { return try await container.privateCloudDatabase.record(for: shareID) as? CKShare }
        catch let error as CKError where error.code == .unknownItem || error.code == .zoneNotFound { return nil }
    }

    var cloudContainer: CKContainer { container }

    // MARK: Accept

    func accept(_ metadata: CKShare.Metadata) async {
        guard Self.enabled else { return }
        do {
            _ = try await container.accept([metadata])
            if let name = metadata.ownerIdentity.nameComponents.map({ PersonNameComponentsFormatter().string(from: $0) }), !name.isEmpty {
                var names = savedNames; names[metadata.share.recordID.zoneID.ownerName] = name; savedNames = names
            }
            status = ""
            await refreshFriends()
        } catch {
            status = "couldn't add friend: \(error.localizedDescription)"
        }
    }

    private var savedNames: [String: String] {
        get { UserDefaults.standard.dictionary(forKey: "friendNames") as? [String: String] ?? [:] }
        set { UserDefaults.standard.set(newValue, forKey: "friendNames") }
    }

    // MARK: Read friends' dots (pull on open)

    func refreshFriends() async {
        guard Self.enabled else { return }
        let db = container.sharedCloudDatabase
        do {
            let mine = Set((try? await existingShare())?.participants.compactMap { $0.userIdentity.userRecordID?.recordName } ?? [])
            var result: [Friend] = []
            for zone in try await db.allRecordZones() {
                let owner = zone.zoneID.ownerName
                var dots: [FriendDot] = []
                let changes = try await db.recordZoneChanges(inZoneWith: zone.zoneID, since: nil)
                for (_, change) in changes.modificationResultsByID {
                    guard let record = try? change.get().record, record.recordType == "Dot",
                          let createdAt = record["createdAt"] as? Date,
                          let raw = record["tags"] as? [String] else { continue }
                    dots.append(FriendDot(id: record.recordID.recordName, ownerID: owner, createdAt: createdAt,
                                          tags: raw.compactMap(Quadrant.init(rawValue:))))
                }
                var name = savedNames[owner]
                if name == nil, let share = try? await db.record(for: CKRecord.ID(recordName: CKRecordNameZoneWideShare, zoneID: zone.zoneID)) as? CKShare,
                   let c = share.owner.userIdentity.nameComponents {
                    let n = PersonNameComponentsFormatter().string(from: c)
                    if !n.isEmpty { name = n; var names = savedNames; names[owner] = n; savedNames = names }
                }
                result.append(Friend(id: owner, name: name ?? "friend", dots: dots.sorted { $0.createdAt > $1.createdAt },
                                     sharesBack: mine.contains(owner)))
            }
            friends = result.sorted { $0.name < $1.name }
            status = ""
        } catch {
            status = "couldn't load friends: \(error.localizedDescription)"
        }
    }
}

enum DotError: Error, LocalizedError {
    case iCloudOff, friendCap
    var errorDescription: String? {
        switch self {
        case .iCloudOff: "iCloud sharing is off in this build"
        case .friendCap: "You've reached the friend limit"
        }
    }
}

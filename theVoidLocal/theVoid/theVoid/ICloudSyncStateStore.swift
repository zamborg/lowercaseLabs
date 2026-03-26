import Foundation

actor ICloudSyncStateStore {
    private struct PersistedState: Codable {
        var zoneChangeTokenData: Data?
        var lastFullReconcileAt: String?
        var lastSyncAt: String?
        var operations: [ICloudSyncOperation]

        init(
            zoneChangeTokenData: Data? = nil,
            lastFullReconcileAt: String? = nil,
            lastSyncAt: String? = nil,
            operations: [ICloudSyncOperation] = []
        ) {
            self.zoneChangeTokenData = zoneChangeTokenData
            self.lastFullReconcileAt = lastFullReconcileAt
            self.lastSyncAt = lastSyncAt
            self.operations = operations
        }
    }

    private let stateURL: URL
    private var state: PersistedState

    init() {
        let fileManager = FileManager.default
        let baseRoot = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let directory = baseRoot.appendingPathComponent("VoidICloudSync", isDirectory: true)
        if !fileManager.fileExists(atPath: directory.path) {
            try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        stateURL = directory.appendingPathComponent("state.json")

        if let data = try? Data(contentsOf: stateURL),
           let decoded = try? JSONDecoder().decode(PersistedState.self, from: data) {
            state = decoded
        } else {
            state = PersistedState()
        }
    }

    func queuedOperations(for userID: String) -> [ICloudSyncOperation] {
        state.operations
            .filter { $0.userID == userID }
            .sorted { left, right in
                let leftDate = ICloudSyncTimestamp.date(from: left.queuedAt) ?? .distantPast
                let rightDate = ICloudSyncTimestamp.date(from: right.queuedAt) ?? .distantPast
                if leftDate != rightDate {
                    return leftDate < rightDate
                }
                return left.id < right.id
            }
    }

    func enqueue(_ operation: ICloudSyncOperation) {
        state.operations.removeAll { existing in
            existing.userID == operation.userID && existing.dedupeKey == operation.dedupeKey
        }
        state.operations.append(operation)
        persist()
    }

    func removeOperations(withIDs ids: Set<String>) {
        guard !ids.isEmpty else { return }
        state.operations.removeAll { ids.contains($0.id) }
        persist()
    }

    func setLastSyncAt(_ date: Date?) {
        state.lastSyncAt = date.map(ICloudSyncTimestamp.string(from:))
        persist()
    }

    func lastSyncAt() -> Date? {
        ICloudSyncTimestamp.date(from: state.lastSyncAt)
    }

    func setLastFullReconcileAt(_ date: Date?) {
        state.lastFullReconcileAt = date.map(ICloudSyncTimestamp.string(from:))
        persist()
    }

    func lastFullReconcileAt() -> Date? {
        ICloudSyncTimestamp.date(from: state.lastFullReconcileAt)
    }

    func setZoneChangeTokenData(_ data: Data?) {
        state.zoneChangeTokenData = data
        persist()
    }

    func zoneChangeTokenData() -> Data? {
        state.zoneChangeTokenData
    }

    private func persist() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let encoded = try? encoder.encode(state) else {
            return
        }
        try? encoded.write(to: stateURL, options: [.atomic])
    }
}

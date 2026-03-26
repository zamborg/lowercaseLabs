import Foundation
import Testing
@testable import theVoid

struct LocalJournalStoreHealthSnapshotTests {
    @Test func persistsAndPreservesHealthSnapshotAcrossUpdates() throws {
        let fileManager = FileManager.default
        let store = LocalJournalStore()
        let userID = "health-test-user-\(UUID().uuidString)"
        let journalUserDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
            .appendingPathComponent("VoidLocalJournal", isDirectory: true)
            .appendingPathComponent(userID, isDirectory: true)

        let draftURL = fileManager.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).m4a")
        try Data("audio".utf8).write(to: draftURL)
        defer {
            try? fileManager.removeItem(at: draftURL)
            try? fileManager.removeItem(at: journalUserDir)
        }

        let transcript = APITranscript(text: "Today was steady", providerMetadata: nil)
        let insight = APIInsight(
            moodScore: 0.32,
            moodTags: ["calm"],
            themes: ["focus"],
            signals: [:],
            safetyFlags: nil
        )
        let snapshot = EntryHealthSnapshot(
            capturedAtISO8601: "2026-03-02T11:59:00.000Z",
            readinessScore: 72,
            confidence: 0.88,
            components: [
                HealthComponentSnapshot(
                    type: .sleepHours,
                    rawValue: 7.2,
                    unit: "h",
                    componentScore: 79,
                    sampledAtISO8601: "2026-03-02T10:30:00.000Z",
                    isStale: false
                )
            ],
            version: 1
        )

        let saved = try store.saveReflection(
            for: userID,
            draftURL: draftURL,
            durationSeconds: 90,
            transcript: transcript,
            insight: insight,
            localDate: "2026-03-02",
            wasSharedToSocial: false,
            healthSnapshot: snapshot
        )

        let listedAfterSave = store.listEntries(for: userID)
        #expect(listedAfterSave.count == 1)
        #expect(listedAfterSave.first?.healthSnapshot == snapshot)

        _ = try store.updateEntryTitle(for: userID, entryID: saved.id, title: "Renamed")
        let listedAfterTitle = store.listEntries(for: userID)
        #expect(listedAfterTitle.first?.healthSnapshot == snapshot)

        _ = try store.updateEntryTags(for: userID, entryID: saved.id, moodTags: ["anxious", "hopeful"])
        let listedAfterTags = store.listEntries(for: userID)
        #expect(listedAfterTags.first?.healthSnapshot == snapshot)

        _ = try store.updateEntryAnalysis(
            for: userID,
            entryID: saved.id,
            transcript: APITranscript(text: "Updated transcript", providerMetadata: nil),
            insight: insight
        )
        let listedAfterAnalysis = store.listEntries(for: userID)
        #expect(listedAfterAnalysis.first?.healthSnapshot == snapshot)
    }
}

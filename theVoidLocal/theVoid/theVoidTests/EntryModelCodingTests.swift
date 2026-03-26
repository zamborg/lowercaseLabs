import Foundation
import Testing
@testable import theVoid

struct EntryModelCodingTests {
    @Test func decodesLegacyEntryWithoutHealthSnapshot() throws {
        let legacyJSON = """
        {
          "id": "entry-1",
          "localDate": "2026-03-02",
          "durationSeconds": 60,
          "status": "ready",
          "createdAt": "2026-03-02T12:00:00Z",
          "title": "Legacy entry",
          "transcript": null,
          "insight": null
        }
        """.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(APIEntry.self, from: legacyJSON)
        #expect(decoded.healthSnapshot == nil)
        #expect(decoded.id == "entry-1")
    }

    @Test func roundTripsEntryWithHealthSnapshot() throws {
        let snapshot = EntryHealthSnapshot(
            capturedAtISO8601: "2026-03-02T12:00:00.000Z",
            readinessScore: 78,
            confidence: 0.82,
            components: [
                HealthComponentSnapshot(
                    type: .sleepHours,
                    rawValue: 7.8,
                    unit: "h",
                    componentScore: 84,
                    sampledAtISO8601: "2026-03-02T11:00:00.000Z",
                    isStale: false
                )
            ],
            version: 1
        )

        let entry = APIEntry(
            id: "entry-2",
            localDate: "2026-03-02",
            durationSeconds: 120,
            status: "ready",
            createdAt: "2026-03-02T12:01:00Z",
            transcript: APITranscript(text: "hello", providerMetadata: nil),
            insight: APIInsight(
                moodScore: 0.4,
                moodTags: ["calm"],
                themes: [],
                signals: [:],
                safetyFlags: nil
            ),
            title: "Entry With Health",
            healthSnapshot: snapshot
        )

        let encoded = try JSONEncoder().encode(entry)
        let decoded = try JSONDecoder().decode(APIEntry.self, from: encoded)

        #expect(decoded.healthSnapshot == snapshot)
        #expect(decoded.id == entry.id)
        #expect(decoded.title == entry.title)
    }
}

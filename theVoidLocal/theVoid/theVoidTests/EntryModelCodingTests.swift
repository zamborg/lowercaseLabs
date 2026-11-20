import Foundation
import Testing
@testable import theVoid

struct EntryModelCodingTests {
    @Test func decodesEntryCreatedBeforeHealthIntegration() throws {
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
        #expect(decoded.id == "entry-1")
    }

    @Test func roundTripsEntryWithoutRemovedHealthField() throws {
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
            title: "A regular entry"
        )

        let encoded = try JSONEncoder().encode(entry)
        let decoded = try JSONDecoder().decode(APIEntry.self, from: encoded)

        let encodedJSON = try #require(String(data: encoded, encoding: .utf8))
        #expect(!encodedJSON.contains("healthSnapshot"))
        #expect(decoded.id == entry.id)
        #expect(decoded.title == entry.title)
    }

    @Test func ignoresSnapshotDataFromOlderAppVersions() throws {
        let legacyJSON = """
        {
          "id": "entry-with-old-snapshot",
          "localDate": "2026-03-02",
          "durationSeconds": 90,
          "status": "ready",
          "createdAt": "2026-03-02T12:00:00Z",
          "title": "Existing entry",
          "transcript": null,
          "insight": null,
          "healthSnapshot": {
            "capturedAtISO8601": "2026-03-02T12:00:00Z",
            "readinessScore": 78,
            "confidence": 0.82,
            "components": [],
            "version": 1
          }
        }
        """.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(APIEntry.self, from: legacyJSON)
        #expect(decoded.id == "entry-with-old-snapshot")
        #expect(decoded.displayTitle == "Existing entry")
    }
}

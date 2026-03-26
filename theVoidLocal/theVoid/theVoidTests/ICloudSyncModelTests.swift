import Foundation
import Testing
@testable import theVoid

struct ICloudSyncModelTests {
    @Test
    func storedLocalEntryLegacyDecodeUsesDefaults() throws {
        let payload = """
        {
          "entry": {
            "id": "entry-1",
            "localDate": "2026-03-01",
            "durationSeconds": 60,
            "status": "ready",
            "createdAt": "2026-03-01T12:00:00Z",
            "title": "Entry",
            "transcript": null,
            "insight": null
          },
          "audioFileName": "entry-1.m4a",
          "wasSharedToSocial": true
        }
        """

        let data = try #require(payload.data(using: .utf8))
        let decoded = try JSONDecoder().decode(StoredLocalEntry.self, from: data)

        #expect(decoded.audioFileName == "entry-1.m4a")
        #expect(decoded.wasSharedToSocial == true)
        #expect(decoded.updatedAt == "2026-03-01T12:00:00Z")
        #expect(decoded.isDeleted == false)
        #expect(decoded.cloudRecordChangeTag == nil)
    }

    @Test
    func timestampParserSupportsFractionalAndBasicISO8601() {
        let basic = ICloudSyncTimestamp.date(from: "2026-03-01T12:00:00Z")
        let fractional = ICloudSyncTimestamp.date(from: "2026-03-01T12:00:00.123Z")

        #expect(basic != nil)
        #expect(fractional != nil)
    }

    @Test
    func operationDedupeKeyIncludesKindAndIdentifiers() {
        let left = ICloudSyncOperation(kind: .upsertEntry, userID: "u1", entryID: "e1")
        let right = ICloudSyncOperation(kind: .upsertEntry, userID: "u1", entryID: "e1")
        let different = ICloudSyncOperation(kind: .deleteEntry, userID: "u1", entryID: "e1")

        #expect(left.dedupeKey == right.dedupeKey)
        #expect(left.dedupeKey != different.dedupeKey)
    }
}

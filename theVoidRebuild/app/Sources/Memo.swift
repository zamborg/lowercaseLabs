import Foundation
import SwiftData

@Model
final class Memo {
    // CloudKit-backed SwiftData: no unique constraints, every property has a default.
    var id: UUID = UUID()
    var createdAt: Date = Date()
    var duration: TimeInterval = 0
    var audioFileName: String = ""
    var transcript: String = ""
    var title: String = ""
    /// Quadrant raw values, always from the fixed enum.
    var tagValues: [String] = []

    init(id: UUID, createdAt: Date, duration: TimeInterval, audioFileName: String, transcript: String,
         title: String = "", tagValues: [String] = []) {
        self.id = id
        self.createdAt = createdAt
        self.duration = duration
        self.audioFileName = audioFileName
        self.transcript = transcript
        self.title = title
        self.tagValues = tagValues
    }

    var tags: [Quadrant] { tagValues.compactMap(Quadrant.init(rawValue:)) }
    var audioURL: URL { Recorder.audioDirectory().appendingPathComponent(audioFileName) }
}

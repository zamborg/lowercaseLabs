import AVFoundation
import Foundation

@MainActor
final class DefaultAudioCaptureService: NSObject, AudioCaptureService {
    private let fileManager = FileManager.default
    private var recorder: AVAudioRecorder?
    private var currentURL: URL?

    var isRecording: Bool {
        recorder?.isRecording == true
    }

    func requestPermission() async -> Bool {
        switch AVAudioApplication.shared.recordPermission {
        case .granted:
            return true
        case .denied:
            return false
        case .undetermined:
            break
        @unknown default:
            return false
        }

        return await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    func startRecording() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let targetURL = makeDraftURL()
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 12_000,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
        ]
        let recorder = try AVAudioRecorder(url: targetURL, settings: settings)
        recorder.prepareToRecord()
        guard recorder.record() else {
            throw APIError.transport("Microphone recording did not start.")
        }
        self.recorder = recorder
        self.currentURL = targetURL
    }

    func stopRecording() throws -> URL {
        recorder?.stop()
        recorder = nil
        guard let currentURL else {
            throw APIError.transport("Missing recording output")
        }
        self.currentURL = nil
        return currentURL
    }

    private func makeDraftURL() -> URL {
        let root = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
            .appendingPathComponent("BlackholeDrafts", isDirectory: true)
        if !fileManager.fileExists(atPath: root.path) {
            try? fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        }
        return root.appendingPathComponent("\(UUID().uuidString).m4a")
    }
}

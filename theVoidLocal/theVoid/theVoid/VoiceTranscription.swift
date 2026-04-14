import AVFoundation
import Foundation
import WhisperKit

// MARK: - WhisperKit runtime for file-based transcription

actor WhisperTranscriptionRuntime {
    static let shared = WhisperTranscriptionRuntime()

    private var whisperKit: WhisperKit?

    var isReady: Bool { whisperKit != nil }

    func prepare() async throws {
        guard whisperKit == nil else { return }
        whisperKit = try await WhisperKit(
            model: "tiny.en",
            downloadBase: try modelDirectory(),
            verbose: false,
            prewarm: true,
            load: true,
            download: true
        )
    }

    func transcribe(audioURL: URL) async throws -> String {
        try await prepare()
        guard let whisperKit else {
            throw VoiceTranscriptionError.notReady
        }
        let results = try await whisperKit.transcribe(
            audioPath: audioURL.path,
            decodeOptions: DecodingOptions(
                verbose: false,
                task: .transcribe,
                language: "en",
                withoutTimestamps: true,
                wordTimestamps: false
            )
        )
        let joined = results.map(\.text).joined(separator: " ")
        return joined.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func unload() async {
        if let w = whisperKit { await w.unloadModels() }
        whisperKit = nil
    }

    private func modelDirectory() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let dir = base.appendingPathComponent("WhisperKitModels", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

enum VoiceTranscriptionError: LocalizedError {
    case notReady

    var errorDescription: String? {
        "Transcription model is not ready. Download it from Settings."
    }
}

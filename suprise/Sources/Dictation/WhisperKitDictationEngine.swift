import Foundation
import WhisperKit

final class WhisperKitDictationEngine: DictationEngine {
    let kind: DictationEngineKind = .whisperKit

    private let runtime = WhisperKitRuntime()

    func prepare() async throws {
        try await runtime.prepare()
    }

    func makeSession(locale: Locale) throws -> DictationSession {
        StreamingWhisperSession(transcriber: runtime, locale: locale)
    }

    func unload() async {
        await runtime.unload()
    }
}

actor WhisperKitRuntime: StreamingWhisperTranscriber {
    private var whisperKit: WhisperKit?

    func prepare() async throws {
        guard whisperKit == nil else {
            return
        }

        let whisperKit = try await WhisperKit(
            model: WhisperModelSelection.modelName,
            downloadBase: try modelDirectory(),
            verbose: false,
            prewarm: true,
            load: true,
            download: true
        )

        self.whisperKit = whisperKit
    }

    func transcribe(samples: [Float], locale _: Locale) async throws -> String {
        guard !samples.isEmpty else {
            return ""
        }

        try await prepare()

        guard let whisperKit else {
            throw DictationError.engineUnavailable("WhisperKit failed to load its model.")
        }

        let results = try await whisperKit.transcribe(
            audioArray: samples,
            decodeOptions: DecodingOptions(
                verbose: false,
                task: .transcribe,
                language: WhisperModelSelection.languageCode,
                withoutTimestamps: true,
                wordTimestamps: false
            )
        )

        return results
            .map(\.text)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func unload() async {
        if let whisperKit {
            await whisperKit.unloadModels()
        }
        whisperKit = nil
    }

    private func modelDirectory() throws -> URL {
        let baseDirectory = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let modelDirectory = baseDirectory.appendingPathComponent("WhisperKitModels", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory,
            withIntermediateDirectories: true,
            attributes: nil
        )
        return modelDirectory
    }
}

enum WhisperModelSelection {
    static let modelName = "tiny.en"
    static let languageCode = "en"
}

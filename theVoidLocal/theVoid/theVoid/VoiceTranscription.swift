import AVFoundation
import Foundation
import WhisperKit

// MARK: - Shared WhisperKit runtime

actor WhisperTranscriptionRuntime: StreamingWhisperTranscribing {
    static let shared = WhisperTranscriptionRuntime()

    private var whisperKit: WhisperKit?
    private var prepareTask: Task<Void, Error>?
    private var preparingConfiguration: WhisperEngineConfiguration?
    private var loadedConfiguration: WhisperEngineConfiguration?

    var isReady: Bool { whisperKit != nil }

    func prepare() async throws {
        try await prepare(configuration: .english)
    }

    func prepare(configuration: WhisperEngineConfiguration) async throws {
        if whisperKit != nil, loadedConfiguration == configuration {
            return
        }
        if let prepareTask {
            let inFlightConfiguration = preparingConfiguration
            try await prepareTask.value
            if whisperKit != nil, inFlightConfiguration == configuration {
                self.prepareTask = nil
                preparingConfiguration = nil
                loadedConfiguration = configuration
                return
            }
        }

        if whisperKit != nil {
            await unload()
        }
        let task = Task { [self] in
            try await initializeWhisperKit(configuration: configuration)
        }
        prepareTask = task
        preparingConfiguration = configuration
        do {
            try await task.value
            prepareTask = nil
            preparingConfiguration = nil
            loadedConfiguration = configuration
        } catch {
            prepareTask = nil
            preparingConfiguration = nil
            throw error
        }
    }

    private func initializeWhisperKit(configuration: WhisperEngineConfiguration) async throws {
        whisperKit = try await WhisperKit(
            model: configuration.modelName,
            downloadBase: try modelDirectory(),
            verbose: false,
            prewarm: true,
            load: true,
            download: true
        )
    }

    func transcribe(audioURL: URL) async throws -> String {
        try await prepareCurrentConfiguration()
        guard let whisperKit else {
            throw VoiceTranscriptionError.notReady
        }
        let results = try await whisperKit.transcribe(
            audioPath: audioURL.path,
            decodeOptions: decodingOptions()
        )
        let joined = results.map(\.text).joined(separator: " ")
        return joined.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func transcribe(samples: [Float]) async throws -> String {
        guard !samples.isEmpty else { return "" }
        try await prepareCurrentConfiguration()
        guard let whisperKit else {
            throw VoiceTranscriptionError.notReady
        }
        let results = try await whisperKit.transcribe(
            audioArray: samples,
            decodeOptions: decodingOptions()
        )
        return results.map(\.text)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func transcribePartial(
        samples: [Float],
        context _: StreamingTranscriptionContext
    ) async throws -> StreamingTranscriptionHypothesis {
        guard !samples.isEmpty else {
            return StreamingTranscriptionHypothesis(text: "", words: [])
        }
        try await prepareCurrentConfiguration()
        guard let whisperKit else {
            throw VoiceTranscriptionError.notReady
        }

        let results = try await whisperKit.transcribe(
            audioArray: samples,
            decodeOptions: decodingOptions()
        )
        let text = results.map(\.text)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return StreamingTranscriptionHypothesis(text: text, words: [])
    }

    func unload() async {
        prepareTask?.cancel()
        prepareTask = nil
        preparingConfiguration = nil
        if let w = whisperKit { await w.unloadModels() }
        whisperKit = nil
        loadedConfiguration = nil
    }

    private func prepareCurrentConfiguration() async throws {
        let configuration = loadedConfiguration
            ?? preparingConfiguration
            ?? .english
        try await prepare(configuration: configuration)
    }

    private func decodingOptions(wordTimestamps: Bool = false) -> DecodingOptions {
        DecodingOptions(
            verbose: false,
            task: .transcribe,
            language: loadedConfiguration?.languageCode ?? "en",
            withoutTimestamps: !wordTimestamps,
            wordTimestamps: wordTimestamps
        )
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

struct WhisperEngineConfiguration: Equatable, Sendable {
    let modelName: String
    let languageCode: String

    static let english = WhisperEngineConfiguration(
        modelName: "tiny.en",
        languageCode: "en"
    )

    init(language: TranscriptionLanguage) {
        modelName = language.whisperModelName
        languageCode = language.languageCode
    }

    private init(modelName: String, languageCode: String) {
        self.modelName = modelName
        self.languageCode = languageCode
    }
}

final class WhisperKitTranscriptionEngine: TranscriptionEngine {
    let kind: TranscriptionEngineKind = .whisperKit

    private let runtime: WhisperTranscriptionRuntime
    private var isPrepared = false

    init(runtime: WhisperTranscriptionRuntime = WhisperTranscriptionRuntime()) {
        self.runtime = runtime
    }

    func prepare(language: TranscriptionLanguage) async throws {
        try await runtime.prepare(configuration: WhisperEngineConfiguration(language: language))
        isPrepared = true
    }

    func makeSession(inputFormat _: AVAudioFormat) throws -> any LiveTranscriptionSession {
        guard isPrepared else {
            throw TranscriptionEngineError.engineNotReady
        }
        return StreamingWhisperSession(transcriber: runtime)
    }

    func transcribe(audioURL: URL) async throws -> String {
        guard isPrepared else {
            throw TranscriptionEngineError.engineNotReady
        }
        return try await runtime.transcribe(audioURL: audioURL)
    }

    func unload() async {
        isPrepared = false
        await runtime.unload()
    }
}

enum VoiceTranscriptionError: LocalizedError {
    case notReady

    var errorDescription: String? {
        "Transcription model is not ready. Download it from Settings."
    }
}

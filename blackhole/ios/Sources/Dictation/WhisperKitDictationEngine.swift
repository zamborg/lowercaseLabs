import Foundation
import WhisperKit

final class WhisperKitDictationEngine: DictationEngine {
    let kind: DictationEngineKind = .whisperKit
    private let runtime: WhisperKitRuntime

    init(runtime: WhisperKitRuntime = .shared) {
        self.runtime = runtime
    }

    func prepare() async throws { try await runtime.prepare() }
    func makeSession(locale: Locale) throws -> DictationSession { StreamingWhisperSession(transcriber: runtime, locale: locale) }
    func unload() async {}
}

actor WhisperKitRuntime: StreamingWhisperTranscriber {
    static let shared = WhisperKitRuntime()

    private var whisperKit: WhisperKit?
    private var prepareTask: Task<Void, Error>?

    func prepare() async throws {
        guard whisperKit == nil else { return }
        if let prepareTask {
            return try await prepareTask.value
        }

        let task = Task { [self] in
            try await self.initializeWhisperKit()
        }
        prepareTask = task
        do {
            try await task.value
            prepareTask = nil
        } catch {
            prepareTask = nil
            throw error
        }
    }

    func transcribe(samples: [Float], locale _: Locale) async throws -> String {
        guard !samples.isEmpty else { return "" }
        try await prepare()
        guard let whisperKit else { throw DictationError.engineUnavailable("WhisperKit failed to load.") }

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
        return results.map(\.text).joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func unload() async {
        if let w = whisperKit { await w.unloadModels() }
        whisperKit = nil
    }

    private func initializeWhisperKit() async throws {
        let base = try modelDirectory()

        if shouldPreferLocalCache(in: base) {
            do {
                whisperKit = try await makeWhisperKit(downloadBase: base, download: false)
                try? markModelReady(in: base)
                return
            } catch {
                clearModelReadyMarker(in: base)
            }
        }

        whisperKit = try await makeWhisperKit(downloadBase: base, download: true)
        try? markModelReady(in: base)
    }

    private func makeWhisperKit(downloadBase: URL, download: Bool) async throws -> WhisperKit {
        try await WhisperKit(
            model: WhisperModelSelection.modelName,
            downloadBase: downloadBase,
            verbose: false,
            prewarm: true,
            load: true,
            download: download
        )
    }

    private func modelDirectory() throws -> URL {
        let base = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dir = base.appendingPathComponent("WhisperKitModels", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func shouldPreferLocalCache(in base: URL) -> Bool {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: modelReadyMarker(in: base).path) {
            return true
        }

        if let enumerator = fileManager.enumerator(
            at: base,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for case let url as URL in enumerator {
                if url.lastPathComponent.localizedCaseInsensitiveContains(WhisperModelSelection.modelName) {
                    return true
                }
            }
        }

        return false
    }

    private func modelReadyMarker(in base: URL) -> URL {
        base.appendingPathComponent(".\(WhisperModelSelection.modelName)-ready")
    }

    private func markModelReady(in base: URL) throws {
        let marker = modelReadyMarker(in: base)
        try Data("ready".utf8).write(to: marker, options: [.atomic])
    }

    private func clearModelReadyMarker(in base: URL) {
        try? FileManager.default.removeItem(at: modelReadyMarker(in: base))
    }
}

enum WhisperModelSelection {
    static let modelName = "tiny.en"
    static let languageCode = "en"
}

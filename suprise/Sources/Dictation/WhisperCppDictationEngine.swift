import Foundation
import whisper

final class WhisperCppDictationEngine: DictationEngine {
    let kind: DictationEngineKind = .whisperCpp

    private let runtime = WhisperCppRuntime()

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

actor WhisperCppRuntime: StreamingWhisperTranscriber {
    private var context: OpaquePointer?

    func prepare() async throws {
        guard context == nil else {
            return
        }

        let modelURL = try await WhisperCppModelStore.ensureModelDownloaded()
        var contextParams = whisper_context_default_params()
        #if targetEnvironment(simulator)
        contextParams.use_gpu = false
        #else
        contextParams.flash_attn = true
        #endif

        let loadedContext = modelURL.path.withCString { pathPointer in
            whisper_init_from_file_with_params(pathPointer, contextParams)
        }

        guard let loadedContext else {
            throw DictationError.engineUnavailable("whisper.cpp could not load \(WhisperCppModelStore.fileName).")
        }

        context = loadedContext
    }

    func transcribe(samples: [Float], locale _: Locale) async throws -> String {
        guard !samples.isEmpty else {
            return ""
        }

        try await prepare()

        guard let context else {
            throw DictationError.engineUnavailable("whisper.cpp is unavailable because its model context was not created.")
        }

        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.print_realtime = false
        params.print_progress = false
        params.print_timestamps = false
        params.print_special = false
        params.translate = false
        params.no_context = true
        params.no_timestamps = true
        params.single_segment = false
        params.suppress_blank = true
        params.n_threads = Int32(max(1, min(8, ProcessInfo.processInfo.processorCount - 2)))

        let resultCode = WhisperModelSelection.languageCode.withCString { languagePointer in
            params.language = languagePointer
            whisper_reset_timings(context)
            return samples.withUnsafeBufferPointer { samplePointer in
                whisper_full(context, params, samplePointer.baseAddress, Int32(samplePointer.count))
            }
        }

        guard resultCode == 0 else {
            throw DictationError.engineUnavailable("whisper.cpp failed during transcription.")
        }

        let segmentCount = Int(whisper_full_n_segments(context))
        let text = (0..<segmentCount)
            .map { String(cString: whisper_full_get_segment_text(context, Int32($0))) }
            .joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return text
    }

    func unload() async {
        if let context {
            whisper_free(context)
        }
        context = nil
    }
}

private enum WhisperCppModelStore {
    static let fileName = "ggml-tiny.en.bin"
    private static let remoteURL = URL(string: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin")!

    static func ensureModelDownloaded() async throws -> URL {
        let destinationURL = try localModelURL()
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            return destinationURL
        }

        let (temporaryURL, response) = try await URLSession.shared.download(from: remoteURL)
        if let httpResponse = response as? HTTPURLResponse, !(200 ... 299).contains(httpResponse.statusCode) {
            throw DictationError.engineUnavailable("whisper.cpp model download failed with status \(httpResponse.statusCode).")
        }

        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }

        try FileManager.default.moveItem(at: temporaryURL, to: destinationURL)
        return destinationURL
    }

    private static func localModelURL() throws -> URL {
        let baseDirectory = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let directory = baseDirectory.appendingPathComponent("WhisperCppModels", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true,
            attributes: nil
        )
        return directory.appendingPathComponent(fileName)
    }
}

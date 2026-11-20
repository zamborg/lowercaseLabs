import AVFoundation
import Speech

enum CaptureError: Error { case micDenied, speechUnavailable, notWarm, micUnavailable }

/// One live transcription session: feed it PCM buffers, read streaming text, finalize on release.
/// Single implementation on purpose: SpeechAnalyzer + SpeechTranscriber. No protocol.
final class LiveTranscription: @unchecked Sendable {
    /// Called on the main queue with (finalized + volatile) text so far.
    var onText: (@Sendable (String) -> Void)?

    private let analyzer: SpeechAnalyzer
    private let continuation: AsyncStream<AnalyzerInput>.Continuation
    private let converter: AVAudioConverter
    private let outputFormat: AVAudioFormat
    private var analysisTask: Task<Void, Error>?
    private var resultsTask: Task<String, Never>?

    /// Resources that are slow to set up. Do this once at app open, not on press.
    struct Warm {
        let locale: Locale
        let format: AVAudioFormat
    }

    static func warm() async throws -> Warm {
        guard SpeechTranscriber.isAvailable else { throw CaptureError.speechUnavailable }
        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale.current) else {
            throw CaptureError.speechUnavailable
        }
        let probe = SpeechTranscriber(locale: locale, preset: .progressiveTranscription)
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [probe]) {
            try await request.downloadAndInstall()
        }
        _ = try await AssetInventory.reserve(locale: locale)
        guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [probe]) else {
            throw CaptureError.speechUnavailable
        }
        return Warm(locale: locale, format: format)
    }

    init(warm: Warm, inputFormat: AVAudioFormat) throws {
        let transcriber = SpeechTranscriber(locale: warm.locale, preset: .progressiveTranscription)
        analyzer = SpeechAnalyzer(modules: [transcriber])
        outputFormat = warm.format
        guard let converter = AVAudioConverter(from: inputFormat, to: warm.format) else {
            throw CaptureError.speechUnavailable
        }
        self.converter = converter
        let stream = AsyncStream.makeStream(of: AnalyzerInput.self, bufferingPolicy: .unbounded)
        continuation = stream.continuation

        let analyzer = analyzer
        analysisTask = Task { try await analyzer.start(inputSequence: stream.stream) }
        resultsTask = Task { [weak self] in
            var finalized = ""
            var volatile = ""
            do {
                for try await result in transcriber.results {
                    let text = String(result.text.characters)
                    if result.isFinal {
                        finalized += text
                        volatile = ""
                    } else {
                        volatile = text
                    }
                    let combined = finalized + volatile
                    DispatchQueue.main.async { self?.onText?(combined) }
                }
            } catch {}
            return (finalized + volatile).trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    /// Called from the audio render thread. Must not block.
    func append(_ buffer: AVAudioPCMBuffer) {
        let ratio = outputFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return }
        var consumed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if consumed { status.pointee = .noDataNow; return nil }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        if error == nil, out.frameLength > 0 {
            continuation.yield(AnalyzerInput(buffer: out))
        }
    }

    /// Release: close input, let the analyzer finalize, return the final transcript.
    func finish() async -> String {
        continuation.finish()
        try? await analyzer.finalizeAndFinishThroughEndOfInput()
        _ = try? await analysisTask?.value
        return await resultsTask?.value ?? ""
    }
}

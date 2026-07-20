import AVFoundation
import Foundation
import Speech

@available(iOS 26.0, *)
final class AppleSpeechTranscriptionEngine: TranscriptionEngine {
    let kind: TranscriptionEngineKind = .apple

    private var selectedLocale: Locale?
    private var reservedLocale: Locale?
    private var analyzerFormat: AVAudioFormat?

    func prepare(language: TranscriptionLanguage) async throws {
        guard SpeechTranscriber.isAvailable else {
            throw TranscriptionEngineError.engineUnavailable
        }
        guard let locale = await SpeechTranscriber.supportedLocale(
            equivalentTo: language.resolvedLocale
        ) else {
            throw TranscriptionEngineError.languageUnavailable(language.title)
        }

        let transcriber = SpeechTranscriber(
            locale: locale,
            preset: .progressiveTranscription
        )
        if let request = try await AssetInventory.assetInstallationRequest(
            supporting: [transcriber]
        ) {
            try await request.downloadAndInstall()
        }
        _ = try await AssetInventory.reserve(locale: locale)
        reservedLocale = locale
        guard let analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(
            compatibleWith: [transcriber]
        ) else {
            throw TranscriptionEngineError.audioFormatUnavailable
        }
        selectedLocale = locale
        self.analyzerFormat = analyzerFormat
    }

    func makeSession(inputFormat _: AVAudioFormat) throws -> any LiveTranscriptionSession {
        guard let selectedLocale, let analyzerFormat else {
            throw TranscriptionEngineError.engineNotReady
        }
        return AppleSpeechAnalyzerSession(
            locale: selectedLocale,
            analyzerFormat: analyzerFormat
        )
    }

    func transcribe(audioURL: URL) async throws -> String {
        guard let selectedLocale else {
            throw TranscriptionEngineError.engineNotReady
        }
        return try await AppleSpeechAnalyzerSession.transcribeFile(
            at: audioURL,
            locale: selectedLocale
        )
    }

    func unload() async {
        if let reservedLocale {
            _ = await AssetInventory.release(reservedLocale: reservedLocale)
        }
        selectedLocale = nil
        reservedLocale = nil
        analyzerFormat = nil
        await SpeechModels.endRetention()
    }
}

@available(iOS 26.0, *)
private final class AppleSpeechAnalyzerSession: LiveTranscriptionSession {
    var onUpdate: ((String) -> Void)?

    private let analyzer: SpeechAnalyzer
    private let inputContinuation: AsyncStream<AnalyzerInput>.Continuation
    private let accumulator = AppleTranscriptAccumulator()
    private let converter: AppleSpeechBufferConverter
    private let analyzerFormat: AVAudioFormat
    private let stateLock = NSLock()
    private var analysisTask: Task<Void, Error>?
    private var resultsTask: Task<Void, Never>?
    private var isFinished = false

    init(
        locale: Locale,
        analyzerFormat: AVAudioFormat
    ) {
        let transcriber = SpeechTranscriber(
            locale: locale,
            preset: .progressiveTranscription
        )
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let input = AsyncStream.makeStream(
            of: AnalyzerInput.self,
            bufferingPolicy: .bufferingNewest(32)
        )

        self.analyzer = analyzer
        self.analyzerFormat = analyzerFormat
        converter = AppleSpeechBufferConverter(outputFormat: analyzerFormat)
        inputContinuation = input.continuation
        analysisTask = Task {
            try await analyzer.prepareToAnalyze(in: analyzerFormat)
            try await analyzer.start(inputSequence: input.stream)
        }
        resultsTask = Task { [weak self, transcriber] in
            do {
                for try await result in transcriber.results {
                    guard let self else { return }
                    let text = accumulator.consume(result)
                    publish(text)
                }
            } catch is CancellationError {
                return
            } catch {
                self?.accumulator.store(error: error)
            }
        }
    }

    func append(_ buffer: AVAudioPCMBuffer) throws {
        let canAppend = stateLock.withLock { !isFinished }
        guard canAppend else { return }
        guard let copy = buffer.transcriptionCopy() else {
            throw TranscriptionEngineError.audioBufferCopyFailed
        }
        let converted = try converter.convert(copy)
        inputContinuation.yield(AnalyzerInput(buffer: converted))
    }

    func finish() async throws -> String {
        let shouldFinish = stateLock.withLock { () -> Bool in
            guard !isFinished else { return false }
            isFinished = true
            return true
        }

        if shouldFinish {
            inputContinuation.finish()
            try await analyzer.finalizeAndFinishThroughEndOfInput()
        }
        try await analysisTask?.value
        await resultsTask?.value

        if let error = accumulator.error {
            throw error
        }
        let text = accumulator.latestText
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            throw TranscriptionEngineError.noSpeechDetected
        }
        return text
    }

    func cancel() {
        let shouldCancel = stateLock.withLock { () -> Bool in
            guard !isFinished else { return false }
            isFinished = true
            return true
        }
        guard shouldCancel else { return }

        inputContinuation.finish()
        analysisTask?.cancel()
        resultsTask?.cancel()
        Task { [analyzer] in
            await analyzer.cancelAndFinishNow()
        }
    }

    private func publish(_ text: String) {
        DispatchQueue.main.async { [weak self] in
            self?.onUpdate?(text)
        }
    }

    static func transcribeFile(at url: URL, locale: Locale) async throws -> String {
        let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let audioFile = try AVAudioFile(forReading: url)
        let accumulator = AppleTranscriptAccumulator()
        let resultsTask = Task {
            do {
                for try await result in transcriber.results {
                    _ = accumulator.consume(result)
                }
            } catch {
                accumulator.store(error: error)
            }
        }

        try await analyzer.prepareToAnalyze(in: audioFile.processingFormat)
        try await analyzer.start(inputAudioFile: audioFile, finishAfterFile: true)
        await resultsTask.value

        if let error = accumulator.error {
            throw error
        }
        let text = accumulator.latestText
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            throw TranscriptionEngineError.noSpeechDetected
        }
        return text
    }
}

@available(iOS 26.0, *)
private final class AppleSpeechBufferConverter {
    private let outputFormat: AVAudioFormat
    private var converter: AVAudioConverter?

    init(outputFormat: AVAudioFormat) {
        self.outputFormat = outputFormat
    }

    func convert(_ buffer: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
        guard buffer.format != outputFormat else {
            return buffer
        }

        if converter?.inputFormat != buffer.format
            || converter?.outputFormat != outputFormat {
            converter = AVAudioConverter(from: buffer.format, to: outputFormat)
            converter?.primeMethod = .none
        }
        guard let converter else {
            throw TranscriptionEngineError.audioFormatUnavailable
        }

        let ratio = converter.outputFormat.sampleRate / converter.inputFormat.sampleRate
        let frameCapacity = AVAudioFrameCount(
            (Double(buffer.frameLength) * ratio).rounded(.up)
        )
        guard let converted = AVAudioPCMBuffer(
            pcmFormat: converter.outputFormat,
            frameCapacity: max(1, frameCapacity)
        ) else {
            throw TranscriptionEngineError.audioFormatUnavailable
        }

        var didProvideInput = false
        var conversionError: NSError?
        let status = converter.convert(to: converted, error: &conversionError) { _, inputStatus in
            defer { didProvideInput = true }
            inputStatus.pointee = didProvideInput ? .noDataNow : .haveData
            return didProvideInput ? nil : buffer
        }
        if status == .error {
            if let conversionError {
                throw conversionError
            }
            throw TranscriptionEngineError.audioFormatUnavailable
        }
        return converted
    }
}

@available(iOS 26.0, *)
private final class AppleTranscriptAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private var finalizedText = ""
    private var volatileText = ""
    private var storedError: Error?

    var latestText: String {
        lock.withLock { joinedText() }
    }

    var error: Error? {
        lock.withLock { storedError }
    }

    func consume(_ result: SpeechTranscriber.Result) -> String {
        let fragment = String(result.text.characters)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return lock.withLock {
            if result.isFinal {
                finalizedText = Self.join(finalizedText, fragment)
                volatileText = ""
            } else {
                volatileText = fragment
            }
            return joinedText()
        }
    }

    func store(error: Error) {
        lock.withLock {
            storedError = error
        }
    }

    private func joinedText() -> String {
        Self.join(finalizedText, volatileText)
    }

    private static func join(_ prefix: String, _ suffix: String) -> String {
        if prefix.isEmpty { return suffix }
        if suffix.isEmpty { return prefix }
        return "\(prefix) \(suffix)"
    }
}

private extension AVAudioPCMBuffer {
    func transcriptionCopy() -> AVAudioPCMBuffer? {
        guard let copy = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameLength
        ) else {
            return nil
        }
        copy.frameLength = frameLength

        let sourceBuffers = UnsafeMutableAudioBufferListPointer(mutableAudioBufferList)
        let destinationBuffers = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
        guard sourceBuffers.count == destinationBuffers.count else {
            return nil
        }

        for index in sourceBuffers.indices {
            let source = sourceBuffers[index]
            var destination = destinationBuffers[index]
            guard let sourceData = source.mData,
                  let destinationData = destination.mData else {
                return nil
            }
            let byteCount = min(source.mDataByteSize, destination.mDataByteSize)
            memcpy(destinationData, sourceData, Int(byteCount))
            destination.mDataByteSize = byteCount
            destinationBuffers[index] = destination
        }
        return copy
    }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock()
        defer { unlock() }
        return body()
    }
}

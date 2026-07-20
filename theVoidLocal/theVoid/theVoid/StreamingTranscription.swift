import AVFoundation
import Foundation

struct StreamingTranscriptionWord: Equatable, Sendable {
    let text: String
    let start: Float
    let end: Float
    let tokens: [Int]
}

struct StreamingTranscriptionHypothesis: Sendable {
    let text: String
    let words: [StreamingTranscriptionWord]
}

struct StreamingTranscriptionContext: Sendable {
    let clipTimestamp: Float
    let prefixTokens: [Int]
}

protocol StreamingWhisperTranscribing: AnyObject {
    func transcribe(samples: [Float]) async throws -> String
    func transcribePartial(
        samples: [Float],
        context: StreamingTranscriptionContext
    ) async throws -> StreamingTranscriptionHypothesis
}

struct StreamingTranscriptStabilizer {
    private let agreementWordCount = 2
    private var confirmedWords: [StreamingTranscriptionWord] = []
    private var previousWords: [StreamingTranscriptionWord] = []
    private var bridgeWords: [StreamingTranscriptionWord] = []

    private(set) var latestText = ""

    var context: StreamingTranscriptionContext {
        StreamingTranscriptionContext(
            clipTimestamp: bridgeWords.first?.start ?? 0,
            prefixTokens: bridgeWords.flatMap(\.tokens)
        )
    }

    mutating func incorporate(_ hypothesis: StreamingTranscriptionHypothesis) -> String {
        let clipTimestamp = context.clipTimestamp
        let incomingWords = hypothesis.words.filter { $0.start + 0.001 >= clipTimestamp }

        guard !incomingWords.isEmpty else {
            let fallback = hypothesis.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !fallback.isEmpty {
                latestText = fallback
            }
            return latestText
        }

        if previousWords.isEmpty {
            previousWords = incomingWords
        } else {
            let commonPrefix = longestCommonPrefix(previousWords, incomingWords)
            if commonPrefix.count >= agreementWordCount {
                confirmedWords.append(contentsOf: commonPrefix.dropLast(agreementWordCount))
                bridgeWords = Array(commonPrefix.suffix(agreementWordCount))
                let newClipTimestamp = bridgeWords.first?.start ?? clipTimestamp
                previousWords = incomingWords.filter { $0.start + 0.001 >= newClipTimestamp }
            } else {
                previousWords = incomingWords
            }
        }

        let visibleClipTimestamp = context.clipTimestamp
        let visibleTail = incomingWords.filter { $0.start + 0.001 >= visibleClipTimestamp }
        latestText = render(confirmedWords + visibleTail)
        return latestText
    }

    private func longestCommonPrefix(
        _ lhs: [StreamingTranscriptionWord],
        _ rhs: [StreamingTranscriptionWord]
    ) -> [StreamingTranscriptionWord] {
        var prefix: [StreamingTranscriptionWord] = []
        for (left, right) in zip(lhs, rhs) {
            guard normalized(left.text) == normalized(right.text) else { break }
            prefix.append(right)
        }
        return prefix
    }

    private func normalized(_ word: String) -> String {
        let alphanumeric = word.unicodeScalars.filter(CharacterSet.alphanumerics.contains)
        let normalized = String(String.UnicodeScalarView(alphanumeric)).lowercased()
        return normalized.isEmpty
            ? word.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            : normalized
    }

    private func render(_ words: [StreamingTranscriptionWord]) -> String {
        words.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

final class PCMBufferSampleConverter {
    private let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16_000,
        channels: 1,
        interleaved: false
    )!

    private var converter: AVAudioConverter?
    private var inputSignature: AudioFormatSignature?

    func samples(from buffer: AVAudioPCMBuffer) throws -> [Float] {
        if matchesOutputFormat(buffer.format) {
            return readFloatSamples(from: buffer)
        }

        let signature = AudioFormatSignature(format: buffer.format)
        if inputSignature != signature {
            converter = AVAudioConverter(from: buffer.format, to: outputFormat)
            inputSignature = signature
        }

        guard let converter else {
            throw StreamingTranscriptionError.audioConversionFailed
        }

        let ratio = outputFormat.sampleRate / buffer.format.sampleRate
        let expectedFrames = max(1, Int(ceil(Double(buffer.frameLength) * ratio)))
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(expectedFrames)
        ) else {
            throw StreamingTranscriptionError.audioConversionFailed
        }

        var didProvideInput = false
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outputStatus in
            if didProvideInput {
                outputStatus.pointee = .noDataNow
                return nil
            }
            didProvideInput = true
            outputStatus.pointee = .haveData
            return buffer
        }

        if let conversionError {
            throw conversionError
        }
        guard status != .error else {
            throw StreamingTranscriptionError.audioConversionFailed
        }
        return readFloatSamples(from: outputBuffer)
    }

    private func matchesOutputFormat(_ format: AVAudioFormat) -> Bool {
        format.sampleRate == outputFormat.sampleRate
            && format.channelCount == outputFormat.channelCount
            && format.commonFormat == outputFormat.commonFormat
            && format.isInterleaved == outputFormat.isInterleaved
    }

    private func readFloatSamples(from buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData?.pointee else { return [] }
        return Array(UnsafeBufferPointer(start: channelData, count: Int(buffer.frameLength)))
    }
}

private struct AudioFormatSignature: Equatable {
    let sampleRate: Double
    let channelCount: AVAudioChannelCount
    let commonFormat: AVAudioCommonFormat
    let isInterleaved: Bool

    init(format: AVAudioFormat) {
        sampleRate = format.sampleRate
        channelCount = format.channelCount
        commonFormat = format.commonFormat
        isInterleaved = format.isInterleaved
    }
}

final class StreamingWhisperSession: LiveTranscriptionSession {
    var onUpdate: ((String) -> Void)?

    private let transcriber: StreamingWhisperTranscribing
    private let converter = PCMBufferSampleConverter()
    private let lock = NSLock()
    private let minimumPartialSampleCount = 6_400
    private let partialStrideSampleCount = 16_000
    private let liveWindowSampleCount = 240_000

    private var accumulatedSamples: [Float] = []
    private var stabilizer = StreamingTranscriptStabilizer()
    private var rollingTranscript = RollingTranscriptAssembler()
    private var partialTask: Task<Void, Never>?
    private var lastRequestedPartialSampleCount = 0
    private var pendingPartial = false
    private var isFinishing = false
    private var isCancelled = false

    init(transcriber: StreamingWhisperTranscribing) {
        self.transcriber = transcriber
    }

    func append(_ buffer: AVAudioPCMBuffer) throws {
        let newSamples = try converter.samples(from: buffer)
        guard !newSamples.isEmpty else { return }

        var request: PartialRequest?
        lock.lock()
        if !isFinishing, !isCancelled {
            accumulatedSamples.append(contentsOf: newSamples)
            if accumulatedSamples.count >= minimumPartialSampleCount {
                if partialTask == nil {
                    let hasEnoughNewAudio = stabilizer.latestText.isEmpty
                        || accumulatedSamples.count - lastRequestedPartialSampleCount >= partialStrideSampleCount
                    if hasEnoughNewAudio {
                        lastRequestedPartialSampleCount = accumulatedSamples.count
                        request = makePartialRequest()
                    }
                } else if accumulatedSamples.count - lastRequestedPartialSampleCount >= partialStrideSampleCount {
                    pendingPartial = true
                }
            }
        }
        lock.unlock()

        if let request {
            startPartialTranscription(request)
        }
    }

    func finish() async throws -> String {
        let taskToAwait = lock.withCriticalSection { () -> Task<Void, Never>? in
            isFinishing = true
            pendingPartial = false
            return partialTask
        }

        await taskToAwait?.value

        let snapshot = lock.withCriticalSection { accumulatedSamples }

        guard !snapshot.isEmpty else {
            throw StreamingTranscriptionError.noSpeechDetected
        }

        let text = try await transcriber.transcribe(samples: snapshot)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            throw StreamingTranscriptionError.noSpeechDetected
        }

        publish(text)
        return text
    }

    func cancel() {
        lock.lock()
        isCancelled = true
        isFinishing = true
        pendingPartial = false
        let taskToCancel = partialTask
        partialTask = nil
        accumulatedSamples.removeAll(keepingCapacity: false)
        lock.unlock()
        taskToCancel?.cancel()
    }

    private func makePartialRequest() -> PartialRequest {
        let usesRollingWindow = accumulatedSamples.count > liveWindowSampleCount
        let samples = usesRollingWindow
            ? Array(accumulatedSamples.suffix(liveWindowSampleCount))
            : accumulatedSamples
        return PartialRequest(
            samples: samples,
            context: stabilizer.context,
            usesRollingWindow: usesRollingWindow
        )
    }

    private func startPartialTranscription(_ request: PartialRequest) {
        lock.lock()
        guard !isFinishing, !isCancelled else {
            lock.unlock()
            return
        }

        let task = Task { [weak self, transcriber] in
            let result: Result<StreamingTranscriptionHypothesis, Error>
            do {
                result = .success(
                    try await transcriber.transcribePartial(
                        samples: request.samples,
                        context: request.context
                    )
                )
            } catch is CancellationError {
                return
            } catch {
                result = .failure(error)
            }
            self?.handlePartialResult(result, request: request)
        }
        partialTask = task
        lock.unlock()
    }

    private func handlePartialResult(
        _ result: Result<StreamingTranscriptionHypothesis, Error>,
        request: PartialRequest
    ) {
        var update: String?
        var nextRequest: PartialRequest?

        lock.lock()
        partialTask = nil
        if case .success(let hypothesis) = result {
            let hypothesisText = stabilizer.incorporate(hypothesis)
            let text = rollingTranscript.incorporate(
                hypothesisText,
                usesRollingWindow: request.usesRollingWindow
            )
            if !text.isEmpty {
                update = text
            }
        }

        if !isFinishing, !isCancelled {
            let hasEnoughNewAudio = accumulatedSamples.count - lastRequestedPartialSampleCount >= partialStrideSampleCount
            if (pendingPartial || hasEnoughNewAudio), accumulatedSamples.count >= minimumPartialSampleCount {
                pendingPartial = false
                lastRequestedPartialSampleCount = accumulatedSamples.count
                nextRequest = makePartialRequest()
            } else {
                pendingPartial = false
            }
        }
        lock.unlock()

        if let update {
            publish(update)
        }
        if let nextRequest {
            startPartialTranscription(nextRequest)
        }
    }

    private func publish(_ text: String) {
        DispatchQueue.main.async { [weak self] in
            self?.onUpdate?(text)
        }
    }
}

private struct PartialRequest {
    let samples: [Float]
    let context: StreamingTranscriptionContext
    let usesRollingWindow: Bool
}

struct RollingTranscriptAssembler {
    private(set) var text = ""

    mutating func incorporate(_ hypothesis: String, usesRollingWindow: Bool) -> String {
        let incoming = words(in: hypothesis)
        guard !incoming.isEmpty else { return text }

        if !usesRollingWindow || text.isEmpty {
            text = incoming.joined(separator: " ")
            return text
        }

        let existing = words(in: text)
        let overlap = longestOverlap(existing: existing, incoming: incoming)
        guard overlap >= 2 else {
            return text
        }

        text = (existing.dropLast(overlap) + incoming).joined(separator: " ")
        return text
    }

    private func longestOverlap(existing: [String], incoming: [String]) -> Int {
        let maximum = min(existing.count, incoming.count)
        guard maximum > 0 else { return 0 }

        for count in stride(from: maximum, through: 1, by: -1) {
            let existingTail = existing.suffix(count).map(normalized)
            let incomingHead = incoming.prefix(count).map(normalized)
            if existingTail == incomingHead {
                return count
            }
        }
        return 0
    }

    private func words(in value: String) -> [String] {
        value.split(whereSeparator: \.isWhitespace).map(String.init)
    }

    private func normalized(_ word: String) -> String {
        word.unicodeScalars
            .filter(CharacterSet.alphanumerics.contains)
            .map(String.init)
            .joined()
            .lowercased()
    }
}

enum StreamingTranscriptionError: LocalizedError {
    case audioConversionFailed
    case audioCaptureFailed
    case noSpeechDetected
    case recorderBusy

    var errorDescription: String? {
        switch self {
        case .audioConversionFailed:
            return "Unable to convert microphone audio for transcription."
        case .audioCaptureFailed:
            return "Unable to save the recording."
        case .noSpeechDetected:
            return "No speech was detected."
        case .recorderBusy:
            return "The previous recording is still being finalized."
        }
    }
}

private extension NSLock {
    func withCriticalSection<T>(_ body: () -> T) -> T {
        lock()
        defer { unlock() }
        return body()
    }
}

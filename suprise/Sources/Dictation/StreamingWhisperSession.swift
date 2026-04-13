import AVFoundation
import Foundation

protocol StreamingWhisperTranscriber {
    func transcribe(samples: [Float], locale: Locale) async throws -> String
}

final class StreamingWhisperSession: DictationSession {
    var onUpdate: ((DictationUpdate) -> Void)?

    private let transcriber: StreamingWhisperTranscriber
    private let locale: Locale
    private let converter = PCMBufferSampleConverter()
    private let lock = NSLock()
    private let minimumPartialSampleCount = 6_400
    private let partialStrideSampleCount = 8_000

    private var accumulatedSamples: [Float] = []
    private var latestText = ""
    private var partialTask: Task<Void, Never>?
    private var lastRequestedPartialSampleCount = 0
    private var pendingPartial = false
    private var isFinishing = false
    private var isCancelled = false

    init(transcriber: StreamingWhisperTranscriber, locale: Locale) {
        self.transcriber = transcriber
        self.locale = locale
    }

    func append(_ buffer: AVAudioPCMBuffer, at _: AVAudioTime?) throws {
        let newSamples = try converter.samples(from: buffer)
        guard !newSamples.isEmpty else {
            return
        }

        var snapshotToTranscribe: [Float]?
        lock.withLock {
            guard !isFinishing, !isCancelled else {
                return
            }

            accumulatedSamples.append(contentsOf: newSamples)

            guard accumulatedSamples.count >= minimumPartialSampleCount else {
                return
            }

            guard partialTask == nil else {
                pendingPartial = true
                return
            }

            let hasNewAudio = accumulatedSamples.count - lastRequestedPartialSampleCount >= partialStrideSampleCount
            guard latestText.isEmpty || hasNewAudio else {
                return
            }

            lastRequestedPartialSampleCount = accumulatedSamples.count
            snapshotToTranscribe = accumulatedSamples
        }

        if let snapshotToTranscribe {
            startPartialTranscription(with: snapshotToTranscribe)
        }
    }

    func finish() async throws -> String {
        let taskToAwait = lock.withLock { () -> Task<Void, Never>? in
            isFinishing = true
            pendingPartial = false
            return partialTask
        }

        await taskToAwait?.value

        let snapshot = lock.withLock { accumulatedSamples }
        guard !snapshot.isEmpty else {
            throw DictationError.noSpeechDetected
        }

        let text = try await transcriber.transcribe(samples: snapshot, locale: locale)
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw DictationError.noSpeechDetected
        }

        lock.withLock {
            latestText = trimmed
        }

        await MainActor.run { [weak self] in
            self?.onUpdate?(DictationUpdate(text: trimmed, isFinal: true))
        }

        return trimmed
    }

    func cancel() {
        let taskToCancel = lock.withLock { () -> Task<Void, Never>? in
            isCancelled = true
            isFinishing = true
            pendingPartial = false
            let partialTask = self.partialTask
            self.partialTask = nil
            accumulatedSamples.removeAll(keepingCapacity: false)
            latestText = ""
            return partialTask
        }

        taskToCancel?.cancel()
    }

    private func startPartialTranscription(with snapshot: [Float]) {
        let task = Task { [weak self, transcriber, locale] in
            let result: Result<String, Error>
            do {
                let text = try await transcriber.transcribe(samples: snapshot, locale: locale)
                result = .success(text)
            } catch is CancellationError {
                return
            } catch {
                result = .failure(error)
            }

            self?.handlePartialResult(result, snapshotSampleCount: snapshot.count)
        }

        lock.withLock {
            partialTask = task
        }
    }

    private func handlePartialResult(_ result: Result<String, Error>, snapshotSampleCount: Int) {
        var update: DictationUpdate?
        var nextSnapshot: [Float]?

        lock.withLock {
            partialTask = nil

            if case .success(let text) = result {
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    latestText = trimmed
                    update = DictationUpdate(text: trimmed, isFinal: false)
                }
            }

            guard !isFinishing, !isCancelled else {
                pendingPartial = false
                return
            }

            let hasPendingAudio = pendingPartial || accumulatedSamples.count - snapshotSampleCount >= partialStrideSampleCount
            pendingPartial = false

            guard hasPendingAudio, accumulatedSamples.count >= minimumPartialSampleCount else {
                return
            }

            lastRequestedPartialSampleCount = accumulatedSamples.count
            nextSnapshot = accumulatedSamples
        }

        if let update {
            DispatchQueue.main.async { [weak self] in
                self?.onUpdate?(update)
            }
        }

        if let nextSnapshot {
            startPartialTranscription(with: nextSnapshot)
        }
    }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock()
        defer { unlock() }
        return body()
    }
}

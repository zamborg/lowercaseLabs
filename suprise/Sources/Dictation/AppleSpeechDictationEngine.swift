import AVFoundation
import Foundation
import Speech

final class AppleSpeechDictationEngine: DictationEngine {
    let kind: DictationEngineKind = .apple

    func prepare() async throws {
        let status = await requestAuthorizationIfNeeded()
        guard status == .authorized else {
            throw DictationError.speechPermissionDenied
        }
    }

    func makeSession(locale: Locale) throws -> DictationSession {
        if let recognizer = SFSpeechRecognizer(locale: locale), recognizer.isAvailable {
            return try AppleSpeechDictationSession(recognizer: recognizer)
        }

        if let fallbackRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en_US")), fallbackRecognizer.isAvailable {
            return try AppleSpeechDictationSession(recognizer: fallbackRecognizer)
        }

        throw DictationError.recognizerUnavailable
    }

    func unload() async {}

    private func requestAuthorizationIfNeeded() async -> SFSpeechRecognizerAuthorizationStatus {
        let current = SFSpeechRecognizer.authorizationStatus()
        if current != .notDetermined {
            return current
        }

        return await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
    }
}

final class AppleSpeechDictationSession: DictationSession {
    var onUpdate: ((DictationUpdate) -> Void)?

    private let request: SFSpeechAudioBufferRecognitionRequest
    private let lock = NSLock()
    private var recognitionTask: SFSpeechRecognitionTask?
    private var latestText = ""
    private var finishContinuation: CheckedContinuation<String, Error>?
    private var resolved = false

    init(recognizer: SFSpeechRecognizer) throws {
        request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = false
        request.taskHint = .dictation
        if #available(iOS 16.0, *) {
            request.addsPunctuation = true
        }

        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            self?.handle(result: result, error: error)
        }
    }

    func append(_ buffer: AVAudioPCMBuffer, at _: AVAudioTime?) throws {
        request.append(buffer)
    }

    func finish() async throws -> String {
        request.endAudio()

        return try await withCheckedThrowingContinuation { continuation in
            let immediateResult = lock.withLock { () -> Result<String, Error>? in
                if resolved {
                    let trimmed = latestText.trimmingCharacters(in: .whitespacesAndNewlines)
                    return trimmed.isEmpty ? .failure(DictationError.noSpeechDetected) : .success(trimmed)
                }
                finishContinuation = continuation
                return nil
            }

            if let immediateResult {
                continuation.resume(with: immediateResult)
                return
            }

            Task { [weak self] in
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                self?.resolveWithLatestTextIfNeeded()
            }
        }
    }

    func cancel() {
        request.endAudio()
        recognitionTask?.cancel()
        let resolution = lock.withLock {
            takeResolutionLocked(
                text: latestText.trimmingCharacters(in: .whitespacesAndNewlines),
                error: latestText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? DictationError.noSpeechDetected : nil
            )
        }
        resumeIfNeeded(resolution)
    }

    private func handle(result: SFSpeechRecognitionResult?, error: Error?) {
        var update: DictationUpdate?
        var resolution: ContinuationResolution?

        if let result {
            let text = result.bestTranscription.formattedString.trimmingCharacters(in: .whitespacesAndNewlines)
            lock.withLock {
                latestText = text
                update = DictationUpdate(text: text, isFinal: result.isFinal)
                if result.isFinal {
                    resolution = takeResolutionLocked(text: text, error: nil)
                }
            }
        }

        if let error, resolution == nil {
            lock.withLock {
                let trimmed = latestText.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty {
                    resolution = takeResolutionLocked(text: nil, error: error)
                } else {
                    resolution = takeResolutionLocked(text: trimmed, error: nil)
                }
            }
        }

        if let update {
            onUpdate?(update)
        }
        resumeIfNeeded(resolution)
    }

    private func resolveWithLatestTextIfNeeded() {
        let resolution = lock.withLock {
            let trimmed = latestText.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                return takeResolutionLocked(text: nil, error: DictationError.noSpeechDetected)
            } else {
                return takeResolutionLocked(text: trimmed, error: nil)
            }
        }
        resumeIfNeeded(resolution)
    }

    private func takeResolutionLocked(text: String?, error: Error?) -> ContinuationResolution? {
        guard let result = makeResult(text: text, error: error) else {
            return nil
        }

        let continuationToResume: CheckedContinuation<String, Error>? = {
            guard !resolved else {
                return nil
            }
            resolved = true
            let continuation = finishContinuation
            finishContinuation = nil
            return continuation
        }()

        guard let continuationToResume else {
            return nil
        }

        return (continuationToResume, result)
    }

    private func resumeIfNeeded(_ resolution: ContinuationResolution?) {
        guard let (continuation, result) = resolution else {
            return
        }
        continuation.resume(with: result)
    }

    private func makeResult(text: String?, error: Error?) -> Result<String, Error>? {
        if let text {
            return .success(text)
        }
        if let error {
            return .failure(error)
        }
        return nil
    }
}

private typealias ContinuationResolution = (CheckedContinuation<String, Error>, Result<String, Error>)

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock()
        defer { unlock() }
        return body()
    }
}

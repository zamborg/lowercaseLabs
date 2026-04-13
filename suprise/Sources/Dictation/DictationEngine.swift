import AVFoundation
import Foundation

struct DictationUpdate {
    let text: String
    let isFinal: Bool
}

protocol DictationSession: AnyObject {
    var onUpdate: ((DictationUpdate) -> Void)? { get set }
    func append(_ buffer: AVAudioPCMBuffer, at time: AVAudioTime?) throws
    func finish() async throws -> String
    func cancel()
}

protocol DictationEngine: AnyObject {
    var kind: DictationEngineKind { get }
    func prepare() async throws
    func makeSession(locale: Locale) throws -> DictationSession
    func unload() async
}

enum DictationEngineKind: String, CaseIterable, Identifiable {
    case apple
    case whisperKit
    case whisperCpp

    var id: String { rawValue }

    var title: String {
        switch self {
        case .apple:
            return "Apple"
        case .whisperKit:
            return "WhisperKit"
        case .whisperCpp:
            return "whisper.cpp"
        }
    }

    var caption: String {
        switch self {
        case .apple:
            return "Apple Speech. Lowest setup cost and the fastest way to sanity-check live dictation."
        case .whisperKit:
            return "Core ML Whisper via WhisperKit. First run downloads a tiny English model."
        case .whisperCpp:
            return "GGML Whisper via whisper.cpp. First run downloads ggml-tiny.en."
        }
    }
}

enum DictationError: LocalizedError {
    case microphonePermissionDenied
    case speechPermissionDenied
    case recognizerUnavailable
    case engineUnavailable(String)
    case noSpeechDetected
    case audioCaptureFailed(String)

    var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            return "Microphone permission is required to start dictation."
        case .speechPermissionDenied:
            return "Speech recognition permission is required to transcribe dictation."
        case .recognizerUnavailable:
            return "Speech recognition is unavailable on this device right now."
        case .engineUnavailable(let message):
            return message
        case .noSpeechDetected:
            return "No speech was detected."
        case .audioCaptureFailed(let message):
            return message
        }
    }
}

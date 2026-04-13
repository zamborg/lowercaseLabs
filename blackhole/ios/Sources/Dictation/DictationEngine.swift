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

    var id: String { rawValue }

    var title: String {
        switch self {
        case .apple: return "Apple"
        case .whisperKit: return "WhisperKit"
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
        case .microphonePermissionDenied: return "Microphone permission is required."
        case .speechPermissionDenied: return "Speech recognition permission is required."
        case .recognizerUnavailable: return "Speech recognition is unavailable on this device."
        case .engineUnavailable(let m): return m
        case .noSpeechDetected: return "No speech detected."
        case .audioCaptureFailed(let m): return m
        }
    }
}

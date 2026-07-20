import AVFoundation
import Foundation
import Speech

enum TranscriptionEngineKind: String, CaseIterable, Identifiable, Sendable {
    case apple
    case whisperKit

    var id: String { rawValue }

    var title: String {
        switch self {
        case .apple:
            return "Apple On-Device"
        case .whisperKit:
            return "WhisperKit"
        }
    }

    var compactTitle: String {
        switch self {
        case .apple:
            return "Apple"
        case .whisperKit:
            return "WhisperKit"
        }
    }

    var detail: String {
        switch self {
        case .apple:
            return "Lowest memory, live system transcription"
        case .whisperKit:
            return "Private local model with broader device support"
        }
    }

    var systemImage: String {
        switch self {
        case .apple:
            return "apple.logo"
        case .whisperKit:
            return "waveform.badge.magnifyingglass"
        }
    }

    var providerIdentifier: String {
        switch self {
        case .apple:
            return "apple_speech"
        case .whisperKit:
            return "whisperkit"
        }
    }

    var isAvailable: Bool {
        switch self {
        case .apple:
            if #available(iOS 26.0, *) {
                return SpeechTranscriber.isAvailable
            }
            return false
        case .whisperKit:
            return true
        }
    }

    static var availableCases: [TranscriptionEngineKind] {
        allCases.filter(\.isAvailable)
    }

    static var defaultChoice: TranscriptionEngineKind {
        if TranscriptionEngineKind.apple.isAvailable {
            return .apple
        }
        return .whisperKit
    }
}

enum TranscriptionLanguage: String, CaseIterable, Identifiable, Sendable {
    case device
    case englishUS = "en_US"
    case englishUK = "en_GB"
    case spanish = "es_ES"
    case french = "fr_FR"
    case german = "de_DE"
    case italian = "it_IT"
    case portuguese = "pt_BR"
    case japanese = "ja_JP"
    case korean = "ko_KR"
    case mandarin = "zh_CN"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .device:
            let languageName = resolvedLocale.localizedString(
                forLanguageCode: languageCode
            ) ?? languageCode.uppercased()
            return "Device Language (\(languageName))"
        case .englishUS:
            return "English (US)"
        case .englishUK:
            return "English (UK)"
        case .spanish:
            return "Spanish"
        case .french:
            return "French"
        case .german:
            return "German"
        case .italian:
            return "Italian"
        case .portuguese:
            return "Portuguese (Brazil)"
        case .japanese:
            return "Japanese"
        case .korean:
            return "Korean"
        case .mandarin:
            return "Mandarin Chinese"
        }
    }

    var resolvedLocale: Locale {
        switch self {
        case .device:
            return .autoupdatingCurrent
        default:
            return Locale(identifier: rawValue)
        }
    }

    var languageCode: String {
        resolvedLocale.language.languageCode?.identifier
            ?? resolvedLocale.identifier.split(separator: "_").first.map(String.init)
            ?? "en"
    }

    var whisperModelName: String {
        languageCode == "en" ? "tiny.en" : "tiny"
    }
}

struct TranscriptionConfiguration: Equatable, Sendable {
    let engine: TranscriptionEngineKind
    let language: TranscriptionLanguage

    var acceptedTranscriptStrategy: String {
        "\(engine.providerIdentifier)_\(language.languageCode)_accepted"
    }
}

protocol LiveTranscriptionSession: AnyObject {
    var onUpdate: ((String) -> Void)? { get set }
    func append(_ buffer: AVAudioPCMBuffer) throws
    func finish() async throws -> String
    func cancel()
}

protocol TranscriptionEngine: AnyObject {
    var kind: TranscriptionEngineKind { get }
    func prepare(language: TranscriptionLanguage) async throws
    func makeSession(inputFormat: AVAudioFormat) throws -> any LiveTranscriptionSession
    func transcribe(audioURL: URL) async throws -> String
    func unload() async
}

@MainActor
final class TranscriptionEngineCoordinator {
    private var loadedConfiguration: TranscriptionConfiguration?
    private var loadedSlot: TranscriptionEngineSlot?
    private var retiredSlots: [TranscriptionEngineSlot] = []
    private var preparationID = UUID()

    var isReady: Bool { loadedSlot != nil }

    func prepare(configuration: TranscriptionConfiguration) async throws {
        guard loadedConfiguration != configuration || loadedSlot == nil else {
            return
        }

        let operationID = UUID()
        preparationID = operationID
        await unloadLoadedEngine()
        let engine: any TranscriptionEngine
        switch configuration.engine {
        case .apple:
            guard #available(iOS 26.0, *) else {
                throw TranscriptionEngineError.engineUnavailable
            }
            engine = AppleSpeechTranscriptionEngine()
        case .whisperKit:
            engine = WhisperKitTranscriptionEngine()
        }

        do {
            try await engine.prepare(language: configuration.language)
            try Task.checkCancellation()
            guard preparationID == operationID else {
                throw CancellationError()
            }
            loadedSlot = TranscriptionEngineSlot(engine: engine)
            loadedConfiguration = configuration
        } catch {
            await engine.unload()
            throw error
        }
    }

    func makeSession(inputFormat: AVAudioFormat) throws -> any LiveTranscriptionSession {
        guard let loadedSlot else {
            throw TranscriptionEngineError.engineNotReady
        }
        let session = try loadedSlot.engine.makeSession(inputFormat: inputFormat)
        loadedSlot.activeUseCount += 1
        return ManagedLiveTranscriptionSession(session: session) { [weak self, loadedSlot] in
            Task { @MainActor in
                await self?.release(loadedSlot)
            }
        }
    }

    func transcribe(audioURL: URL) async throws -> String {
        guard let loadedSlot else {
            throw TranscriptionEngineError.engineNotReady
        }
        loadedSlot.activeUseCount += 1
        do {
            let text = try await loadedSlot.engine.transcribe(audioURL: audioURL)
            await release(loadedSlot)
            return text
        } catch {
            await release(loadedSlot)
            throw error
        }
    }

    func unload() async {
        preparationID = UUID()
        await unloadLoadedEngine()
    }

    private func unloadLoadedEngine() async {
        if let loadedSlot {
            loadedSlot.isRetired = true
            if loadedSlot.activeUseCount == 0 {
                await loadedSlot.engine.unload()
            } else {
                retiredSlots.append(loadedSlot)
            }
        }
        loadedSlot = nil
        loadedConfiguration = nil
    }

    private func release(_ slot: TranscriptionEngineSlot) async {
        slot.activeUseCount = max(0, slot.activeUseCount - 1)
        guard slot.isRetired, slot.activeUseCount == 0 else { return }
        await slot.engine.unload()
        retiredSlots.removeAll { $0 === slot }
    }
}

@MainActor
private final class TranscriptionEngineSlot {
    let engine: any TranscriptionEngine
    var activeUseCount = 0
    var isRetired = false

    init(engine: any TranscriptionEngine) {
        self.engine = engine
    }
}

private final class ManagedLiveTranscriptionSession: LiveTranscriptionSession {
    private let session: any LiveTranscriptionSession
    private let releaseLock = NSLock()
    private var onRelease: (() -> Void)?

    var onUpdate: ((String) -> Void)? {
        get { session.onUpdate }
        set { session.onUpdate = newValue }
    }

    init(
        session: any LiveTranscriptionSession,
        onRelease: @escaping () -> Void
    ) {
        self.session = session
        self.onRelease = onRelease
    }

    deinit {
        session.cancel()
        releaseOnce()
    }

    func append(_ buffer: AVAudioPCMBuffer) throws {
        try session.append(buffer)
    }

    func finish() async throws -> String {
        do {
            let text = try await session.finish()
            releaseOnce()
            return text
        } catch {
            releaseOnce()
            throw error
        }
    }

    func cancel() {
        session.cancel()
        releaseOnce()
    }

    private func releaseOnce() {
        let release = releaseLock.withLock { () -> (() -> Void)? in
            defer { onRelease = nil }
            return onRelease
        }
        release?()
    }
}

enum TranscriptionEngineError: LocalizedError {
    case engineUnavailable
    case engineNotReady
    case languageUnavailable(String)
    case noSpeechDetected
    case audioBufferCopyFailed
    case audioFormatUnavailable

    var errorDescription: String? {
        switch self {
        case .engineUnavailable:
            return "This transcription engine is unavailable on this device."
        case .engineNotReady:
            return "The transcription engine is still getting ready."
        case .languageUnavailable(let language):
            return "\(language) is unavailable for this transcription engine."
        case .noSpeechDetected:
            return "No speech was detected."
        case .audioBufferCopyFailed:
            return "Unable to process microphone audio for transcription."
        case .audioFormatUnavailable:
            return "No compatible microphone format is available for transcription."
        }
    }
}

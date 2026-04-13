import Foundation
import SwiftUI

@MainActor
final class IngestViewModel: ObservableObject {
    enum Status {
        case idle, starting, listening, finalizing, uploading

        var title: String {
            switch self {
            case .idle: return "Ready"
            case .starting: return "Starting"
            case .listening: return "Listening"
            case .finalizing: return "Finalizing"
            case .uploading: return "Saving"
            }
        }

        var symbolName: String {
            switch self {
            case .idle: return "mic.fill"
            case .starting: return "mic.badge.plus"
            case .listening: return "waveform"
            case .finalizing: return "waveform.badge.mic"
            case .uploading: return "arrow.up.circle"
            }
        }

        var detail: String {
            switch self {
            case .idle: return "Hold the button and speak into the void."
            case .starting: return "Preparing microphone…"
            case .listening: return "Speak naturally. Text appears while you talk."
            case .finalizing: return "Finishing transcription…"
            case .uploading: return "Analyzing and saving…"
            }
        }

        var holdInstruction: String {
            switch self {
            case .idle: return "Press and hold to dictate."
            case .starting: return "Keep holding…"
            case .listening: return "Release to stop."
            case .finalizing, .uploading: return "Processing…"
            }
        }
    }

    @Published private(set) var status: Status = .idle
    @Published var transcriptText = ""
    @Published var selectedEngineKind: DictationEngineKind = .apple
    @Published var alertMessage: String?
    @Published var lastPostedItem: Item?
    @Published private(set) var isSwitchingEngine = false

    var isBusy: Bool { status == .starting || status == .finalizing || status == .uploading || isSwitchingEngine }
    var isListening: Bool { status == .listening }
    var areControlsLocked: Bool { isBusy || isListening }
    var canSubmit: Bool { status == .idle && !transcriptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    private let audioCapture = AudioCapturePipeline()
    private let engineCoordinator = DictationEngineCoordinator()
    private var currentSession: DictationSession?
    private var startTask: Task<Void, Never>?
    private var engineSwitchTask: Task<Void, Never>?
    private var isHoldActive = false
    private var pendingStopAfterStart = false

    func selectEngine(_ kind: DictationEngineKind) {
        guard selectedEngineKind != kind else { return }
        selectedEngineKind = kind
        guard status == .idle else { return }
        engineSwitchTask?.cancel()
        isSwitchingEngine = true
        engineSwitchTask = Task { [weak self] in await self?.releaseLoadedEngine() }
    }

    func pressDidStart() {
        guard status == .idle else { return }
        isHoldActive = true
        pendingStopAfterStart = false
        startTask?.cancel()
        startTask = Task { [weak self] in await self?.startDictation() }
    }

    func pressDidEnd() {
        isHoldActive = false
        switch status {
        case .starting: pendingStopAfterStart = true
        case .listening: Task { [weak self] in await self?.stopDictation() }
        default: break
        }
    }

    func submitTranscript() {
        let text = transcriptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        Task { [weak self] in await self?.uploadItem(text: text) }
    }

    func clearTranscript() {
        transcriptText = ""
    }

    private func startDictation() async {
        status = .starting
        transcriptText = ""

        if let t = engineSwitchTask { await t.value }

        let hasMic = await audioCapture.requestPermission()
        guard hasMic else {
            alertMessage = DictationError.microphonePermissionDenied.localizedDescription
            resetState()
            return
        }

        do {
            let engine = try await engineCoordinator.engine(for: selectedEngineKind)
            let session = try engine.makeSession(locale: .current)
            session.onUpdate = { [weak self] update in
                Task { @MainActor [weak self] in self?.transcriptText = update.text }
            }
            try audioCapture.start { [weak session] buffer, when in
                try? session?.append(buffer, at: when)
            }
            currentSession = session
            status = .listening
            if pendingStopAfterStart || !isHoldActive { await stopDictation() }
        } catch {
            alertMessage = error.localizedDescription
            currentSession?.cancel()
            currentSession = nil
            audioCapture.stop()
            resetState()
        }
    }

    private func stopDictation() async {
        guard status == .listening, let session = currentSession else { return }
        status = .finalizing
        audioCapture.stop()
        currentSession = nil

        do {
            let finalText = try await session.finish().trimmingCharacters(in: .whitespacesAndNewlines)
            guard !finalText.isEmpty else {
                alertMessage = DictationError.noSpeechDetected.localizedDescription
                resetState()
                return
            }
            transcriptText = finalText
            resetState(keepTranscript: true)
        } catch {
            alertMessage = error.localizedDescription
            resetState()
        }
    }

    private func uploadItem(text: String) async {
        status = .uploading
        do {
            let item = try await APIClient.shared.createItem(content: text)
            lastPostedItem = item
            transcriptText = ""
            resetState()
        } catch {
            alertMessage = error.localizedDescription
            resetState(keepTranscript: true)
        }
    }

    private func resetState(keepTranscript: Bool = false) {
        status = .idle
        isHoldActive = false
        pendingStopAfterStart = false
        startTask = nil
        if !keepTranscript { transcriptText = "" }
    }

    private func releaseLoadedEngine() async {
        await engineCoordinator.unloadCurrentEngine()
        guard !Task.isCancelled else { return }
        isSwitchingEngine = false
        engineSwitchTask = nil
    }
}

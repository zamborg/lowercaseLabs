import Foundation
import SwiftUI

@MainActor
final class DictationViewModel: ObservableObject {
    enum Status {
        case idle
        case starting
        case listening
        case finalizing
    }

    @Published private(set) var selectedEngineKind: DictationEngineKind = .whisperKit
    @Published var transcriptText = ""
    @Published var savedRecords: [DictationRecord] = []
    @Published var alertMessage: String?
    @Published private(set) var status: Status = .idle
    @Published private(set) var isSwitchingEngine = false

    var isBusy: Bool {
        status == .starting || status == .finalizing || isSwitchingEngine
    }

    var isListening: Bool {
        status == .listening
    }

    var areControlsLocked: Bool {
        isBusy || isListening
    }

    var statusTitle: String {
        if isSwitchingEngine {
            return "Switching Engine"
        }

        switch status {
        case .idle:
            return "Ready"
        case .starting:
            return "Starting"
        case .listening:
            return "Listening"
        case .finalizing:
            return "Saving"
        }
    }

    var statusDetail: String {
        if isSwitchingEngine {
            return "Unloading the previous model before the next dictation run."
        }

        switch status {
        case .idle:
            return "Hold the button to stream speech into text, then release to save it."
        case .starting:
            return "Preparing the microphone and loading the selected model. First run may download it."
        case .listening:
            return "Speak naturally. Text should update while you talk."
        case .finalizing:
            return "Finishing transcription and writing the text file."
        }
    }

    var holdInstruction: String {
        if isSwitchingEngine {
            return "Hold on while the previous engine unloads."
        }

        switch status {
        case .idle:
            return "Press and hold to dictate."
        case .starting:
            return "Keep holding while dictation starts."
        case .listening:
            return "Release to stop and save."
        case .finalizing:
            return "Finalizing transcript."
        }
    }

    var statusSymbolName: String {
        if isSwitchingEngine {
            return "arrow.triangle.2.circlepath"
        }

        switch status {
        case .idle:
            return "sparkle.magnifyingglass"
        case .starting:
            return "mic.badge.plus"
        case .listening:
            return "waveform"
        case .finalizing:
            return "square.and.arrow.down"
        }
    }

    private let store = DictationStore()
    private let audioCapture = AudioCapturePipeline()
    private let engineCoordinator = DictationEngineCoordinator()
    private var currentSession: DictationSession?
    private var startTask: Task<Void, Never>?
    private var engineSwitchTask: Task<Void, Never>?
    private var isHoldActive = false
    private var pendingStopAfterStart = false

    func reloadSavedDictations() {
        savedRecords = store.listRecords()
    }

    func delete(_ record: DictationRecord) {
        do {
            try store.delete(record)
            savedRecords = store.listRecords()
        } catch {
            alertMessage = error.localizedDescription
        }
    }

    func selectEngine(_ kind: DictationEngineKind) {
        guard selectedEngineKind != kind else {
            return
        }

        selectedEngineKind = kind

        guard status == .idle else {
            return
        }

        engineSwitchTask?.cancel()
        isSwitchingEngine = true
        engineSwitchTask = Task { [weak self] in
            await self?.releaseLoadedEngine()
        }
    }

    func pressDidStart() {
        guard status == .idle else {
            return
        }

        isHoldActive = true
        pendingStopAfterStart = false

        startTask?.cancel()
        startTask = Task { [weak self] in
            await self?.startDictation()
        }
    }

    func pressDidEnd() {
        isHoldActive = false

        switch status {
        case .starting:
            pendingStopAfterStart = true
        case .listening:
            Task { [weak self] in
                await self?.stopDictation()
            }
        case .idle, .finalizing:
            break
        }
    }

    private func startDictation() async {
        status = .starting
        transcriptText = ""

        if let engineSwitchTask {
            await engineSwitchTask.value
        }

        let hasMicAccess = await audioCapture.requestPermission()
        guard hasMicAccess else {
            alertMessage = DictationError.microphonePermissionDenied.localizedDescription
            resetState()
            return
        }

        do {
            let engine = try await engineCoordinator.engine(for: selectedEngineKind)
            let session = try engine.makeSession(locale: Locale.current)
            session.onUpdate = { [weak self] update in
                Task { @MainActor [weak self] in
                    self?.transcriptText = update.text
                }
            }

            try audioCapture.start { [weak session] buffer, when in
                try? session?.append(buffer, at: when)
            }

            currentSession = session
            status = .listening

            if pendingStopAfterStart || !isHoldActive {
                await stopDictation()
            }
        } catch {
            alertMessage = error.localizedDescription
            currentSession?.cancel()
            currentSession = nil
            audioCapture.stop()
            resetState()
        }
    }

    private func stopDictation() async {
        guard status == .listening, let session = currentSession else {
            return
        }

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
            _ = try store.save(text: finalText)
            savedRecords = store.listRecords()
            resetState(keepTranscript: true)
        } catch {
            alertMessage = error.localizedDescription
            resetState()
        }
    }

    private func resetState(keepTranscript: Bool = false) {
        status = .idle
        isHoldActive = false
        pendingStopAfterStart = false
        startTask = nil
        if !keepTranscript {
            transcriptText = ""
        }
    }

    private func releaseLoadedEngine() async {
        await engineCoordinator.unloadCurrentEngine()
        guard !Task.isCancelled else {
            return
        }
        isSwitchingEngine = false
        engineSwitchTask = nil
    }
}

import Foundation
import Observation

@MainActor
@Observable
final class CaptureViewModel {
    enum SubmissionState: String {
        case idle
        case recording
        case uploading
        case transcribing
        case result
        case failed
    }

    var submissionState: SubmissionState = .idle
    var selectedMode: CaptureMode = .auto
    var transcriptPreview = ""
    var routeSummary = "Awaiting capture"
    var statusMessage = "One button. Capture anything."
    var errorMessage: String?
    var pendingConfirmation: TranscriptionFinalResult?
    var recordingStartedAt: Date?
    var elapsedSeconds: TimeInterval = 0

    private let audioCaptureService: AudioCaptureService
    private let transcriptionClient: TranscriptionSessionClient
    private let routerClient: CaptureRouterClient
    private let notesRepository: NotesRepository
    private let todosRepository: TodosRepository
    private let navigator: AppNavigator
    private let onNoteCreated: @MainActor () async -> Void
    private let onTodoCreated: @MainActor () async -> Void
    private var timerTask: Task<Void, Never>?

    init(
        audioCaptureService: AudioCaptureService,
        transcriptionClient: TranscriptionSessionClient,
        routerClient: CaptureRouterClient,
        notesRepository: NotesRepository,
        todosRepository: TodosRepository,
        navigator: AppNavigator,
        onNoteCreated: @escaping @MainActor () async -> Void,
        onTodoCreated: @escaping @MainActor () async -> Void
    ) {
        self.audioCaptureService = audioCaptureService
        self.transcriptionClient = transcriptionClient
        self.routerClient = routerClient
        self.notesRepository = notesRepository
        self.todosRepository = todosRepository
        self.navigator = navigator
        self.onNoteCreated = onNoteCreated
        self.onTodoCreated = onTodoCreated
    }

    var isRecording: Bool {
        submissionState == .recording
    }

    func syncPendingMode() {
        guard let mode = navigator.consumePendingCaptureMode() else {
            return
        }
        selectedMode = mode
        statusMessage = mode == .todo ? "Voice add locked to todo mode." : "Voice search ready."
    }

    func toggleRecording() async {
        errorMessage = nil

        if audioCaptureService.isRecording {
            await stopRecordingAndProcess()
            return
        }

        let permitted = await audioCaptureService.requestPermission()
        guard permitted else {
            errorMessage = "Microphone permission is required."
            submissionState = .failed
            return
        }

        do {
            try audioCaptureService.startRecording()
            recordingStartedAt = Date()
            elapsedSeconds = 0
            submissionState = .recording
            statusMessage = "Listening for your capture..."
            startTimer()
        } catch {
            errorMessage = error.localizedDescription
            submissionState = .failed
        }
    }

    func confirmCreateNote() async {
        guard let pendingConfirmation else { return }
        do {
            let payload = try pendingConfirmation.routePayload.decodePayload(NoteDraftPayload.self)
            _ = try await notesRepository.createNote(
                NoteCreateRequest(
                    title: payload.title,
                    bodyMarkdown: payload.bodyMarkdown,
                    tags: payload.tags,
                    sourceCaptureId: pendingConfirmation.captureId
                )
            )
            self.pendingConfirmation = nil
            submissionState = .result
            routeSummary = "Note saved"
            statusMessage = "Capture routed to Notes."
            await onNoteCreated()
        } catch {
            errorMessage = error.localizedDescription
            submissionState = .failed
        }
    }

    func confirmCreateTodo() async {
        guard let pendingConfirmation else { return }
        do {
            let payload = try pendingConfirmation.routePayload.decodePayload(TodoDraftPayload.self)
            _ = try await todosRepository.createTodo(
                TodoCreateRequest(
                    title: payload.title,
                    descriptionMarkdown: payload.descriptionMarkdown,
                    priority: payload.priority,
                    deadlineAt: payload.deadlineAt,
                    tags: payload.tags,
                    sourceCaptureId: pendingConfirmation.captureId
                )
            )
            self.pendingConfirmation = nil
            submissionState = .result
            routeSummary = "Todo saved"
            statusMessage = "Capture routed to Todos."
            await onTodoCreated()
        } catch {
            errorMessage = error.localizedDescription
            submissionState = .failed
        }
    }

    func confirmSearch() {
        guard let pendingConfirmation else { return }
        let payload = (try? pendingConfirmation.routePayload.decodePayload(SearchDraftPayload.self)) ?? SearchDraftPayload(queryText: pendingConfirmation.transcript)
        navigator.openSearchResults(query: payload.queryText)
        self.pendingConfirmation = nil
        submissionState = .result
        routeSummary = "Search results opened"
        statusMessage = payload.queryText
    }

    private func stopRecordingAndProcess() async {
        do {
            let audioURL = try audioCaptureService.stopRecording()
            stopTimer()
            await processRecording(at: audioURL)
            try? FileManager.default.removeItem(at: audioURL)
        } catch {
            errorMessage = error.localizedDescription
            submissionState = .failed
        }
    }

    private func processRecording(at audioURL: URL) async {
        pendingConfirmation = nil
        transcriptPreview = ""
        routeSummary = "Processing capture"
        errorMessage = nil
        submissionState = .uploading
        statusMessage = "Uploading capture..."
        do {
            let stream = try await transcriptionClient.start(audioURL: audioURL, mode: selectedMode)
            for try await event in stream {
                switch event {
                case .sessionCreated(let sessionID):
                    routeSummary = "Session \(sessionID.prefix(8))"
                case .uploading:
                    submissionState = .uploading
                    statusMessage = "Uploading capture..."
                case .transcribing:
                    submissionState = .transcribing
                    statusMessage = "Transcribing and routing..."
                case .partialTranscript(let text):
                    transcriptPreview = text
                case .finalResult(let result):
                    transcriptPreview = result.transcript
                    await handleFinalResult(result)
                case .failed(let message):
                    errorMessage = message
                    statusMessage = "Capture failed."
                    submissionState = .failed
                }
            }
        } catch {
            errorMessage = error.localizedDescription
            statusMessage = "Capture failed."
            submissionState = .failed
        }
    }

    private func handleFinalResult(_ result: TranscriptionFinalResult) async {
        routeSummary = "\(result.route.rawValue.capitalized) • \(Int(result.routeConfidence * 100))%"

        switch routerClient.action(for: result) {
        case .autoCreateNote(let request):
            do {
                _ = try await notesRepository.createNote(request)
                submissionState = .result
                routeSummary = "Note saved"
                statusMessage = "Note saved automatically."
                await onNoteCreated()
            } catch {
                errorMessage = error.localizedDescription
                statusMessage = "Note creation failed."
                submissionState = .failed
            }
        case .autoCreateTodo(let request):
            do {
                _ = try await todosRepository.createTodo(request)
                submissionState = .result
                routeSummary = "Todo saved"
                statusMessage = "Todo saved automatically."
                await onTodoCreated()
            } catch {
                errorMessage = error.localizedDescription
                statusMessage = "Todo creation failed."
                submissionState = .failed
            }
        case .openSearch(let query):
            navigator.openSearchResults(query: query)
            submissionState = .result
            routeSummary = "Search opened"
            statusMessage = "Opened search results."
        case .requireConfirmation:
            pendingConfirmation = result
            submissionState = .result
            statusMessage = "Low confidence route. Confirm where this belongs."
        }
    }

    private func startTimer() {
        stopTimer()
        timerTask = Task { [weak self] in
            while let self, !Task.isCancelled, audioCaptureService.isRecording {
                if let recordingStartedAt {
                    elapsedSeconds = Date().timeIntervalSince(recordingStartedAt)
                }
                try? await Task.sleep(for: .seconds(0.25))
            }
        }
    }

    private func stopTimer() {
        timerTask?.cancel()
        timerTask = nil
    }
}

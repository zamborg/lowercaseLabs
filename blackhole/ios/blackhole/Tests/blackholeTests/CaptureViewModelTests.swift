import Foundation
import XCTest
@testable import blackhole

@MainActor
final class CaptureViewModelTests: XCTestCase {
    func testAutoCreatedNoteFailureSurfacesError() async throws {
        let audioURL = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).m4a")
        try Data("note payload".utf8).write(to: audioURL)
        defer { try? FileManager.default.removeItem(at: audioURL) }

        let model = CaptureViewModel(
            audioCaptureService: StubAudioCaptureService(audioURL: audioURL),
            transcriptionClient: StubTranscriptionSessionClient(
                events: [
                    .sessionCreated("session-1"),
                    .uploading,
                    .transcribing,
                    .finalResult(
                        TranscriptionFinalResult(
                            sessionId: "session-1",
                            transcript: "write down product direction",
                            route: .note,
                            routeConfidence: 0.96,
                            routePayload: [
                                "title": .string("Product direction"),
                                "bodyMarkdown": .string("write down product direction"),
                                "tags": .array([]),
                            ],
                            captureId: "capture-1"
                        )
                    ),
                ]
            ),
            routerClient: DefaultCaptureRouterClient(),
            notesRepository: FailingNotesRepository(),
            todosRepository: StubTodosRepository(),
            navigator: AppNavigator(),
            onNoteCreated: {},
            onTodoCreated: {}
        )

        await model.toggleRecording()
        XCTAssertEqual(model.submissionState, .recording)

        await model.toggleRecording()
        XCTAssertEqual(model.submissionState, .failed)
        XCTAssertEqual(model.statusMessage, "Note creation failed.")
        XCTAssertEqual(model.errorMessage, "note create failed")
    }
}

@MainActor
private final class StubAudioCaptureService: AudioCaptureService {
    private(set) var isRecording = false
    private let audioURL: URL

    init(audioURL: URL) {
        self.audioURL = audioURL
    }

    func requestPermission() async -> Bool {
        true
    }

    func startRecording() throws {
        isRecording = true
    }

    func stopRecording() throws -> URL {
        isRecording = false
        return audioURL
    }
}

@MainActor
private final class StubTranscriptionSessionClient: TranscriptionSessionClient {
    private let events: [TranscriptEvent]

    init(events: [TranscriptEvent]) {
        self.events = events
    }

    func start(audioURL: URL, mode: CaptureMode) async throws -> AsyncThrowingStream<TranscriptEvent, Error> {
        AsyncThrowingStream { continuation in
            for event in events {
                continuation.yield(event)
            }
            continuation.finish()
        }
    }
}

@MainActor
private struct FailingNotesRepository: NotesRepository {
    func listNotes(cursor: String?, limit: Int) async throws -> NotePage {
        NotePage(items: [], nextCursor: nil)
    }

    func getNote(id: String) async throws -> NoteRecord {
        throw APIError.transport("unimplemented")
    }

    func createNote(_ request: NoteCreateRequest) async throws -> NoteRecord {
        throw APIError.transport("note create failed")
    }

    func updateNote(id: String, request: NoteUpdateRequest) async throws -> NoteRecord {
        throw APIError.transport("unimplemented")
    }
}

@MainActor
private struct StubTodosRepository: TodosRepository {
    func listTodos(status: String?) async throws -> TodoPage {
        TodoPage(items: [], nextCursor: nil)
    }

    func getTodo(id: String) async throws -> TodoRecord {
        throw APIError.transport("unimplemented")
    }

    func createTodo(_ request: TodoCreateRequest) async throws -> TodoRecord {
        throw APIError.transport("unimplemented")
    }

    func updateTodo(id: String, request: TodoUpdateRequest) async throws -> TodoRecord {
        throw APIError.transport("unimplemented")
    }
}

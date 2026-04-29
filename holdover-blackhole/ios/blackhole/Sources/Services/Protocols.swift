import Foundation

@MainActor
protocol AudioCaptureService: AnyObject {
    var isRecording: Bool { get }
    func requestPermission() async -> Bool
    func startRecording() throws
    func stopRecording() throws -> URL
}

@MainActor
protocol TranscriptionSessionClient {
    func start(audioURL: URL, mode: CaptureMode) async throws -> AsyncThrowingStream<TranscriptEvent, Error>
}

@MainActor
protocol NotesRepository {
    func listNotes(cursor: String?, limit: Int) async throws -> NotePage
    func getNote(id: String) async throws -> NoteRecord
    func createNote(_ request: NoteCreateRequest) async throws -> NoteRecord
    func updateNote(id: String, request: NoteUpdateRequest) async throws -> NoteRecord
}

@MainActor
protocol TodosRepository {
    func listTodos(status: String?) async throws -> TodoPage
    func getTodo(id: String) async throws -> TodoRecord
    func createTodo(_ request: TodoCreateRequest) async throws -> TodoRecord
    func updateTodo(id: String, request: TodoUpdateRequest) async throws -> TodoRecord
}

@MainActor
protocol SearchRepository {
    func searchNotes(query: String, cursor: String?, limit: Int) async throws -> SearchPage
}

protocol AuthSessionStore {
    func loadSession() -> StoredAuthSession?
    func saveSession(_ session: StoredAuthSession)
    func clearSession()
    func loadBaseURL() -> String?
    func saveBaseURL(_ baseURL: String)
}

@MainActor
protocol AuthService {
    func signInDev(displayName: String, email: String?) async throws -> StoredAuthSession
}


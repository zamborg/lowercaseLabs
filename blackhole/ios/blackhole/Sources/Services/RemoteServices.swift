import Foundation

@MainActor
final class RemoteAuthService: AuthService {
    private let apiClient: APIClient

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func signInDev(displayName: String, email: String?) async throws -> StoredAuthSession {
        let request = DevAuthRequest(displayName: displayName, email: email)
        let body = try apiClient.encode(request)
        let urlRequest = try apiClient.request(path: "/auth/dev", method: "POST", body: body, requiresAuth: false)
        let response: AuthSessionResponse = try await apiClient.send(urlRequest, as: AuthSessionResponse.self)
        return StoredAuthSession(accessToken: response.accessToken, user: response.user)
    }
}

@MainActor
final class RemoteNotesRepository: NotesRepository {
    private let apiClient: APIClient

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func listNotes(cursor: String?, limit: Int) async throws -> NotePage {
        let path = "/blackhole/notes?limit=\(limit)\(cursor.map { "&cursor=\($0)" } ?? "")"
        let request = try apiClient.request(path: path, method: "GET")
        return try await apiClient.send(request, as: NotePage.self)
    }

    func getNote(id: String) async throws -> NoteRecord {
        let request = try apiClient.request(path: "/blackhole/notes/\(id)", method: "GET")
        return try await apiClient.send(request, as: NoteRecord.self)
    }

    func createNote(_ payload: NoteCreateRequest) async throws -> NoteRecord {
        let request = try apiClient.request(path: "/blackhole/notes", method: "POST", body: try apiClient.encode(payload))
        return try await apiClient.send(request, as: NoteRecord.self)
    }

    func updateNote(id: String, request payload: NoteUpdateRequest) async throws -> NoteRecord {
        let request = try apiClient.request(path: "/blackhole/notes/\(id)", method: "PATCH", body: try apiClient.encode(payload))
        return try await apiClient.send(request, as: NoteRecord.self)
    }
}

@MainActor
final class RemoteTodosRepository: TodosRepository {
    private let apiClient: APIClient

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func listTodos(status: String?) async throws -> TodoPage {
        let suffix = status.map { "?status=\($0)" } ?? ""
        let request = try apiClient.request(path: "/blackhole/todos\(suffix)", method: "GET")
        return try await apiClient.send(request, as: TodoPage.self)
    }

    func getTodo(id: String) async throws -> TodoRecord {
        let request = try apiClient.request(path: "/blackhole/todos/\(id)", method: "GET")
        return try await apiClient.send(request, as: TodoRecord.self)
    }

    func createTodo(_ payload: TodoCreateRequest) async throws -> TodoRecord {
        let request = try apiClient.request(path: "/blackhole/todos", method: "POST", body: try apiClient.encode(payload))
        return try await apiClient.send(request, as: TodoRecord.self)
    }

    func updateTodo(id: String, request payload: TodoUpdateRequest) async throws -> TodoRecord {
        let request = try apiClient.request(path: "/blackhole/todos/\(id)", method: "PATCH", body: try apiClient.encode(payload))
        return try await apiClient.send(request, as: TodoRecord.self)
    }
}

@MainActor
final class RemoteSearchRepository: SearchRepository {
    private let apiClient: APIClient

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func searchNotes(query: String, cursor: String?, limit: Int) async throws -> SearchPage {
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        let path = "/blackhole/search?q=\(encoded)&limit=\(limit)\(cursor.map { "&cursor=\($0)" } ?? "")"
        let request = try apiClient.request(path: path, method: "GET")
        return try await apiClient.send(request, as: SearchPage.self)
    }
}

@MainActor
final class RemoteTranscriptionSessionClient: TranscriptionSessionClient {
    private let apiClient: APIClient

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func start(audioURL: URL, mode: CaptureMode) async throws -> AsyncThrowingStream<TranscriptEvent, Error> {
        let data = try Data(contentsOf: audioURL)
        let boundary = "Boundary-\(UUID().uuidString)"
        let body = makeMultipartBody(boundary: boundary, fileName: audioURL.lastPathComponent, data: data, mode: mode)
        let createRequest = try apiClient.request(
            path: "/blackhole/transcriptions",
            method: "POST",
            body: body,
            contentType: "multipart/form-data; boundary=\(boundary)"
        )
        let response: TranscriptionSessionCreateResponse = try await apiClient.send(createRequest, as: TranscriptionSessionCreateResponse.self)

        return AsyncThrowingStream { continuation in
            continuation.yield(.sessionCreated(response.transcriptionSessionId))
            let task = Task {
                do {
                    var emittedUploading = false
                    var emittedTranscribing = false

                    for _ in 0..<120 {
                        let request = try apiClient.request(
                            path: "/blackhole/transcriptions/\(response.transcriptionSessionId)",
                            method: "GET"
                        )
                        let session = try await apiClient.send(request, as: TranscriptionSessionRecord.self)

                        switch session.status {
                        case "created", "uploading":
                            if !emittedUploading {
                                emittedUploading = true
                                continuation.yield(.uploading)
                            }
                        case "transcribing":
                            if !emittedUploading {
                                emittedUploading = true
                                continuation.yield(.uploading)
                            }
                            if !emittedTranscribing {
                                emittedTranscribing = true
                                continuation.yield(.transcribing)
                            }
                        case "completed":
                            if !emittedUploading {
                                continuation.yield(.uploading)
                            }
                            if !emittedTranscribing {
                                continuation.yield(.transcribing)
                            }
                            guard let result = session.asFinalResult() else {
                                throw APIError.transport("Completed transcription session was missing final result data.")
                            }
                            continuation.yield(.finalResult(result))
                            continuation.finish()
                            return
                        case "failed":
                            continuation.yield(.failed(session.errorMessage ?? "Transcription failed"))
                            continuation.finish()
                            return
                        default:
                            break
                        }

                        try await Task.sleep(for: .milliseconds(350))
                    }

                    continuation.yield(.failed("Transcription timed out before completion."))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    private func makeMultipartBody(boundary: String, fileName: String, data: Data, mode: CaptureMode) -> Data {
        var body = Data()
        let prefix = "--\(boundary)\r\n"

        body.append(Data(prefix.utf8))
        body.append(Data("Content-Disposition: form-data; name=\"mode\"\r\n\r\n".utf8))
        body.append(Data("\(mode.rawValue)\r\n".utf8))

        body.append(Data(prefix.utf8))
        body.append(Data("Content-Disposition: form-data; name=\"audio\"; filename=\"\(fileName)\"\r\n".utf8))
        body.append(Data("Content-Type: application/octet-stream\r\n\r\n".utf8))
        body.append(data)
        body.append(Data("\r\n".utf8))
        body.append(Data("--\(boundary)--\r\n".utf8))
        return body
    }
}

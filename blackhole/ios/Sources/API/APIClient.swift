import Foundation

actor APIClient {
    static let shared = APIClient()

    // Update this URL after deploying to Fly.io
    private let baseURL = URL(string: "https://blackhole.fly.dev")!
    private var sessionToken: String?

    func setSessionToken(_ token: String?) {
        sessionToken = token
    }

    // MARK: - Auth

    func signIn(identityToken: String) async throws -> String {
        struct Body: Encodable { let identity_token: String }
        struct Response: Decodable { let session_token: String }
        let response: Response = try await post("/auth/apple", body: Body(identity_token: identityToken), authenticated: false)
        return response.session_token
    }

    // MARK: - Items

    func createItem(content: String) async throws -> Item {
        struct Body: Encodable { let content: String }
        return try await post("/items", body: Body(content: content))
    }

    func listItems() async throws -> [Item] {
        return try await get("/items")
    }

    func updateItem(id: String, completed: Bool) async throws -> Item {
        struct Body: Encodable { let completed: Bool }
        return try await patch("/items/\(id)", body: Body(completed: completed))
    }

    func deleteItem(id: String) async throws {
        try await delete("/items/\(id)")
    }

    func search(query: String) async throws -> [Item] {
        struct Body: Encodable { let query: String }
        return try await post("/search", body: Body(query: query))
    }

    // MARK: - HTTP

    private func makeRequest(path: String, method: String, body: Data? = nil, authenticated: Bool = true) async throws -> Data {
        let url = URL(string: baseURL.absoluteString + path)!
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if authenticated, let token = sessionToken {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let body { req.httpBody = body }

        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            throw APIError.serverError(http.statusCode, String(data: data, encoding: .utf8) ?? "Unknown error")
        }
        return data
    }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        let data = try await makeRequest(path: path, method: "GET")
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func post<B: Encodable, T: Decodable>(_ path: String, body: B, authenticated: Bool = true) async throws -> T {
        let data = try await makeRequest(path: path, method: "POST", body: try JSONEncoder().encode(body), authenticated: authenticated)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func patch<B: Encodable, T: Decodable>(_ path: String, body: B) async throws -> T {
        let data = try await makeRequest(path: path, method: "PATCH", body: try JSONEncoder().encode(body))
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func delete(_ path: String) async throws {
        _ = try await makeRequest(path: path, method: "DELETE")
    }
}

enum APIError: LocalizedError {
    case invalidResponse
    case serverError(Int, String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse: return "Invalid server response."
        case .serverError(let code, let msg): return "Server error \(code): \(msg)"
        }
    }
}

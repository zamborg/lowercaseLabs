import Foundation

private actor ItemCache {
    static let shared = ItemCache()

    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private let fileManager = FileManager.default

    func loadItems(namespace: String?) -> [Item] {
        guard let fileURL = fileURL(namespace: namespace),
              let data = try? Data(contentsOf: fileURL),
              let items = try? decoder.decode([Item].self, from: data) else {
            return []
        }
        return items
    }

    func saveItems(_ items: [Item], namespace: String?) {
        guard let fileURL = fileURL(namespace: namespace),
              let data = try? encoder.encode(items) else { return }

        do {
            try fileManager.createDirectory(
                at: fileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: fileURL, options: [.atomic])
        } catch {
            assertionFailure("Failed to save item cache: \(error)")
        }
    }

    func upsert(_ newItems: [Item], namespace: String?) {
        guard !newItems.isEmpty else { return }
        var items = loadItems(namespace: namespace)
        for item in newItems {
            if let index = items.firstIndex(where: { $0.id == item.id }) {
                items[index] = item
            } else {
                items.insert(item, at: 0)
            }
        }
        saveItems(items.sorted { $0.createdAtDate > $1.createdAtDate }, namespace: namespace)
    }

    func deleteItem(id: String, namespace: String?) {
        var items = loadItems(namespace: namespace)
        items.removeAll { $0.id == id }
        saveItems(items, namespace: namespace)
    }

    func clear(namespace: String?) {
        guard let fileURL = fileURL(namespace: namespace),
              fileManager.fileExists(atPath: fileURL.path) else { return }
        try? fileManager.removeItem(at: fileURL)
    }

    private func fileURL(namespace: String?) -> URL? {
        guard let namespace, !namespace.isEmpty else { return nil }
        guard let baseURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return nil
        }
        return baseURL
            .appendingPathComponent("blackhole", isDirectory: true)
            .appendingPathComponent("items-\(namespace).json")
    }
}

actor APIClient {
    static let shared = APIClient()

    // Update this URL after deploying to Fly.io
    private let baseURL = URL(string: "https://blackhole.fly.dev")!
    private var sessionToken: String?
    private var cacheNamespace: String?

    func setSessionToken(_ token: String?) {
        sessionToken = token
        cacheNamespace = Self.cacheNamespace(for: token)
    }

    func cachedItems() async -> [Item] {
        await ItemCache.shared.loadItems(namespace: cacheNamespace)
    }

    func clearCachedItems() async {
        await ItemCache.shared.clear(namespace: cacheNamespace)
    }

    // MARK: - Auth

    func signIn(identityToken: String) async throws -> String {
        struct Body: Encodable { let identity_token: String }
        struct Response: Decodable { let session_token: String }
        let response: Response = try await post("/auth/apple", body: Body(identity_token: identityToken), authenticated: false)
        return response.session_token
    }

    // MARK: - Items

    func createItem(content: String) async throws -> [Item] {
        struct Body: Encodable { let content: String }
        let items: [Item] = try await post("/items", body: Body(content: content))
        await ItemCache.shared.upsert(items, namespace: cacheNamespace)
        return items
    }

    func editItem(
        id: String,
        title: String,
        content: String,
        type: String,
        epicId: String?,
        dueDate: String?,
        tags: [String]
    ) async throws -> Item {
        struct Body: Encodable {
            let title: String
            let content: String
            let type: String
            let epic_id: String?
            let due_date: String?
            let tags: [String]
        }
        let item: Item = try await patch("/items/\(id)", body: Body(
            title: title, content: content, type: type, epic_id: epicId, due_date: dueDate, tags: tags
        ))
        await ItemCache.shared.upsert([item], namespace: cacheNamespace)
        return item
    }

    func listItems() async throws -> [Item] {
        let items: [Item] = try await get("/items")
        await ItemCache.shared.saveItems(items, namespace: cacheNamespace)
        return items
    }

    func updateItem(id: String, completed: Bool) async throws -> Item {
        struct Body: Encodable { let completed: Bool }
        let item: Item = try await patch("/items/\(id)", body: Body(completed: completed))
        await ItemCache.shared.upsert([item], namespace: cacheNamespace)
        return item
    }

    func deleteItem(id: String) async throws {
        try await delete("/items/\(id)")
        await ItemCache.shared.deleteItem(id: id, namespace: cacheNamespace)
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

    private static func cacheNamespace(for token: String?) -> String? {
        guard let token,
              let payload = token.split(separator: ".").dropFirst().first,
              let data = Data(base64URLString: String(payload)),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let subject = object["sub"] as? String else {
            return nil
        }
        return subject
            .unicodeScalars
            .map { CharacterSet.alphanumerics.contains($0) ? String($0) : "-" }
            .joined()
    }
}

private extension Data {
    init?(base64URLString: String) {
        var base64 = base64URLString
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        let padding = base64.count % 4
        if padding > 0 {
            base64.append(String(repeating: "=", count: 4 - padding))
        }
        self.init(base64Encoded: base64)
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

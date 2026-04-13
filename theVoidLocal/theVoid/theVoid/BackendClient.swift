import Foundation

// MARK: - Networking

enum APIError: LocalizedError {
    case invalidURL
    case transport(String)
    case server(Int, String)
    case decoding

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid API URL."
        case .transport(let message):
            return message
        case .server(_, let message):
            return message
        case .decoding:
            return "Could not decode server response."
        }
    }
}

final class BackendClient {
    static let localBaseURLString = "http://127.0.0.1:8080"
    static let productionBaseURLString = "https://thevoid-local.fly.dev"
    static let defaultBaseURLString = BackendClient.productionBaseURLString

    private var baseURL: URL
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(baseURL: URL = URL(string: BackendClient.defaultBaseURLString)!) {
        self.baseURL = baseURL
        self.encoder = JSONEncoder()
        self.decoder = JSONDecoder()
        self.encoder.keyEncodingStrategy = .convertToSnakeCase
        self.decoder.keyDecodingStrategy = .convertFromSnakeCase
    }

    var baseURLString: String {
        baseURL.absoluteString
    }

    struct AuthPayload: Encodable {
        let identityToken: String
        let nonce: String?
        let displayName: String?
        let dailyCheckinTimeLocal: String
        let timezone: String
    }

    struct EntryCreatePayload: Encodable {
        let localDate: String
        let durationSeconds: Int
    }

    struct CompleteUploadPayload: Encodable {
        let contentType: String?
    }

    struct UpdateProfilePayload: Encodable {
        let displayName: String?
        let dailyCheckinTimeLocal: String
        let timezone: String
        let notificationEnabled: Bool
    }

    struct UpdateSocialDotPayload: Encodable {
        let moodScore: Double?
        let moodTags: [String]
        let dotColor: String?
    }

    struct InvitePayload: Encodable {
        let expiresInDays: Int
        let maxUses: Int
    }

    struct AcceptInvitePayload: Encodable {
        let token: String
    }

    struct FeedbackPayload: Encodable {
        let kind: String
        let message: String
    }

    struct HealthResponse: Decodable {
        let status: String
        let time: String
    }

    func updateBaseURL(_ value: String) throws {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed),
              let scheme = url.scheme?.lowercased(),
              (scheme == "http" || scheme == "https"),
              url.host != nil else {
            throw APIError.invalidURL
        }
        baseURL = url
    }

    private func buildRequest(
        url: URL,
        method: String,
        token: String? = nil,
        body: Data? = nil,
        contentType: String = "application/json"
    ) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = method
        if let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            request.httpBody = body
            request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        }
        return request
    }

    private func decodeError(_ data: Data, statusCode: Int) -> APIError {
        if let message = try? decoder.decode(APIMessage.self, from: data).message {
            return .server(statusCode, message)
        }
        let fallback = String(data: data, encoding: .utf8) ?? "Request failed"
        return .server(statusCode, fallback)
    }

    private func send<T: Decodable>(_ request: URLRequest, decode: T.Type) async throws -> T {
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw APIError.transport(error.localizedDescription)
        }

        guard let http = response as? HTTPURLResponse else {
            throw APIError.transport("Invalid response")
        }

        guard (200..<300).contains(http.statusCode) else {
            throw decodeError(data, statusCode: http.statusCode)
        }

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decoding
        }
    }

    private func sendVoid(_ request: URLRequest) async throws {
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw APIError.transport(error.localizedDescription)
        }

        guard let http = response as? HTTPURLResponse else {
            throw APIError.transport("Invalid response")
        }

        guard (200..<300).contains(http.statusCode) else {
            throw decodeError(data, statusCode: http.statusCode)
        }
    }

    func signInWithApple(identityToken: String, nonce: String?, displayName: String?, dailyCheckinTimeLocal: String, timezone: String) async throws -> APIAuthSession {
        let payload = AuthPayload(
            identityToken: identityToken,
            nonce: nonce,
            displayName: displayName,
            dailyCheckinTimeLocal: dailyCheckinTimeLocal,
            timezone: timezone
        )
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/auth/apple", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "POST", body: body)
        return try await send(request, decode: APIAuthSession.self)
    }

    func updateProfile(token: String, displayName: String?, dailyCheckinTimeLocal: String, timezone: String, notificationEnabled: Bool) async throws -> APIUserProfile {
        let payload = UpdateProfilePayload(
            displayName: displayName,
            dailyCheckinTimeLocal: dailyCheckinTimeLocal,
            timezone: timezone,
            notificationEnabled: notificationEnabled
        )
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/me", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "PATCH", token: token, body: body)
        return try await send(request, decode: APIUserProfile.self)
    }

    func createEntry(token: String, localDate: String, durationSeconds: Int) async throws -> APIEntryCreate {
        let payload = EntryCreatePayload(localDate: localDate, durationSeconds: durationSeconds)
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/entries", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "POST", token: token, body: body)
        return try await send(request, decode: APIEntryCreate.self)
    }

    func uploadAudio(uploadURL: String, token: String, audioData: Data) async throws {
        guard let url = URL(string: uploadURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(
            url: url,
            method: "PUT",
            token: token,
            body: audioData,
            contentType: "application/octet-stream"
        )
        try await sendVoid(request)
    }

    func completeUpload(entryID: String, token: String) async throws {
        let payload = CompleteUploadPayload(contentType: "audio/m4a")
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/entries/\(entryID)/complete_upload", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "POST", token: token, body: body)
        _ = try await send(request, decode: APIMessage.self)
    }

    func fetchEntries(token: String) async throws -> [APIEntry] {
        guard let url = URL(string: "/entries", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "GET", token: token)
        return try await send(request, decode: [APIEntry].self)
    }

    func fetchEntry(entryID: String, token: String) async throws -> APIEntry {
        guard let url = URL(string: "/entries/\(entryID)", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "GET", token: token)
        return try await send(request, decode: APIEntry.self)
    }

    func fetchAudio(entryID: String, token: String) async throws -> Data {
        guard let url = URL(string: "/entries/\(entryID)/audio", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "GET", token: token)

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw APIError.transport(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else {
            throw APIError.transport("Invalid response")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw decodeError(data, statusCode: http.statusCode)
        }
        return data
    }

    func fetchSocialDots(
        token: String,
        localDate: String? = nil,
        history: Bool = false,
        limit: Int? = nil
    ) async throws -> APISocialDotsEnvelope {
        guard let endpoint = URL(string: "/social/dots", relativeTo: baseURL),
              var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: true) else {
            throw APIError.invalidURL
        }

        var queryItems: [URLQueryItem] = []
        if let localDate, !localDate.isEmpty {
            queryItems.append(URLQueryItem(name: "local_date", value: localDate))
        }
        if history {
            queryItems.append(URLQueryItem(name: "history", value: "true"))
        }
        if let limit, limit > 0 {
            queryItems.append(URLQueryItem(name: "limit", value: String(limit)))
        }
        if !queryItems.isEmpty {
            components.queryItems = queryItems
        }
        guard let url = components.url else {
            throw APIError.invalidURL
        }

        let request = buildRequest(url: url, method: "GET", token: token)
        return try await send(request, decode: APISocialDotsEnvelope.self)
    }

    func publishLocalDot(
        token: String,
        localDate: String,
        moodScore: Double,
        moodTags: [String],
        dotColor: String?
    ) async throws {
        let payload = UpdateSocialDotPayload(
            moodScore: max(-2, min(moodScore, 2)),
            moodTags: Array(moodTags.prefix(8)),
            dotColor: dotColor
        )
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/social/presence/\(localDate)/dot", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "PUT", token: token, body: body)
        _ = try await send(request, decode: APIMessage.self)
    }

    func deleteLocalDot(token: String, localDate: String) async throws {
        guard let url = URL(string: "/social/presence/\(localDate)/dot", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "DELETE", token: token)
        _ = try await send(request, decode: APIMessage.self)
    }

    func createInvite(token: String) async throws -> APIInvite {
        let payload = InvitePayload(expiresInDays: 7, maxUses: 25)
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/friends/invite", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "POST", token: token, body: body)
        return try await send(request, decode: APIInvite.self)
    }

    func acceptInvite(token: String, inviteToken: String) async throws {
        let payload = AcceptInvitePayload(token: inviteToken)
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/friends/accept", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "POST", token: token, body: body)
        _ = try await send(request, decode: APIMessage.self)
    }

    func submitFeedback(token: String, kind: String, message: String) async throws {
        let payload = FeedbackPayload(kind: kind, message: message)
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/feedback", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "POST", token: token, body: body)
        _ = try await send(request, decode: APIMessage.self)
    }

    func healthCheck() async throws -> HealthResponse {
        guard let url = URL(string: "/health", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "GET")
        return try await send(request, decode: HealthResponse.self)
    }
}

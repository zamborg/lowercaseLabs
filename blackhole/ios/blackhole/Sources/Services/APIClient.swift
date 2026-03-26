import Foundation

enum APIError: LocalizedError {
    case invalidBaseURL
    case missingAuthToken
    case transport(String)
    case server(Int, String)
    case decoding

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL:
            return "Invalid API URL."
        case .missingAuthToken:
            return "Missing auth token."
        case .transport(let message):
            return message
        case .server(_, let message):
            return message
        case .decoding:
            return "Could not decode server response."
        }
    }
}

@MainActor
final class RuntimeConfiguration {
    var baseURLString: String
    var accessToken: String?

    init(baseURLString: String, accessToken: String?) {
        self.baseURLString = baseURLString
        self.accessToken = accessToken
    }
}

@MainActor
final class APIClient {
    private let config: RuntimeConfiguration
    private let encoder = JSONEncoder.blackhole
    private let decoder = JSONDecoder.blackhole

    init(config: RuntimeConfiguration) {
        self.config = config
    }

    func setBaseURL(_ value: String) {
        config.baseURLString = value
    }

    func setAccessToken(_ value: String?) {
        config.accessToken = value
    }

    func absoluteURL(path: String) throws -> URL {
        let trimmed = config.baseURLString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let baseURL = URL(string: trimmed),
              let scheme = baseURL.scheme,
              ["http", "https"].contains(scheme.lowercased()) else {
            throw APIError.invalidBaseURL
        }
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw APIError.invalidBaseURL
        }
        return url
    }

    func request(path: String, method: String, body: Data? = nil, requiresAuth: Bool = true, contentType: String = "application/json") throws -> URLRequest {
        let url = try absoluteURL(path: path)
        var request = URLRequest(url: url)
        request.httpMethod = method
        if requiresAuth {
            guard let token = config.accessToken else {
                throw APIError.missingAuthToken
            }
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            request.httpBody = body
            request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        }
        return request
    }

    func send<T: Decodable>(_ request: URLRequest, as type: T.Type) async throws -> T {
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
            let message = String(data: data, encoding: .utf8) ?? "Request failed"
            throw APIError.server(http.statusCode, message)
        }
        do {
            return try decoder.decode(type, from: data)
        } catch {
            throw APIError.decoding
        }
    }

    func sendVoid(_ request: URLRequest) async throws {
        let (_, response): (Data, URLResponse)
        do {
            (_, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw APIError.transport(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else {
            throw APIError.transport("Invalid response")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw APIError.server(http.statusCode, "Request failed")
        }
    }

    func encode<T: Encodable>(_ payload: T) throws -> Data {
        try encoder.encode(payload)
    }
}


import Foundation

public struct ToolSpec: Sendable {
    public let name: String
    public let description: String
    public let parameters: JSONValue
    public init(name: String, description: String, parameters: JSONValue) {
        self.name = name; self.description = description; self.parameters = parameters
    }
}

public struct ToolCall: Codable, Sendable, Equatable {
    public let id: String?
    public let name: String
    public let arguments: JSONValue

    public init(name: String, arguments: JSONValue, id: String? = nil) {
        self.name = name
        self.arguments = arguments
        self.id = id
    }
}

public enum ProviderError: Error, Sendable, Equatable {
    case http(status: Int, body: String)
    case noToolCall(raw: String)
    case badResponse(String)
    case transport(String)
}

public protocol LLMProvider: Sendable {
    var name: String { get }
    var model: String { get }
    func route(system: String, messages: [ConversationMessage], tools: [ToolSpec]) async throws -> ToolCall
    func listModels() async throws -> [String]
}

public protocol HTTPTransport: Sendable {
    func send(_ request: URLRequest) async throws -> (Data, Int)
}

public struct URLSessionTransport: HTTPTransport {
    public init() {}
    public func send(_ request: URLRequest) async throws -> (Data, Int) {
        do {
            let (data, resp) = try await URLSession.shared.data(for: request)
            return (data, (resp as? HTTPURLResponse)?.statusCode ?? -1)
        } catch {
            throw ProviderError.transport(error.localizedDescription)
        }
    }
}

func jsonRequest(url: String, method: String = "POST", headers: [String: String], body: JSONValue?, timeout: TimeInterval) -> URLRequest {
    var r = URLRequest(url: URL(string: url)!)
    r.httpMethod = method
    r.timeoutInterval = timeout
    for (k, v) in headers { r.setValue(v, forHTTPHeaderField: k) }
    if let body { r.setValue("application/json", forHTTPHeaderField: "Content-Type"); r.httpBody = try? JSONEncoder().encode(body) }
    return r
}

func checked(_ result: (Data, Int)) throws -> JSONValue {
    let (data, status) = result
    guard (200..<300).contains(status) else {
        // Truncate: error bodies can echo request details; never include headers (keys live there).
        throw ProviderError.http(status: status, body: String(decoding: data.prefix(300), as: UTF8.self))
    }
    do { return try JSONValue.parse(data) } catch { throw ProviderError.badResponse("invalid JSON") }
}

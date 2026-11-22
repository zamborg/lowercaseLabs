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
    /// Exact wire data for resuming this tool call on its originating model.
    var continuation: ToolContinuation?

    public init(name: String, arguments: JSONValue, id: String? = nil) {
        self.name = name
        self.arguments = arguments
        self.id = id
    }
}

struct ToolContinuation: Codable, Equatable, Sendable {
    enum Format: String, Codable, Sendable {
        case openAIResponses, anthropicMessages, openRouterChat
    }

    let format: Format
    let model: String
    let payload: JSONValue
}

extension ToolCall {
    func continuation(for format: ToolContinuation.Format, model: String) -> JSONValue? {
        guard let continuation, continuation.format == format, continuation.model == model else { return nil }
        return continuation.payload
    }
}

public struct AssistantCompletion: Sendable, Equatable {
    public let text: String?
    public let toolCall: ToolCall?

    public init(text: String? = nil, toolCall: ToolCall? = nil) {
        self.text = text
        self.toolCall = toolCall
    }
}

public enum ProviderError: Error, Sendable, Equatable {
    case http(status: Int, body: String)
    case badResponse(String)
    case transport(String)
    case outputLimit
    case unsupportedReasoning
}

public struct ProviderModel: Identifiable, Sendable, Equatable {
    public let id: String
    public let name: String
    public let reasoningEfforts: [ReasoningEffort]?

    public init(id: String, name: String? = nil, reasoningEfforts: [ReasoningEffort]? = nil) {
        self.id = id
        self.name = name ?? id
        self.reasoningEfforts = reasoningEfforts
    }
}

public enum CompletionProvider: String, CaseIterable, Sendable {
    case openai, anthropic, openrouter

    public var title: String {
        switch self {
        case .openai: "OpenAI"
        case .anthropic: "Anthropic"
        case .openrouter: "OpenRouter"
        }
    }

    public var apiKeyEnvironmentVariable: String { "\(rawValue.uppercased())_API_KEY" }
    public var defaultModel: String? { self == .anthropic ? "claude-haiku-5-5" : nil }

    public func reasoningOptions(for model: String, catalog: ProviderModel? = nil) -> [ReasoningEffort] {
        let model = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !model.isEmpty else { return [] }
        if let efforts = catalog?.reasoningEfforts { return efforts }
        switch self {
        case .openai: return ReasoningEffort.openAIOptions(for: model)
        case .anthropic: return ReasoningEffort.anthropicOptions(for: model)
        case .openrouter: return [.none, .low, .medium, .high]
        }
    }

    public func makeProvider(apiKey: String, model: String, reasoningEffort: ReasoningEffort = .automatic) -> any LLMProvider {
        switch self {
        case .openai: OpenAIProvider(apiKey: apiKey, model: model, reasoningEffort: reasoningEffort)
        case .anthropic: AnthropicProvider(apiKey: apiKey, model: model, reasoningEffort: reasoningEffort)
        case .openrouter: OpenRouterProvider(apiKey: apiKey, model: model, reasoningEffort: reasoningEffort)
        }
    }
}

public protocol LLMProvider: Sendable {
    var name: String { get }
    var model: String { get }
    func complete(system: String, messages: [ConversationMessage], tools: [ToolSpec]) async throws -> AssistantCompletion
    func listModels() async throws -> [ProviderModel]
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
        } catch let error as URLError where error.code == .cancelled && Task.isCancelled {
            throw CancellationError()
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

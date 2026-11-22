import Foundation

/// OpenAI Chat Completions with optional function calls.
public struct OpenAIProvider: LLMProvider {
    public let name = "openai"
    public let model: String
    private let apiKey: String
    private let transport: HTTPTransport
    private let timeout: TimeInterval
    private let maxTokens: Int

    public init(apiKey: String, model: String, transport: HTTPTransport = URLSessionTransport(),
                timeout: TimeInterval = 10, maxTokens: Int = 400) {
        self.apiKey = apiKey; self.model = model; self.transport = transport
        self.timeout = timeout; self.maxTokens = maxTokens
    }

    private var headers: [String: String] { ["Authorization": "Bearer \(apiKey)"] }

    func requestBody(system: String, messages: [ConversationMessage], tools: [ToolSpec], reasoningNone: Bool = true) -> JSONValue {
        var body = ChatCompletions.fields(model: model, system: system, messages: messages, tools: tools)
        if !tools.isEmpty {
            body["tool_choice"] = .string("auto")
            body["parallel_tool_calls"] = .bool(false)
        }
        body["max_completion_tokens"] = .number(Double(maxTokens))
        // Routing needs no thinking, and some models reject tools while reasoning is on.
        if reasoningNone { body["reasoning_effort"] = .string("none") }
        return .object(body)
    }

    public func complete(system: String, messages: [ConversationMessage], tools: [ToolSpec]) async throws -> AssistantCompletion {
        func attempt(_ none: Bool) async throws -> JSONValue {
            let req = jsonRequest(url: "https://api.openai.com/v1/chat/completions", headers: headers,
                                  body: requestBody(system: system, messages: messages, tools: tools, reasoningNone: none), timeout: timeout)
            return try checked(try await transport.send(req))
        }
        do { return try ChatCompletions.parseCompletion(try await attempt(true)) }
        catch ProviderError.http(let status, let body) where status == 400 && body.contains("reasoning_effort") {
            // Model doesn't accept reasoning_effort "none" (older models): retry without it.
            return try ChatCompletions.parseCompletion(try await attempt(false))
        }
    }

    public func listModels() async throws -> [ProviderModel] {
        let req = jsonRequest(url: "https://api.openai.com/v1/models", method: "GET", headers: headers, body: nil, timeout: timeout)
        let json = try checked(try await transport.send(req))
        guard case .array(let data)? = json["data"] else { throw ProviderError.badResponse("no data") }
        let excluded = ["audio", "realtime", "tts", "transcribe", "image", "search", "embedding", "instruct", "codex", "moderation"]
        return data.compactMap { $0["id"]?.stringValue }
            .filter { id in (id.hasPrefix("gpt") || id.hasPrefix("o")) && !excluded.contains(where: id.contains) }
            .sorted()
            .map { ProviderModel(id: $0) }
    }
}

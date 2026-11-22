import Foundation

/// OpenAI completions and Responses, with local conversation history.
public struct OpenAIProvider: LLMProvider {
    public let name = "openai"
    public let model: String
    private let apiKey: String
    private let transport: HTTPTransport
    private let timeout: TimeInterval
    private let maxTokens: Int
    private let responsesEffort: ReasoningEffort?

    public init(apiKey: String, model: String, reasoningEffort: ReasoningEffort = .automatic,
                transport: HTTPTransport = URLSessionTransport(), timeout: TimeInterval? = nil, maxTokens: Int? = nil) {
        self.apiKey = apiKey; self.model = model; self.transport = transport
        let effort = reasoningEffort == .automatic ? ReasoningEffort.openAIOptions(for: model).first : reasoningEffort
        self.responsesEffort = effort
        self.timeout = timeout ?? (effort ?? .automatic).requestTimeout
        self.maxTokens = maxTokens ?? (effort ?? .automatic).outputTokenLimit
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
        if let effort = responsesEffort {
            return try await completeWithResponses(system: system, messages: messages, tools: tools, effort: effort)
        }
        func attempt(_ none: Bool) async throws -> JSONValue {
            let req = jsonRequest(url: "https://api.openai.com/v1/chat/completions", headers: headers,
                                  body: requestBody(system: system, messages: messages, tools: tools, reasoningNone: none), timeout: timeout)
            return try checked(try await transport.send(req))
        }
        do { return try ChatCompletions.parseCompletion(try await attempt(true)) }
        catch ProviderError.http(let status, let body) where status == 400 && body.contains("reasoning_effort") {
            // Older chat models reject the parameter altogether.
            return try ChatCompletions.parseCompletion(try await attempt(false))
        }
    }

    private func completeWithResponses(system: String, messages: [ConversationMessage], tools: [ToolSpec],
                                       effort: ReasoningEffort) async throws -> AssistantCompletion {
        let req = jsonRequest(url: "https://api.openai.com/v1/responses", headers: headers,
                              body: OpenAIResponses.requestBody(model: model, system: system, messages: messages, tools: tools,
                                                                 effort: effort, maxTokens: maxTokens), timeout: timeout)
        let result = try await transport.send(req)
        do { return try OpenAIResponses.parseCompletion(checked(result), model: model) }
        catch ProviderError.http(let status, let body) where status == 400 && body.lowercased().contains("reasoning") {
            throw ProviderError.unsupportedReasoning
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

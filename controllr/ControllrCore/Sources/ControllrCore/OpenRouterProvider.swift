import Foundation

public struct OpenRouterProvider: LLMProvider {
    public let name = "openrouter"
    public let model: String
    private let apiKey: String
    private let transport: HTTPTransport
    private let timeout: TimeInterval
    private let maxTokens: Int

    public init(apiKey: String, model: String, transport: HTTPTransport = URLSessionTransport(),
                timeout: TimeInterval = 20, maxTokens: Int = 400) {
        self.apiKey = apiKey
        self.model = model
        self.transport = transport
        self.timeout = timeout
        self.maxTokens = maxTokens
    }

    private var headers: [String: String] { ["Authorization": "Bearer \(apiKey)"] }

    public func complete(system: String, messages: [ConversationMessage], tools: [ToolSpec]) async throws -> AssistantCompletion {
        var body = ChatCompletions.fields(model: model, system: system, messages: messages, tools: tools)
        body["max_tokens"] = .number(Double(maxTokens))
        if !tools.isEmpty {
            body["provider"] = .object(["require_parameters": .bool(true)])
        }
        let req = jsonRequest(url: "https://openrouter.ai/api/v1/chat/completions", headers: headers,
                              body: .object(body), timeout: timeout)
        let json = try checked(try await transport.send(req))
        try Self.checkGenerationError(json)
        return try ChatCompletions.parseCompletion(json)
    }

    public func listModels() async throws -> [ProviderModel] {
        let req = jsonRequest(url: "https://openrouter.ai/api/v1/models/user?output_modalities=text",
                              method: "GET", headers: headers, body: nil, timeout: timeout)
        let json = try checked(try await transport.send(req))
        guard case .array(let models)? = json["data"] else { throw ProviderError.badResponse("no models") }
        return models.compactMap { model in
            guard let id = model["id"]?.stringValue,
                  case .array(let supported)? = model["supported_parameters"],
                  supported.contains(.string("tools")) else { return nil }
            if case .array(let outputs)? = model["architecture"]?["output_modalities"],
               !outputs.contains(.string("text")) { return nil }
            if case .array(let inputs)? = model["architecture"]?["input_modalities"],
               !inputs.contains(.string("text")) { return nil }
            return ProviderModel(id: id, name: model["name"]?.stringValue)
        }
    }

    private static func checkGenerationError(_ json: JSONValue) throws {
        let error: JSONValue?
        if let embedded = json["error"]?.objectValue {
            error = .object(embedded)
        } else if case .array(let choices)? = json["choices"], let embedded = choices.first?["error"]?.objectValue {
            error = .object(embedded)
        } else {
            error = nil
        }
        guard let error else { return }
        let status = error["code"]?.numberValue.flatMap(Int.init(exactly:)) ?? 502
        throw ProviderError.http(status: status, body: "OpenRouter generation error")
    }
}

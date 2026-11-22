import Foundation

public struct OpenRouterProvider: LLMProvider {
    public let name = "openrouter"
    public let model: String
    private let apiKey: String
    private let transport: HTTPTransport
    private let timeout: TimeInterval
    private let maxTokens: Int
    private let reasoningEffort: ReasoningEffort

    public init(apiKey: String, model: String, reasoningEffort: ReasoningEffort = .automatic,
                transport: HTTPTransport = URLSessionTransport(), timeout: TimeInterval? = nil, maxTokens: Int? = nil) {
        self.apiKey = apiKey
        self.model = model
        self.transport = transport
        self.reasoningEffort = reasoningEffort
        self.timeout = timeout ?? max(20, reasoningEffort.requestTimeout)
        self.maxTokens = maxTokens ?? reasoningEffort.outputTokenLimit
    }

    private var headers: [String: String] { ["Authorization": "Bearer \(apiKey)"] }

    public func complete(system: String, messages: [ConversationMessage], tools: [ToolSpec]) async throws -> AssistantCompletion {
        var body = ChatCompletions.fields(model: model, system: system, messages: messages, tools: tools, useOpenRouterHistory: true)
        body["max_tokens"] = .number(Double(maxTokens))
        if reasoningEffort != .automatic { body["reasoning"] = .object(["effort": .string(reasoningEffort.rawValue)]) }
        if !tools.isEmpty || reasoningEffort != .automatic {
            body["provider"] = .object(["require_parameters": .bool(true)])
        }
        let req = jsonRequest(url: "https://openrouter.ai/api/v1/chat/completions", headers: headers,
                              body: .object(body), timeout: timeout)
        let json = try checked(try await transport.send(req))
        try Self.checkGenerationError(json)
        return try ChatCompletions.parseCompletion(json, openRouterModel: model)
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
            let supportedEfforts = model["reasoning"]?["supported_efforts"]
            let noReasoningOff = model["reasoning"]?["mandatory"] == .bool(true)
            let efforts: [ReasoningEffort]
            switch supportedEfforts {
            case .array(let levels):
                efforts = ReasoningEffort.allCases.filter { levels.contains(.string($0.rawValue)) }
            case .null:
                efforts = ReasoningEffort.allCases
            default:
                efforts = supported.contains(.string("reasoning")) && model["reasoning"] == nil
                    ? [.none, .low, .medium, .high] : []
            }
            return ProviderModel(id: id, name: model["name"]?.stringValue,
                                 reasoningEfforts: efforts.filter { $0 != .automatic && !(noReasoningOff && $0 == .none) })
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

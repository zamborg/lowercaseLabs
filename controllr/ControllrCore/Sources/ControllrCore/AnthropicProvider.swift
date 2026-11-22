import Foundation

/// Anthropic Messages API with optional tool use.
public struct AnthropicProvider: LLMProvider {
    public let name = "anthropic"
    public let model: String
    private let apiKey: String
    private let transport: HTTPTransport
    private let timeout: TimeInterval
    private let maxTokens: Int
    private let reasoningEffort: ReasoningEffort

    public init(apiKey: String, model: String, reasoningEffort: ReasoningEffort = .automatic,
                transport: HTTPTransport = URLSessionTransport(), timeout: TimeInterval? = nil, maxTokens: Int? = nil) {
        self.apiKey = apiKey; self.model = model; self.transport = transport
        self.reasoningEffort = reasoningEffort
        self.timeout = timeout ?? reasoningEffort.requestTimeout
        self.maxTokens = maxTokens ?? reasoningEffort.outputTokenLimit
    }

    private var headers: [String: String] { ["x-api-key": apiKey, "anthropic-version": "2023-06-01"] }

    func requestBody(system: String, messages: [ConversationMessage], tools: [ToolSpec]) -> JSONValue {
        let history: [JSONValue] = messages.flatMap { message -> [JSONValue] in
            switch message {
            case .user(let text):
                return [.object(["role": .string("user"), "content": .string(text)])]
            case .assistant(let text), .followUp(_, _, let text):
                return [.object(["role": .string("assistant"), "content": .string(text)])]
            case .tool(let call, let output, let assistantText):
                let id = call.id ?? "toolu_\(UUID().uuidString)"
                let use: JSONValue = .object([
                    "type": .string("tool_use"), "id": .string(id),
                    "name": .string(call.name), "input": call.arguments,
                ])
                let result: JSONValue = .object([
                    "type": .string("tool_result"), "tool_use_id": .string(id), "content": .string(output),
                ])
                var blocks: [JSONValue] = []
                if case .array(let original)? = call.continuation(for: .anthropicMessages, model: model) {
                    blocks = original
                } else {
                    if let assistantText { blocks.append(.object(["type": .string("text"), "text": .string(assistantText)])) }
                    blocks.append(use)
                }
                return [
                    .object(["role": .string("assistant"), "content": .array(blocks)]),
                    .object(["role": .string("user"), "content": .array([result])]),
                ]
            }
        }
        var body: [String: JSONValue] = [
            "model": .string(model),
            "max_tokens": .number(Double(maxTokens)),
            "system": .string(system),
            "messages": .array(history),
        ]
        if !tools.isEmpty {
            body["tools"] = .array(tools.map { t in
                .object(["name": .string(t.name), "description": .string(t.description), "input_schema": t.parameters])
            })
            body["tool_choice"] = .object(["type": .string("auto"), "disable_parallel_tool_use": .bool(true)])
        }
        if reasoningEffort != .automatic {
            body["thinking"] = .object(["type": .string("adaptive"), "display": .string("omitted")])
            body["output_config"] = .object(["effort": .string(reasoningEffort.rawValue)])
        }
        return .object(body)
    }

    public func complete(system: String, messages: [ConversationMessage], tools: [ToolSpec]) async throws -> AssistantCompletion {
        guard reasoningEffort != .none, reasoningEffort != .minimal else { throw ProviderError.unsupportedReasoning }
        let req = jsonRequest(url: "https://api.anthropic.com/v1/messages", headers: headers,
                              body: requestBody(system: system, messages: messages, tools: tools), timeout: timeout)
        let result = try await transport.send(req)
        do { return try Self.parseCompletion(checked(result), model: model) }
        catch ProviderError.http(let status, let body) where reasoningEffort != .automatic && status == 400
            && (body.lowercased().contains("effort") || body.lowercased().contains("thinking")) {
            throw ProviderError.unsupportedReasoning
        }
    }

    static func parseCompletion(_ json: JSONValue, model: String? = nil) throws -> AssistantCompletion {
        if json["stop_reason"] == .string("max_tokens") { throw ProviderError.outputLimit }
        guard case .array(let content)? = json["content"] else { throw ProviderError.badResponse("no content") }
        let text = content.compactMap { $0["type"]?.stringValue == "text" ? $0["text"]?.stringValue : nil }
            .joined(separator: "\n")
        let calls = content.filter { $0["type"]?.stringValue == "tool_use" }
        guard !calls.isEmpty else { return AssistantCompletion(text: text.isEmpty ? nil : text) }
        guard calls.count == 1, let block = calls.first,
              let name = block["name"]?.stringValue, let input = block["input"] else {
            throw ProviderError.badResponse("invalid tool use")
        }
        var call = ToolCall(name: name, arguments: input, id: block["id"]?.stringValue)
        if let model, content.contains(where: { $0["type"] == .string("thinking") || $0["type"] == .string("redacted_thinking") }) {
            call.continuation = ToolContinuation(format: .anthropicMessages, model: model, payload: .array(content))
        }
        return AssistantCompletion(text: text.isEmpty ? nil : text, toolCall: call)
    }

    public func listModels() async throws -> [ProviderModel] {
        var models: [ProviderModel] = []
        var nextPage: String?
        var seenCursors: Set<String> = []
        repeat {
            var url = URLComponents()
            url.scheme = "https"
            url.host = "api.anthropic.com"
            url.path = "/v1/models"
            url.queryItems = [URLQueryItem(name: "limit", value: "100")]
            if let nextPage { url.queryItems?.append(URLQueryItem(name: "after_id", value: nextPage)) }

            let req = jsonRequest(url: url.string!, method: "GET", headers: headers, body: nil, timeout: timeout)
            let json = try checked(try await transport.send(req))
            guard case .array(let data)? = json["data"] else { throw ProviderError.badResponse("no models") }
            models.append(contentsOf: data.compactMap { item in
                guard let id = item["id"]?.stringValue else { return nil }
                let capabilities = item["capabilities"]
                let adaptive = capabilities?["thinking"]?["types"]?["adaptive"]?["supported"]
                let efforts = capabilities?["effort"]
                let levels: [ReasoningEffort]?
                if let adaptive, let efforts {
                    levels = adaptive == .bool(true)
                        ? [.low, .medium, .high, .xhigh, .max].filter { efforts[$0.rawValue]?["supported"] == .bool(true) }
                        : []
                } else {
                    levels = nil
                }
                return ProviderModel(id: id, name: item["display_name"]?.stringValue, reasoningEfforts: levels)
            })
            if json["has_more"] == .bool(true) {
                guard let lastID = json["last_id"]?.stringValue, !lastID.isEmpty, seenCursors.insert(lastID).inserted else {
                    throw ProviderError.badResponse("invalid model list cursor")
                }
                nextPage = lastID
            } else {
                nextPage = nil
            }
        } while nextPage != nil
        return models
    }
}

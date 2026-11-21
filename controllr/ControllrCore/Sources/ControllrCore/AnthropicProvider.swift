import Foundation

/// Anthropic Messages API with forced tool use.
public struct AnthropicProvider: LLMProvider {
    public let name = "anthropic"
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

    private var headers: [String: String] { ["x-api-key": apiKey, "anthropic-version": "2023-06-01"] }

    func requestBody(system: String, messages: [ConversationMessage], tools: [ToolSpec]) -> JSONValue {
        let history: [JSONValue] = messages.flatMap { message -> [JSONValue] in
            switch message {
            case .user(let text):
                return [.object(["role": .string("user"), "content": .string(text)])]
            case .assistant(let text):
                return [.object(["role": .string("assistant"), "content": .string(text)])]
            case .tool(let call, let output):
                let id = call.id ?? "toolu_\(UUID().uuidString)"
                let use: JSONValue = .object([
                    "type": .string("tool_use"), "id": .string(id),
                    "name": .string(call.name), "input": call.arguments,
                ])
                let result: JSONValue = .object([
                    "type": .string("tool_result"), "tool_use_id": .string(id), "content": .string(output),
                ])
                return [
                    .object(["role": .string("assistant"), "content": .array([use])]),
                    .object(["role": .string("user"), "content": .array([result])]),
                ]
            }
        }
        return .object([
            "model": .string(model),
            "max_tokens": .number(Double(maxTokens)),
            "system": .string(system),
            "messages": .array(history),
            "tools": .array(tools.map { t in
                .object(["name": .string(t.name), "description": .string(t.description), "input_schema": t.parameters])
            }),
            "tool_choice": .object(["type": .string("any"), "disable_parallel_tool_use": .bool(true)]),
        ])
    }

    public func route(system: String, messages: [ConversationMessage], tools: [ToolSpec]) async throws -> ToolCall {
        let req = jsonRequest(url: "https://api.anthropic.com/v1/messages", headers: headers,
                              body: requestBody(system: system, messages: messages, tools: tools), timeout: timeout)
        let json = try checked(try await transport.send(req))
        return try Self.parseToolCall(json)
    }

    static func parseToolCall(_ json: JSONValue) throws -> ToolCall {
        guard case .array(let content)? = json["content"] else { throw ProviderError.badResponse("no content") }
        for block in content where block["type"]?.stringValue == "tool_use" {
            if let name = block["name"]?.stringValue, let input = block["input"] {
                return ToolCall(name: name, arguments: input, id: block["id"]?.stringValue)
            }
        }
        throw ProviderError.noToolCall(raw: json.jsonString())
    }

    public func listModels() async throws -> [String] {
        let req = jsonRequest(url: "https://api.anthropic.com/v1/models?limit=100", method: "GET", headers: headers, body: nil, timeout: timeout)
        let json = try checked(try await transport.send(req))
        guard case .array(let data)? = json["data"] else { throw ProviderError.badResponse("no data") }
        return data.compactMap { $0["id"]?.stringValue }
    }
}

import Foundation

/// Anthropic Messages API with optional tool use.
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
                if let assistantText { blocks.append(.object(["type": .string("text"), "text": .string(assistantText)])) }
                blocks.append(use)
                return [
                    .object(["role": .string("assistant"), "content": .array(blocks)]),
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
            "tool_choice": .object(["type": .string("auto"), "disable_parallel_tool_use": .bool(true)]),
        ])
    }

    public func complete(system: String, messages: [ConversationMessage], tools: [ToolSpec]) async throws -> AssistantCompletion {
        let req = jsonRequest(url: "https://api.anthropic.com/v1/messages", headers: headers,
                              body: requestBody(system: system, messages: messages, tools: tools), timeout: timeout)
        let json = try checked(try await transport.send(req))
        return try Self.parseCompletion(json)
    }

    static func parseCompletion(_ json: JSONValue) throws -> AssistantCompletion {
        guard case .array(let content)? = json["content"] else { throw ProviderError.badResponse("no content") }
        let text = content.compactMap { $0["type"]?.stringValue == "text" ? $0["text"]?.stringValue : nil }
            .joined(separator: "\n")
        let calls = content.filter { $0["type"]?.stringValue == "tool_use" }
        guard !calls.isEmpty else { return AssistantCompletion(text: text.isEmpty ? nil : text) }
        guard calls.count == 1, let block = calls.first,
              let name = block["name"]?.stringValue, let input = block["input"] else {
            throw ProviderError.badResponse("invalid tool use")
        }
        return AssistantCompletion(text: text.isEmpty ? nil : text,
                                   toolCall: ToolCall(name: name, arguments: input, id: block["id"]?.stringValue))
    }

    public func listModels() async throws -> [String] {
        let req = jsonRequest(url: "https://api.anthropic.com/v1/models?limit=100", method: "GET", headers: headers, body: nil, timeout: timeout)
        let json = try checked(try await transport.send(req))
        guard case .array(let data)? = json["data"] else { throw ProviderError.badResponse("no data") }
        return data.compactMap { $0["id"]?.stringValue }
    }
}

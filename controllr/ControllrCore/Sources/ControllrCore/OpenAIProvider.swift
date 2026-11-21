import Foundation

/// OpenAI Chat Completions with forced tool calling.
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
        let history: [JSONValue] = messages.flatMap { message -> [JSONValue] in
            switch message {
            case .user(let text):
                return [.object(["role": .string("user"), "content": .string(text)])]
            case .assistant(let text):
                return [.object(["role": .string("assistant"), "content": .string(text)])]
            case .tool(let call, let output):
                let id = call.id ?? "call_\(UUID().uuidString)"
                let toolCall: JSONValue = .object([
                    "id": .string(id), "type": .string("function"),
                    "function": .object(["name": .string(call.name), "arguments": .string(call.arguments.jsonString())]),
                ])
                return [
                    .object(["role": .string("assistant"), "tool_calls": .array([toolCall])]),
                    .object(["role": .string("tool"), "tool_call_id": .string(id), "content": .string(output)]),
                ]
            }
        }
        var body: [String: JSONValue] = [
            "model": .string(model),
            "messages": .array([.object(["role": .string("system"), "content": .string(system)])] + history),
            "tools": .array(tools.map { t in
                .object(["type": .string("function"),
                         "function": .object(["name": .string(t.name), "description": .string(t.description), "parameters": t.parameters])])
            }),
            "tool_choice": .string("required"),
            "parallel_tool_calls": .bool(false),
            "max_completion_tokens": .number(Double(maxTokens)),
        ]
        // Routing needs no thinking, and some models reject tools while reasoning is on.
        if reasoningNone { body["reasoning_effort"] = .string("none") }
        return .object(body)
    }

    public func route(system: String, messages: [ConversationMessage], tools: [ToolSpec]) async throws -> ToolCall {
        func attempt(_ none: Bool) async throws -> JSONValue {
            let req = jsonRequest(url: "https://api.openai.com/v1/chat/completions", headers: headers,
                                  body: requestBody(system: system, messages: messages, tools: tools, reasoningNone: none), timeout: timeout)
            return try checked(try await transport.send(req))
        }
        do { return try Self.parseToolCall(try await attempt(true)) }
        catch ProviderError.http(let status, let body) where status == 400 && body.contains("reasoning_effort") {
            // Model doesn't accept reasoning_effort "none" (older models): retry without it.
            return try Self.parseToolCall(try await attempt(false))
        }
    }

    static func parseToolCall(_ json: JSONValue) throws -> ToolCall {
        guard case .array(let choices)? = json["choices"], let message = choices.first?["message"] else {
            throw ProviderError.badResponse("no choices")
        }
        guard case .array(let calls)? = message["tool_calls"], let first = calls.first,
              let fn = first["function"], let name = fn["name"]?.stringValue else {
            throw ProviderError.noToolCall(raw: message.jsonString())
        }
        let argText = fn["arguments"]?.stringValue ?? "{}"
        guard let args = try? JSONValue.parse(Data(argText.utf8)) else {
            throw ProviderError.badResponse("tool arguments are not JSON")
        }
        return ToolCall(name: name, arguments: args, id: first["id"]?.stringValue)
    }

    public func listModels() async throws -> [String] {
        let req = jsonRequest(url: "https://api.openai.com/v1/models", method: "GET", headers: headers, body: nil, timeout: timeout)
        let json = try checked(try await transport.send(req))
        guard case .array(let data)? = json["data"] else { throw ProviderError.badResponse("no data") }
        return data.compactMap { $0["id"]?.stringValue }.sorted()
    }
}

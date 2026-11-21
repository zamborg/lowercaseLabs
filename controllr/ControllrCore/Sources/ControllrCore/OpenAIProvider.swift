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

    func requestBody(system: String, transcript: String, tools: [ToolSpec]) -> JSONValue {
        .object([
            "model": .string(model),
            "messages": .array([
                .object(["role": .string("system"), "content": .string(system)]),
                .object(["role": .string("user"), "content": .string(transcript)]),
            ]),
            "tools": .array(tools.map { t in
                .object(["type": .string("function"),
                         "function": .object(["name": .string(t.name), "description": .string(t.description), "parameters": t.parameters])])
            }),
            "tool_choice": .string("required"),
            "parallel_tool_calls": .bool(false),
            "max_completion_tokens": .number(Double(maxTokens)),
        ])
    }

    public func route(system: String, transcript: String, tools: [ToolSpec]) async throws -> ToolCall {
        let req = jsonRequest(url: "https://api.openai.com/v1/chat/completions", headers: headers,
                              body: requestBody(system: system, transcript: transcript, tools: tools), timeout: timeout)
        let json = try checked(try await transport.send(req))
        return try Self.parseToolCall(json)
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
        return ToolCall(name: name, arguments: args)
    }

    public func listModels() async throws -> [String] {
        let req = jsonRequest(url: "https://api.openai.com/v1/models", method: "GET", headers: headers, body: nil, timeout: timeout)
        let json = try checked(try await transport.send(req))
        guard case .array(let data)? = json["data"] else { throw ProviderError.badResponse("no data") }
        return data.compactMap { $0["id"]?.stringValue }.sorted()
    }
}

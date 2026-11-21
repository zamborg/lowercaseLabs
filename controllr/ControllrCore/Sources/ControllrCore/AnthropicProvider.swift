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

    func requestBody(system: String, transcript: String, tools: [ToolSpec]) -> JSONValue {
        .object([
            "model": .string(model),
            "max_tokens": .number(Double(maxTokens)),
            "system": .string(system),
            "messages": .array([.object(["role": .string("user"), "content": .string(transcript)])]),
            "tools": .array(tools.map { t in
                .object(["name": .string(t.name), "description": .string(t.description), "input_schema": t.parameters])
            }),
            "tool_choice": .object(["type": .string("any"), "disable_parallel_tool_use": .bool(true)]),
        ])
    }

    public func route(system: String, transcript: String, tools: [ToolSpec]) async throws -> ToolCall {
        let req = jsonRequest(url: "https://api.anthropic.com/v1/messages", headers: headers,
                              body: requestBody(system: system, transcript: transcript, tools: tools), timeout: timeout)
        let json = try checked(try await transport.send(req))
        return try Self.parseToolCall(json)
    }

    static func parseToolCall(_ json: JSONValue) throws -> ToolCall {
        guard case .array(let content)? = json["content"] else { throw ProviderError.badResponse("no content") }
        for block in content where block["type"]?.stringValue == "tool_use" {
            if let name = block["name"]?.stringValue, let input = block["input"] { return ToolCall(name: name, arguments: input) }
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

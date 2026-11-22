import Foundation
import XCTest
@testable import ControllrCore

private actor StubTransport: HTTPTransport {
    private var responses: [String]
    private(set) var requests: [URLRequest] = []

    init(_ responses: [String]) { self.responses = responses }

    func send(_ request: URLRequest) async throws -> (Data, Int) {
        requests.append(request)
        guard !responses.isEmpty else { throw ProviderError.badResponse("Unexpected request") }
        return (Data(responses.removeFirst().utf8), 200)
    }
}

final class ProviderAdapterTests: XCTestCase {
    func testOpenAIResponsesKeepsReasoningAndToolCallThroughAFollowUp() async throws {
        let initial = #"""
        {"status":"completed","output":[
          {"type":"reasoning","id":"rs_123","summary":[],"encrypted_content":"opaque"},
          {"type":"message","role":"assistant","content":[{"type":"output_text","text":"I can help."}]},
          {"type":"function_call","id":"fc_123","call_id":"call_123","name":"clarify","arguments":"{\"question\":\"Indoors?\"}"}
        ]}
        """#
        let transport = StubTransport([
            initial,
            #"{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Try stretching indoors."}]}]}"#,
        ])
        let provider = OpenAIProvider(apiKey: "test", model: "gpt-6-luna", reasoningEffort: .medium, transport: transport)
        let router = Router(provider: provider)
        let user = ConversationMessage.user("Help me choose a break")
        let question = await router.route(messages: [user])
        XCTAssertEqual(question.decision.summary, "I can help. Indoors?")
        let saved = try JSONEncoder().encode(XCTUnwrap(question.assistantMessage))
        let followUp = try JSONDecoder().decode(ConversationMessage.self, from: saved)
        let reply = await router.route(messages: [user, followUp, .user("Yes")])
        XCTAssertEqual(reply.decision.summary, "Try stretching indoors.")

        let requests = await transport.requests
        XCTAssertEqual(requests.map { $0.url?.path }, ["/v1/responses", "/v1/responses"])
        let body = try JSONValue.parse(XCTUnwrap(requests.last?.httpBody))
        XCTAssertEqual(body["store"], .bool(false))
        XCTAssertEqual(body["reasoning"]?["effort"], .string("medium"))
        XCTAssertEqual(body["max_output_tokens"], .number(8_192))
        guard case .array(let input)? = body["input"],
              case .array(let original)? = try JSONValue.parse(Data(initial.utf8))["output"] else {
            return XCTFail("Missing tool continuation")
        }
        XCTAssertEqual(Array(input.dropFirst().dropLast()), original)
        XCTAssertEqual(input.last?["call_id"], .string("call_123"))
        XCTAssertEqual(input.last?["output"], .string("Yes"))
        let alternate = OpenAIResponses.requestBody(model: "gpt-6.2-sol", system: "sys",
            messages: [.tool(call: try XCTUnwrap(question.toolCall), output: "Yes")], tools: [], effort: .low, maxTokens: 4096)
        guard case .array(let alternateInput)? = alternate["input"] else { return XCTFail("Missing portable history") }
        XCTAssertFalse(alternateInput.contains { $0["type"] == .string("reasoning") })
        XCTAssertEqual(alternateInput.last?["call_id"], .string("call_123"))
    }

    func testOpenAIAutomaticUsesResponsesWithoutReasoningOnLuna() async throws {
        let transport = StubTransport([
            #"{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]}"#,
        ])
        let result = await Router(provider: OpenAIProvider(apiKey: "test", model: "gpt-6-luna", transport: transport)).route("Hello")
        let requests = await transport.requests
        let request = try XCTUnwrap(requests.first)
        XCTAssertEqual(request.url?.path, "/v1/responses")
        let body = try JSONValue.parse(XCTUnwrap(request.httpBody))
        XCTAssertEqual(body["reasoning"]?["effort"], .string("none"))
        XCTAssertEqual(result.decision.action, .error)
        XCTAssertTrue(result.decision.summary.contains("output tokens"))
    }

    func testOpenRouterListsOnlyTextModelsThatSupportTools() async throws {
        let transport = StubTransport([#"""
        {"data":[
          {"id":"anthropic/claude-sonnet","name":"Claude Sonnet","supported_parameters":["tools","reasoning"],
           "reasoning":{"supported_efforts":["low","high","none"],"mandatory":true}},
          {"id":"example/chat","name":"Text only","supported_parameters":["temperature"]},
          {"id":"example/audio","supported_parameters":["tools"],"architecture":{"output_modalities":["audio"]}},
          {"id":"example/transcription","supported_parameters":["tools"],"architecture":{"input_modalities":["audio"]}}
        ]}
        """#])
        let provider = OpenRouterProvider(apiKey: "router-test-key", model: "", transport: transport)
        let models = try await provider.listModels()
        XCTAssertEqual(models, [ProviderModel(id: "anthropic/claude-sonnet", name: "Claude Sonnet", reasoningEfforts: [.low, .high])])
        let requests = await transport.requests
        let request = try XCTUnwrap(requests.first)
        XCTAssertEqual(request.url?.absoluteString, "https://openrouter.ai/api/v1/models/user?output_modalities=text")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer router-test-key")
    }

    func testOpenRouterFollowUpUsesSharedChatWireAndMatchingToolID() async throws {
        let transport = StubTransport([
            #"{"choices":[{"message":{"content":"I can help.","reasoning_details":[{"type":"reasoning.encrypted","data":"opaque"}],"tool_calls":[{"id":"call_123","type":"function","function":{"name":"clarify","arguments":"{\"question\":\"Indoors?\"}"}}]}}]}"#,
            #"{"choices":[{"message":{"content":"Try stretching indoors."}}]}"#,
        ])
        let provider = OpenRouterProvider(apiKey: "router-test-key", model: "anthropic/claude-sonnet", reasoningEffort: .high, transport: transport)
        let router = Router(provider: provider)
        let user = ConversationMessage.user("Help me choose a break")
        let question = await router.route(messages: [user])
        XCTAssertTrue(question.decision.listensAfterSpeaking)
        XCTAssertEqual(question.decision.summary, "I can help. Indoors?")
        let followUp = try XCTUnwrap(question.assistantMessage)
        let reply = await router.route(messages: [user, followUp, .user("Yes")])
        XCTAssertEqual(reply.decision.summary, "Try stretching indoors.")
        XCTAssertEqual(reply.provider, "openrouter")
        XCTAssertNil(reply.toolCall)

        let requests = await transport.requests
        XCTAssertEqual(requests.count, 2)
        let request = requests[1]
        XCTAssertEqual(request.url?.absoluteString, "https://openrouter.ai/api/v1/chat/completions")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer router-test-key")
        let body = try JSONValue.parse(XCTUnwrap(request.httpBody))
        XCTAssertEqual(body["model"], .string("anthropic/claude-sonnet"))
        XCTAssertEqual(body["provider"]?["require_parameters"], .bool(true))
        XCTAssertNil(body["reasoning_effort"])
        XCTAssertEqual(body["reasoning"]?["effort"], .string("high"))
        XCTAssertNil(body["max_completion_tokens"])
        guard case .array(let messages)? = body["messages"] else { return XCTFail("Missing messages") }
        XCTAssertEqual(messages.last?["tool_call_id"], .string("call_123"))
        XCTAssertEqual(messages.last?["content"], .string("Yes"))
        let assistant = messages[messages.count - 2]
        XCTAssertEqual(assistant["reasoning_details"], .array([.object(["type": .string("reasoning.encrypted"), "data": .string("opaque")])]))
    }

    func testOpenRouterSurfacesEmbeddedGenerationFailure() async throws {
        let transport = StubTransport([
            #"{"choices":[],"error":{"code":402,"message":"Insufficient credits"}}"#,
            #"{"choices":[{"message":{"content":""},"error":{"code":502,"message":"Upstream failed"}}]}"#,
        ])
        let router = Router(provider: OpenRouterProvider(apiKey: "test", model: "example/text", transport: transport))
        let credits = await router.route("Hello")
        XCTAssertEqual(credits.decision.action, .error)
        XCTAssertEqual(credits.decision.summary, "Your provider requires credits for this request.")
        let upstream = await router.route("Hello")
        XCTAssertEqual(upstream.decision.action, .error)
    }

    func testAnthropicCatalogKeepsPagingWithNamesAndNativeHeaders() async throws {
        let transport = StubTransport([
            #"{"data":[{"id":"claude-sonnet","display_name":"Claude Sonnet"}],"has_more":true,"last_id":"claude-sonnet"}"#,
            #"{"data":[{"id":"claude-haiku","display_name":"Claude Haiku"}],"has_more":false,"last_id":"claude-haiku"}"#,
        ])
        let provider = AnthropicProvider(apiKey: "claude-test-key", model: "", transport: transport)
        let models = try await provider.listModels()
        XCTAssertEqual(models, [
            ProviderModel(id: "claude-sonnet", name: "Claude Sonnet"),
            ProviderModel(id: "claude-haiku", name: "Claude Haiku"),
        ])
        let requests = await transport.requests
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests[0].url?.absoluteString, "https://api.anthropic.com/v1/models?limit=100")
        XCTAssertEqual(requests[1].url?.absoluteString, "https://api.anthropic.com/v1/models?limit=100&after_id=claude-sonnet")
        XCTAssertEqual(requests[1].value(forHTTPHeaderField: "x-api-key"), "claude-test-key")
        XCTAssertEqual(requests[1].value(forHTTPHeaderField: "anthropic-version"), "2023-06-01")
    }

    func testAnthropicRepliesToFollowUpUsingNativeToolResult() async throws {
        let transport = StubTransport([
            #"{"content":[{"type":"thinking","thinking":"","signature":"opaque"},{"type":"text","text":"I can help."},{"type":"tool_use","id":"toolu_123","name":"clarify","input":{"question":"Indoors?"}}]}"#,
            #"{"content":[{"type":"text","text":"Try stretching indoors."}]}"#,
        ])
        let router = Router(provider: AnthropicProvider(apiKey: "claude-test-key", model: "claude-haiku-5-5",
                                                        reasoningEffort: .low, transport: transport))
        let user = ConversationMessage.user("Help me choose a break")
        let question = await router.route(messages: [user])
        XCTAssertEqual(question.decision.summary, "I can help. Indoors?")
        let followUp = try XCTUnwrap(question.assistantMessage)
        let reply = await router.route(messages: [user, followUp, .user("Yes")])
        XCTAssertEqual(reply.decision.summary, "Try stretching indoors.")
        XCTAssertEqual(reply.provider, "anthropic")
        XCTAssertNil(reply.toolCall)

        let requests = await transport.requests
        XCTAssertEqual(requests.count, 2)
        let request = requests[1]
        XCTAssertEqual(request.url?.absoluteString, "https://api.anthropic.com/v1/messages")
        XCTAssertEqual(request.value(forHTTPHeaderField: "x-api-key"), "claude-test-key")
        XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"))
        let body = try JSONValue.parse(XCTUnwrap(request.httpBody))
        XCTAssertEqual(body["thinking"], .object(["type": .string("adaptive"), "display": .string("omitted")]))
        XCTAssertEqual(body["output_config"]?["effort"], .string("low"))
        guard case .array(let messages)? = body["messages"],
              case .array(let results)? = messages.last?["content"] else { return XCTFail("Missing tool result") }
        XCTAssertEqual(results.first?["tool_use_id"], .string("toolu_123"))
        XCTAssertEqual(results.first?["content"], .string("Yes"))
        guard case .array(let original)? = messages[messages.count - 2]["content"] else { return XCTFail("Missing assistant turn") }
        XCTAssertEqual(original.first, .object(["type": .string("thinking"), "thinking": .string(""), "signature": .string("opaque")]))
    }
}

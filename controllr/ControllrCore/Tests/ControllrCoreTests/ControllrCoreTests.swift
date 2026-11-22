import XCTest
@testable import ControllrCore

struct MockProvider: LLMProvider {
    let name = "mock"; let model = "mock-1"
    let result: Result<AssistantCompletion, ProviderError>
    func complete(system: String, messages: [ConversationMessage], tools: [ToolSpec]) async throws -> AssistantCompletion { try result.get() }
    func listModels() async throws -> [ProviderModel] { [] }
}

struct MockExecutor: ModuleExecutor {
    var outcome: Result<ExecutionOutcome, ExecutionError> = .success(.spoken("Done."))
    var prepareError: ExecutionError? = nil
    func prepare(_ module: Module, arguments: [String: JSONValue]) async throws -> [String: JSONValue] {
        if let e = prepareError { throw e }
        var a = arguments; if module.id == "send_message" { a["recipient"] = .string("+15550100") }; return a
    }
    func execute(_ module: Module, arguments: [String: JSONValue]) async throws -> ExecutionOutcome { try outcome.get() }
}

func call(_ name: String, _ args: [String: JSONValue]) -> Result<AssistantCompletion, ProviderError> {
    .success(AssistantCompletion(toolCall: ToolCall(name: name, arguments: .object(args))))
}

final class ControllrCoreTests: XCTestCase {
    func router(_ r: Result<AssistantCompletion, ProviderError>, executor: ModuleExecutor? = nil) -> Router {
        Router(provider: MockProvider(result: r), executor: executor)
    }

    // schema -> tool JSON
    func testSchemaHasRequiredAndEnum() throws {
        let m = try XCTUnwrap(Registry.v1.module(id: "media_control"))
        let s = m.parametersSchema
        XCTAssertEqual(s["required"], .array([.string("command")]))
        XCTAssertEqual(s["properties"]?["command"]?["enum"], .array(["play", "pause", "next", "previous"].map { .string($0) }))
        XCTAssertEqual(s["additionalProperties"], .bool(false))
    }

    func testOnlyExecutableModulesBecomeTools() {
        let tools = router(.success(AssistantCompletion(text: "Hello"))).tools
        XCTAssertEqual(tools.count, Registry.v1.modules.filter { $0.kind != .response }.count)
        XCTAssertEqual(Set(tools.map(\.name)).count, tools.count)
        XCTAssertFalse(tools.contains { $0.name == "answer" })
        XCTAssertTrue(tools.contains { $0.name == "clarify" })
    }

    // tool call -> Decision
    func testShortcutModuleBecomesRunDecision() async throws {
        let r = await router(call("media_control", ["command": .string("next")])).route("skip this song")
        XCTAssertEqual(r.decision.action, .run)
        XCTAssertEqual(r.decision.shortcut, "media_control")
        XCTAssertEqual(r.decision.input["command"], .string("next"))
        XCTAssertFalse(r.decision.confirm)
    }

    func testAnswerRespondsAndClarifyClarifies() async {
        let a = await router(.success(AssistantCompletion(text: "It's 11.52."))).route("18% of 64")
        XCTAssertEqual(a.decision.action, .respond)
        XCTAssertEqual(a.decision.shortcut, "speak")
        XCTAssertEqual(a.decision.input["speak"], .string("It's 11.52."))
        XCTAssertFalse(a.decision.listensAfterSpeaking)
        let c = await router(call("clarify", ["question": .string("Who?")])).route("text them")
        XCTAssertEqual(c.decision.action, .clarify)
        XCTAssertTrue(c.decision.listensAfterSpeaking)
    }

    func testAssistantTextAndToolCallSpeakAndListen() async throws {
        let call = ToolCall(name: "clarify", arguments: .object(["question": .string("Will you be indoors?")]), id: "follow_1")
        let result = await router(.success(AssistantCompletion(text: "A quiet walk could help.", toolCall: call)))
            .route("I prefer quiet.")
        let decision = result.decision
        XCTAssertEqual(decision.action, .clarify)
        XCTAssertEqual(decision.summary, "A quiet walk could help. Will you be indoors?")
        XCTAssertEqual(decision.input["speak"], .string(decision.summary))
        XCTAssertEqual(result.assistantMessage, .followUp(call: call, assistantText: "A quiet walk could help.", spoken: decision.summary))
        let payload = try JSONValue.parse(Data(decision.json.utf8))
        XCTAssertEqual(payload["voice_next"], .string("listen"))
        XCTAssertEqual(payload["action"], .string("clarify"))
    }

    func testDirectSpokenQuestionKeepsListeningWithoutToolCall() async throws {
        let result = await router(.success(AssistantCompletion(text: "Quiet works. Are you indoors or outdoors? ")))
            .route("I prefer quiet.")
        XCTAssertNil(result.toolCall)
        XCTAssertEqual(result.moduleName, "clarify")
        XCTAssertEqual(try JSONValue.parse(Data(result.decision.json.utf8))["voice_next"], .string("listen"))

        let example = await router(.success(AssistantCompletion(text: #"Try asking “Are you indoors?”"#)))
            .route("Give me an example question.")
        XCTAssertFalse(example.decision.listensAfterSpeaking)

        let empty = await router(.success(AssistantCompletion())).route("Thanks.")
        XCTAssertEqual(try JSONValue.parse(Data(empty.decision.json.utf8))["voice_next"], .string("stop"))
    }

    // confirmation comes from the registry, never the model
    func testConfirmComesFromRegistry() async {
        let r = await router(call("send_message", ["recipient": .string("Sam"), "body": .string("late")])).route("text sam")
        XCTAssertTrue(r.decision.confirm)
        XCTAssertEqual(r.decision.summary, "Text Sam: late")
    }

    func testModulePolicyHidesDisabledToolAndRejectsStaleCalls() async {
        let policy = ModulePolicy(preferences: ["flashlight": .init(enabled: false)])
        let ex = MockExecutor(prepareError: .failed("should never prepare"))
        let router = Router(provider: MockProvider(result: call("flashlight", ["on": .bool(true)])),
                            executor: ex, policy: policy)
        XCTAssertFalse(router.tools.contains { $0.name == "flashlight" })
        let result = await router.route("Turn on the flashlight")
        XCTAssertEqual(result.decision.summary, "That action is turned off in Modules.")
    }

    func testModulePolicyOverridesShortcutAndStagesNativeWrites() async throws {
        let policy = ModulePolicy(preferences: [
            "send_message": .init(confirmation: .runDirectly),
            "add_reminder": .init(confirmation: .askFirst)
        ])
        let message = Router(provider: MockProvider(result: call("send_message", ["recipient": .string("Sam"), "body": .string("Hi")])),
                             executor: MockExecutor(), policy: policy)
        let direct = await message.route("Text Sam hi")
        XCTAssertFalse(direct.decision.confirm)
        XCTAssertEqual(try JSONValue.parse(Data(direct.decision.json.utf8))["confirmation"], .string("run"))

        let reminder = Router(provider: MockProvider(result: call("add_reminder", ["title": .string("Buy milk")])),
                              executor: MockExecutor(outcome: .failure(.failed("should never execute"))), policy: policy)
        let proposed = await reminder.route("Remind me to buy milk")
        XCTAssertEqual(proposed.decision.shortcut, Decision.nativeActionID)
        XCTAssertTrue(proposed.decision.confirm)
    }

    func testMissingMessageBodyAsksForItWithoutPreparingAction() async {
        let ex = MockExecutor(prepareError: .failed("unexpected prepare"))
        let result = await router(call("send_message", ["recipient": .string("Sam")]), executor: ex).route("text Sam")
        XCTAssertEqual(result.decision.action, .clarify)
        XCTAssertEqual(result.decision.summary, "What should I say?")
    }

    func testModelCannotSkipConfirmation() async {
        let r = await router(call("send_message", ["recipient": .string("Sam"), "body": .string("hi"), "confirm": .bool(false)])).route("x")
        XCTAssertEqual(r.decision.action, .error)   // unexpected argument rejected
    }

    // invalid arguments -> error
    func testInvalidArgumentsBecomeError() async {
        for bad: [String: JSONValue] in [[:], ["command": .string("rewind")], ["command": .number(1)]] {
            let r = await router(call("media_control", bad)).route("x")
            XCTAssertEqual(r.decision.action, .error, "\(bad)")
            XCTAssertNotNil(r.error)
        }
    }

    func testUnknownToolBecomesError() async {
        let r = await router(call("launch_missiles", [:])).route("x")
        XCTAssertEqual(r.decision.action, .error)
    }

    func testProviderFailuresBecomeSpokenErrors() async {
        let r = await router(.failure(.http(status: 401, body: ""))).route("x")
        XCTAssertEqual(r.decision.action, .error)
        XCTAssertEqual(r.decision.input["speak"], .string("Your API key was rejected."))
    }

    // native modules
    func testNativeDryRunDoesNotExecute() async {
        let r = await router(call("add_reminder", ["title": .string("milk")]), executor: MockExecutor()).route("x", execute: false)
        XCTAssertEqual(r.decision.action, .done)
        XCTAssertTrue(r.decision.summary.hasPrefix("Would run"))
    }

    func testNativeExecutes() async {
        let r = await router(call("add_reminder", ["title": .string("milk")]), executor: MockExecutor(outcome: .success(.spoken("Added.")))).route("x")
        XCTAssertEqual(r.decision.input["speak"], .string("Added."))
    }

    func testNativeClarifyFromExecutor() async {
        let ex = MockExecutor(outcome: .failure(.clarify("Which Sam?")))
        let r = await router(call("call_contact", ["recipient": .string("Sam")]), executor: ex).route("x")
        XCTAssertEqual(r.decision.action, .clarify)
    }

    func testNativeURLBecomesOpenURLDecision() async {
        let ex = MockExecutor(outcome: .success(.openURL("tel:+15550100", summary: "Call Sam")))
        let r = await router(call("call_contact", ["recipient": .string("Sam")]), executor: ex).route("call sam")
        XCTAssertEqual(r.decision.action, .run)
        XCTAssertEqual(r.decision.shortcut, "open_url_confirm")   // call needs confirmation (from registry)
        XCTAssertTrue(r.decision.confirm)
        XCTAssertEqual(r.decision.input["url"], .string("tel:+15550100"))
        let m = await router(call("directions", ["destination": .string("x")]),
                             executor: MockExecutor(outcome: .success(.openURL("https://maps.apple.com/?daddr=x", summary: "x")))).route("x")
        XCTAssertEqual(m.decision.shortcut, "open_url")
        XCTAssertFalse(m.decision.confirm)
    }

    func testPrepareResolvesShortcutArguments() async {
        let r = await router(call("send_message", ["recipient": .string("Sam"), "body": .string("hi")]), executor: MockExecutor()).route("x")
        XCTAssertEqual(r.decision.input["recipient"], .string("+15550100"))
    }

    func testPrepareClarifyStopsTheRun() async {
        let ex = MockExecutor(prepareError: .clarify("Which Sam?"))
        let r = await router(call("send_message", ["recipient": .string("Sam"), "body": .string("hi")]), executor: ex).route("x")
        XCTAssertEqual(r.decision.action, .clarify)
        XCTAssertEqual(r.decision.shortcut, "speak")
    }

    // decision JSON is flat and stable
    func testDecisionJSONShape() throws {
        let d = Decision(action: .run, shortcut: "flashlight", input: ["on": .bool(true)], confirm: false, summary: "Flashlight true")
        let v = try JSONValue.parse(Data(d.json.utf8))
        XCTAssertEqual(v["action"], .string("run"))
        XCTAssertEqual(v["shortcut"], .string("flashlight"))
        XCTAssertEqual(v["input"]?["on"], .bool(true))
        XCTAssertEqual(v["confirm"], .bool(false))
    }

    // providers
    func testOpenAIRequestAllowsOptionalToolCall() throws {
        let p = OpenAIProvider(apiKey: "k", model: "m")
        let body = p.requestBody(system: "s", messages: [.user("t")], tools: router(.success(AssistantCompletion())).tools)
        XCTAssertEqual(body["tool_choice"], .string("auto"))
        XCTAssertEqual(body["model"], .string("m"))
    }

    func testAnthropicRequestAllowsOptionalToolUse() throws {
        let p = AnthropicProvider(apiKey: "k", model: "m")
        let body = p.requestBody(system: "s", messages: [.user("t")], tools: router(.success(AssistantCompletion())).tools)
        XCTAssertEqual(body["tool_choice"]?["type"], .string("auto"))
        XCTAssertEqual(body["system"], .string("s"))
    }

    func testOpenAIParsesTextAlongsideToolCall() throws {
        let json = try JSONValue.parse(Data(#"{"choices":[{"message":{"content":"Good idea.","tool_calls":[{"id":"call_1","function":{"name":"clarify","arguments":"{\"question\":\"Indoors?\"}"}}]}}]}"#.utf8))
        let completion = try ChatCompletions.parseCompletion(json)
        XCTAssertEqual(completion.text, "Good idea.")
        XCTAssertEqual(completion.toolCall, ToolCall(name: "clarify", arguments: .object(["question": .string("Indoors?")]), id: "call_1"))
    }

    func testAnthropicParsesTextAlongsideToolCall() throws {
        let json = try JSONValue.parse(Data(#"{"content":[{"type":"text","text":"Good idea."},{"type":"tool_use","id":"toolu_1","name":"clarify","input":{"question":"Indoors?"}}]}"#.utf8))
        let completion = try AnthropicProvider.parseCompletion(json)
        XCTAssertEqual(completion.text, "Good idea.")
        XCTAssertEqual(completion.toolCall, ToolCall(name: "clarify", arguments: .object(["question": .string("Indoors?")]), id: "toolu_1"))
    }

    func testProvidersAcceptPlainAssistantText() throws {
        let openAI = try JSONValue.parse(Data(#"{"choices":[{"message":{"content":"Hello."}}]}"#.utf8))
        let anthropic = try JSONValue.parse(Data(#"{"content":[{"type":"text","text":"Hello."}]}"#.utf8))
        XCTAssertEqual(try ChatCompletions.parseCompletion(openAI), AssistantCompletion(text: "Hello."))
        XCTAssertEqual(try AnthropicProvider.parseCompletion(anthropic), AssistantCompletion(text: "Hello."))
    }

    func testPromptIncludesDateAndRules() {
        let p = Prompt.system(now: Date(timeIntervalSince1970: 0), timeZone: TimeZone(identifier: "America/New_York")!,
                              settings: RouterSettings(homeAddress: "x", nicknames: ["mom": "Linda"]))
        XCTAssertTrue(p.contains("1969-12-31"))
        XCTAssertTrue(p.contains("Prefer clarify"))
        XCTAssertTrue(p.contains("mom = Linda"))
    }
}

actor ScriptedProvider: LLMProvider {
    let name = "scripted"
    let model = "scripted-1"
    private var completions: [AssistantCompletion]
    private(set) var requests: [[ConversationMessage]] = []
    private(set) var systemPrompts: [String] = []

    init(_ completions: [AssistantCompletion]) { self.completions = completions }

    func complete(system: String, messages: [ConversationMessage], tools: [ToolSpec]) async throws -> AssistantCompletion {
        requests.append(messages)
        systemPrompts.append(system)
        guard !completions.isEmpty else { throw ProviderError.badResponse("unexpected request") }
        return completions.removeFirst()
    }

    func listModels() async throws -> [ProviderModel] { [] }
}

final class ConversationRoutingTests: XCTestCase {
    func testCustomInstructionsAreSentAndCapturedWithRuntimeContext() async {
        let provider = ScriptedProvider([AssistantCompletion(text: "Done.")])
        let settings = RouterSettings(homeAddress: "home", nicknames: ["mom": "Linda"],
                                      systemInstructions: "Answer in a single paragraph.")
        let now = Date(timeIntervalSince1970: 2_000_000_000)
        let result = await Router(provider: provider, settings: settings).route("Hello", now: now)
        let sent = await provider.systemPrompts

        XCTAssertEqual(sent, [Prompt.system(now: now, settings: settings)])
        XCTAssertEqual(result.systemPrompt, sent.first)
        XCTAssertTrue(result.systemPrompt?.contains("Answer in a single paragraph.") == true)
        XCTAssertTrue(result.systemPrompt?.contains("mom = Linda") == true)
        XCTAssertFalse(result.systemPrompt?.contains("Prefer clarify") == true)
    }

    func testFollowUpUsesUserAndAssistantRoles() async throws {
        let call = ToolCall(name: "send_message", arguments: .object([
            "recipient": .string("Sam"), "body": .string("Running late."),
        ]))
        let provider = ScriptedProvider([AssistantCompletion(toolCall: call)])
        let history: [ConversationMessage] = [
            .user("Text Sam"), .assistant("What should I say?"), .user("Running late."),
        ]
        let result = await Router(provider: provider).route(messages: history, execute: false)
        let requests = await provider.requests
        XCTAssertEqual(requests, [history])
        XCTAssertEqual(result.decision.shortcut, "send_message")
        XCTAssertTrue(result.decision.confirm)
        XCTAssertEqual(result.decision.input["body"], .string("Running late."))
    }

    func testTwoFollowUpToolsGetHumanRepliesAsResults() async throws {
        let first = ToolCall(name: "clarify", arguments: .object(["question": .string("Quiet or active?")]), id: "call_first")
        let second = ToolCall(name: "clarify", arguments: .object(["question": .string("Indoors or outdoors?")]), id: "call_second")
        let provider = ScriptedProvider([
            AssistantCompletion(toolCall: first),
            AssistantCompletion(text: "A quiet break sounds good.", toolCall: second),
            AssistantCompletion(text: "Try five minutes of stretching."),
        ])
        let router = Router(provider: provider)
        let initial = ConversationMessage.user("Ask me two questions before recommending a break.")
        let one = await router.route(messages: [initial])
        let pendingOne = try XCTUnwrap(one.assistantMessage)
        let two = await router.route(messages: [initial, pendingOne, .user("Quiet.")])
        let pendingTwo = try XCTUnwrap(two.assistantMessage)
        let three = await router.route(messages: [initial, pendingOne, .user("Quiet."), pendingTwo, .user("Indoors.")])
        let requests = await provider.requests
        XCTAssertEqual(requests.last, [
            initial,
            .tool(call: first, output: "Quiet."),
            .tool(call: second, output: "Indoors.", assistantText: "A quiet break sounds good."),
        ])
        XCTAssertTrue(one.decision.listensAfterSpeaking)
        XCTAssertTrue(two.decision.listensAfterSpeaking)
        XCTAssertFalse(three.decision.listensAfterSpeaking)
        XCTAssertEqual(three.decision.summary, "Try five minutes of stretching.")
    }

    func testStoredToolResultWithoutAssistantTextStillDecodes() throws {
        let data = Data(#"{"tool":{"call":{"name":"search_saved_items","arguments":{}},"output":"[]"}}"#.utf8)
        XCTAssertEqual(try JSONDecoder().decode(ConversationMessage.self, from: data),
                       .tool(call: ToolCall(name: "search_saved_items", arguments: .object([:])), output: "[]"))
    }

    func testSavedLookupContinuesTheSameTurn() async throws {
        let lookup = ToolCall(name: "search_saved_items", arguments: .object(["query": .string("gym")]), id: "call_gym")
        let provider = ScriptedProvider([AssistantCompletion(text: "I'll check.", toolCall: lookup),
                                         AssistantCompletion(text: "Nine.")])
        let message = ConversationMessage.user("When does my gym close?")
        let savedFacts = #"[{"kind":"memory","title":"Gym hours","text":"Closes at 9 PM"}]"#
        let router = Router(provider: provider, executor: MockExecutor(outcome: .success(.toolResult(savedFacts))))

        let result = await router.route(messages: [message])
        let exchange = ConversationMessage.tool(call: lookup, output: savedFacts, assistantText: "I'll check.")
        let requests = await provider.requests
        XCTAssertEqual(requests, [[message], [message, exchange]])
        XCTAssertEqual(result.toolMessages, [exchange])
        XCTAssertEqual(result.decision.action, .respond)
        XCTAssertEqual(result.decision.summary, "Nine.")
    }

    func testProvidersEncodeToolResultsWithMatchingIDs() throws {
        let lookup = ToolCall(name: "search_saved_items", arguments: .object([:]), id: "toolu_memory")
        let messages: [ConversationMessage] = [.user("What's saved?"), .tool(call: lookup, output: "[]", assistantText: "I'll check.")]
        let openAI = OpenAIProvider(apiKey: "dummy", model: "test")
        let anthropic = AnthropicProvider(apiKey: "dummy", model: "test")

        let openAIBody = openAI.requestBody(system: "s", messages: messages, tools: [])
        guard case .array(let openAIMessages) = openAIBody["messages"],
              case .array(let calls) = openAIMessages[2]["tool_calls"] else { return XCTFail("missing OpenAI tool call") }
        XCTAssertEqual(openAIMessages[2]["content"], .string("I'll check."))
        XCTAssertEqual(openAIMessages[3]["role"], .string("tool"))
        XCTAssertEqual(calls.first?["id"], openAIMessages[3]["tool_call_id"])

        let anthropicBody = anthropic.requestBody(system: "s", messages: messages, tools: [])
        guard case .array(let anthropicMessages) = anthropicBody["messages"],
              case .array(let uses) = anthropicMessages[1]["content"],
              case .array(let results) = anthropicMessages[2]["content"] else { return XCTFail("missing Anthropic tool exchange") }
        XCTAssertEqual(anthropicMessages[2]["role"], .string("user"))
        XCTAssertEqual(uses.first?["text"], .string("I'll check."))
        XCTAssertEqual(uses.last?["id"], results.first?["tool_use_id"])
    }
}

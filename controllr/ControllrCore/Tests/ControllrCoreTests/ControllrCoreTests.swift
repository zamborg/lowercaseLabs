import XCTest
@testable import ControllrCore

struct MockProvider: LLMProvider {
    let name = "mock"; let model = "mock-1"
    let result: Result<ToolCall, ProviderError>
    func route(system: String, transcript: String, tools: [ToolSpec]) async throws -> ToolCall { try result.get() }
    func listModels() async throws -> [String] { [] }
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

func call(_ name: String, _ args: [String: JSONValue]) -> Result<ToolCall, ProviderError> {
    .success(ToolCall(name: name, arguments: .object(args)))
}

final class ControllrCoreTests: XCTestCase {
    func router(_ r: Result<ToolCall, ProviderError>, executor: ModuleExecutor? = nil) -> Router {
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

    func testEveryModuleBecomesATool() {
        let tools = router(call("answer", ["text": .string("x")])).tools
        XCTAssertEqual(tools.count, Registry.v1.modules.count)
        XCTAssertEqual(Set(tools.map(\.name)).count, tools.count)
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
        let a = await router(call("answer", ["text": .string("It's 11.52.")])).route("18% of 64")
        XCTAssertEqual(a.decision.action, .respond)
        XCTAssertEqual(a.decision.shortcut, "speak")
        XCTAssertEqual(a.decision.input["speak"], .string("It's 11.52."))
        let c = await router(call("clarify", ["question": .string("Who?")])).route("text them")
        XCTAssertEqual(c.decision.action, .clarify)
    }

    // confirmation comes from the registry, never the model
    func testConfirmComesFromRegistry() async {
        let r = await router(call("send_message", ["recipient": .string("Sam"), "body": .string("late")])).route("text sam")
        XCTAssertTrue(r.decision.confirm)
        XCTAssertEqual(r.decision.summary, "Text Sam: late")
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
    func testOpenAIRequestForcesToolCall() throws {
        let p = OpenAIProvider(apiKey: "k", model: "m")
        let body = p.requestBody(system: "s", transcript: "t", tools: router(call("answer", [:])).tools)
        XCTAssertEqual(body["tool_choice"], .string("required"))
        XCTAssertEqual(body["model"], .string("m"))
    }

    func testAnthropicRequestForcesToolUse() throws {
        let p = AnthropicProvider(apiKey: "k", model: "m")
        let body = p.requestBody(system: "s", transcript: "t", tools: router(call("answer", [:])).tools)
        XCTAssertEqual(body["tool_choice"]?["type"], .string("any"))
        XCTAssertEqual(body["system"], .string("s"))
    }

    func testOpenAIParse() throws {
        let json = try JSONValue.parse(Data(#"{"choices":[{"message":{"tool_calls":[{"function":{"name":"answer","arguments":"{\"text\":\"hi\"}"}}]}}]}"#.utf8))
        XCTAssertEqual(try OpenAIProvider.parseToolCall(json), ToolCall(name: "answer", arguments: .object(["text": .string("hi")])))
    }

    func testAnthropicParse() throws {
        let json = try JSONValue.parse(Data(#"{"content":[{"type":"text","text":"ok"},{"type":"tool_use","name":"answer","input":{"text":"hi"}}]}"#.utf8))
        XCTAssertEqual(try AnthropicProvider.parseToolCall(json), ToolCall(name: "answer", arguments: .object(["text": .string("hi")])))
    }

    func testProviderWithoutToolCallThrows() throws {
        let json = try JSONValue.parse(Data(#"{"choices":[{"message":{"content":"hello"}}]}"#.utf8))
        XCTAssertThrowsError(try OpenAIProvider.parseToolCall(json))
    }

    func testPromptIncludesDateAndRules() {
        let p = Prompt.system(now: Date(timeIntervalSince1970: 0), timeZone: TimeZone(identifier: "America/New_York")!,
                              settings: RouterSettings(homeAddress: "x", nicknames: ["mom": "Linda"]))
        XCTAssertTrue(p.contains("1969-12-31"))
        XCTAssertTrue(p.contains("Prefer clarify"))
        XCTAssertTrue(p.contains("mom = Linda"))
    }
}

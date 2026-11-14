import XCTest
@testable import ParlorCore

/// A scripted OpenAI-style server: SSE chat completions with text, tool calls and usage.
final class FakeOpenAI: @unchecked Sendable {
    let server: LoopbackServer
    private let lock = NSLock()
    private(set) var requests: [(path: String, auth: String?, body: [String: Any])] = []
    var port: Int { Int(server.port) }

    init() {
        let box = Box()
        server = LoopbackServer { req in
            let body = (try? JSONSerialization.jsonObject(with: req.body) as? [String: Any]) ?? [:]
            box.fake?.record(req.path, req.headers["authorization"], body)
            if req.method == "GET" { return .init(status: 200, body: Data(#"{"object":"list","data":[{"id":"fake-a"},{"id":"fake-b"}]}"#.utf8)) }
            if (body["model"] as? String) == "broken" { return .init(status: 500, body: Data(#"{"error":{"message":"model exploded","type":"server_error"}}"#.utf8)) }
            let msgs = body["messages"] as? [[String: Any]] ?? []
            let last = msgs.last
            let wantsTool = (last?["role"] as? String) == "user" && ((last?["content"] as? String) ?? "").contains("tool")
            let slow = ((last?["content"] as? String) ?? "").contains("slow")
            return .init(streaming: "text/event-stream") { w in
                func chunk(_ delta: [String: Any], finish: String? = nil) async -> Bool {
                    let o: [String: Any] = ["choices": [["index": 0, "delta": delta, "finish_reason": finish as Any? ?? NSNull()]]]
                    return await w.write(Data(("data: " + String(data: try! JSONSerialization.data(withJSONObject: o), encoding: .utf8)! + "\n\n").utf8))
                }
                if wantsTool {
                    _ = await chunk(["role": "assistant", "content": ""])
                    _ = await chunk(["tool_calls": [["index": 0, "id": "call_1", "type": "function", "function": ["name": "echo__say", "arguments": ""]]]])
                    _ = await chunk(["tool_calls": [["index": 0, "function": ["arguments": "{\"text\":"]]]])
                    _ = await chunk(["tool_calls": [["index": 0, "function": ["arguments": "\"hi\"}"]]]])
                    _ = await chunk([:], finish: "tool_calls")
                } else if slow {
                    for i in 0..<200 {
                        if !(await chunk(["content": "w\(i) "])) { box.fake?.noteDisconnect(); return }
                        try? await Task.sleep(nanoseconds: 20_000_000)
                    }
                } else {
                    for t in ["Hello", " from", " fake"] { _ = await chunk(["content": t]) }
                    _ = await chunk([:], finish: "stop")
                }
                let usage: [String: Any] = ["choices": [], "usage": ["prompt_tokens": 11, "completion_tokens": 3, "prompt_tokens_details": ["cached_tokens": 4]]]
                _ = await w.write(Data(("data: " + String(data: try! JSONSerialization.data(withJSONObject: usage), encoding: .utf8)! + "\n\ndata: [DONE]\n\n").utf8))
            }
        }
        box.fake = self
    }
    final class Box: @unchecked Sendable { weak var fake: FakeOpenAI? }
    private(set) var disconnected = false
    func noteDisconnect() { lock.lock(); disconnected = true; lock.unlock() }
    func record(_ p: String, _ a: String?, _ b: [String: Any]) { lock.lock(); requests.append((p, a, b)); lock.unlock() }
    func start() async throws { try await server.start(port: 0) }
}

/// A runtime that fails the way an unloadable model does.
struct FailingRuntime: LanguageModelRuntime {
    func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
        AsyncThrowingStream { c in c.finish(throwing: model.contains("missing") ? ParlorError.modelNotInstalled(model) : ParlorError.message("not enough memory to load \(model)")) }
    }
    func loadedModel() async -> String? { nil }
    func unload() async {}
}

struct EchoProvider: ToolProvider {
    let id = "echo"
    func listTools() async throws -> [ToolDef] {
        [ToolDef(server: "echo", name: "say", description: "Echo text.", schema: .object(["type": .string("object"), "properties": .object(["text": .object(["type": .string("string")])])]))]
    }
    func call(tool: String, arguments: JSON) async throws -> ToolOutput { ToolOutput(text: "echo: " + (arguments.objectValue?["text"]?.stringValue ?? "")) }
}

final class ProviderTests: XCTestCase {
    func collect(_ s: AsyncThrowingStream<LLMEvent, Error>) async throws -> (text: String, calls: [ToolCallRecord], stats: LLMStats?) {
        var text = "", calls: [ToolCallRecord] = [], stats: LLMStats?
        for try await e in s {
            switch e { case .text(let t): text += t; case .toolCall(let c): calls.append(c); case .stats(let s): stats = s }
        }
        return (text, calls, stats)
    }

    func router(_ fake: FakeOpenAI, key: String? = nil, locality: ProviderConfig.Locality = .local, home: ParlorHome = tempHome()) -> ProviderRouter {
        let r = ProviderRouter(inner: DemoRuntime())
        r.configure([ProviderConfig(id: "fake", baseURL: "http://127.0.0.1:\(fake.port)/v1", keyRef: key, locality: locality, enabled: true)], home: home)
        return r
    }

    func testProviderStreamsTextToolCallsAndUsage() async throws {
        let fake = FakeOpenAI(); try await fake.start()
        setenv("PARLOR_KEY_FAKEKEY", "sk-test-123", 1)
        let r = router(fake, key: "fakekey")
        let req = LLMRequest(messages: [LLMMessage(role: .system, text: "sys"), LLMMessage(role: .user, text: "say hi")], temperature: 0.2, maxTokens: 50)
        let a = try await collect(r.generate(model: "fake/model-x", request: req))
        XCTAssertEqual(a.text, "Hello from fake")
        XCTAssertEqual(a.stats?.promptTokens, 11)
        XCTAssertEqual(a.stats?.completionTokens, 3)
        XCTAssertEqual(a.stats?.cachedTokens, 4)
        let sent = try XCTUnwrap(fake.requests.first)
        XCTAssertEqual(sent.path, "/v1/chat/completions")
        XCTAssertEqual(sent.auth, "Bearer sk-test-123")
        XCTAssertEqual(sent.body["model"] as? String, "model-x", "the provider prefix is stripped")
        XCTAssertEqual(sent.body["stream"] as? Bool, true)
        XCTAssertEqual(sent.body["max_tokens"] as? Int, 50)

        // Tools out, tool calls back with arguments assembled from fragments, then a tool result goes in.
        let tool = ToolDef(server: "echo", name: "say", description: "Echo", schema: .object(["type": .string("object")]))
        let b = try await collect(r.generate(model: "fake/model-x", request: LLMRequest(messages: [LLMMessage(role: .user, text: "use the tool")], tools: [tool])))
        let call = try XCTUnwrap(b.calls.first)
        XCTAssertEqual(call.name, "echo__say")
        XCTAssertEqual(call.id, "call_1")
        XCTAssertEqual(call.arguments.objectValue?["text"]?.stringValue, "hi")
        let toolSpec = ((fake.requests.last?.body["tools"] as? [[String: Any]])?.first?["function"] as? [String: Any])
        XCTAssertEqual(toolSpec?["name"] as? String, "echo__say")
        let followUp = LLMRequest(messages: [
            LLMMessage(role: .user, text: "use the tool"),
            LLMMessage(role: .assistant, text: "", toolCalls: [call]),
            LLMMessage(role: .tool, text: "echo: hi", toolCallID: call.id, toolName: call.name)], tools: [tool])
        _ = try await collect(r.generate(model: "fake/model-x", request: followUp))
        let wire = try XCTUnwrap(fake.requests.last?.body["messages"] as? [[String: Any]])
        XCTAssertEqual(wire.map { $0["role"] as? String }, ["user", "assistant", "tool"])
        XCTAssertEqual(((wire[1]["tool_calls"] as? [[String: Any]])?.first?["id"] as? String), "call_1")
        XCTAssertEqual(wire[2]["tool_call_id"] as? String, "call_1")
        XCTAssertNotNil((wire[1]["tool_calls"] as? [[String: Any]])?.first?["function"])
    }

    func testProviderErrorsOffSwitchAndCancellation() async throws {
        let fake = FakeOpenAI(); try await fake.start()
        let r = router(fake)
        do {
            _ = try await collect(r.generate(model: "fake/broken", request: LLMRequest(messages: [LLMMessage(role: .user, text: "x")])))
            XCTFail("expected an error")
        } catch { XCTAssertTrue(error.localizedDescription.contains("500") && error.localizedDescription.contains("model exploded"), error.localizedDescription) }

        // MLX names fall through to the inner runtime, with or without the mlx/ prefix.
        let a = try await collect(r.generate(model: "mlx-community/Some-Model", request: LLMRequest(messages: [LLMMessage(role: .user, text: "yo")])))
        XCTAssertTrue(a.text.contains("(demo) You said: yo"))
        let b = try await collect(r.generate(model: "mlx/mlx-community/Some-Model", request: LLMRequest(messages: [LLMMessage(role: .user, text: "yo")])))
        XCTAssertTrue(b.text.contains("(demo)"))

        // A remote provider starts off.
        let remote = ProviderConfig(id: "cloud", baseURL: "https://example.invalid/v1", locality: .remote)
        XCTAssertFalse(remote.enabled)
        r.configure([remote], home: tempHome())
        do { _ = try await collect(r.generate(model: "cloud/x", request: LLMRequest(messages: [LLMMessage(role: .user, text: "x")]))); XCTFail() }
        catch { XCTAssertTrue(error.localizedDescription.contains("is off")) }

        // Cancelling stops the server's stream promptly.
        r.configure([ProviderConfig(id: "fake", baseURL: "http://127.0.0.1:\(fake.port)/v1")], home: tempHome())
        let task = Task { try await self.collect(r.generate(model: "fake/m", request: LLMRequest(messages: [LLMMessage(role: .user, text: "slow please")]))) }
        try await Task.sleep(nanoseconds: 400_000_000)
        task.cancel()
        _ = try? await task.value
        let deadline = Date().addingTimeInterval(5)
        while !fake.disconnected, Date() < deadline { try await Task.sleep(nanoseconds: 100_000_000) }
        XCTAssertTrue(fake.disconnected, "cancelling the request closes the connection")
    }

    func testProviderConfigParsingAndProblems() throws {
        let json = #"{"providers":[{"id":"llama","baseURL":"http://127.0.0.1:8081/v1","models":["qwen"],"service":{"command":"llama-server -m x.gguf --port 8081"}},{"id":"cloud","baseURL":"https://api.example.com/v1","locality":"remote","keyRef":"cloud-key"},{"id":"mlx","baseURL":"http://x/v1"}]}"#
        let s = try JSONDecoder().decode(ParlorSettings.self, from: Data(json.utf8))
        XCTAssertEqual(s.providers.count, 3)
        XCTAssertTrue(s.providers[0].enabled, "local providers start on")
        XCTAssertEqual(s.providers[0].service?.idleMinutes, 30)
        XCTAssertFalse(s.providers[1].enabled, "remote providers start off")
        XCTAssertNotNil(s.providers[2].problem, "mlx is reserved")
        XCTAssertEqual(s.runtimeAPI.enabled, false)
        XCTAssertEqual(s.runtimeAPI.port, 11500)
        XCTAssertEqual(KeyStore.envName("my-key"), "PARLOR_KEY_MY_KEY")
    }

    /// A provider with a `service` is launched on the first request and stopped when idle; the log has its output.
    func testProviderServiceStartsOnDemandAndStopsIdle() async throws {
        let home = tempHome(); try home.ensure()
        let script = home.root.appendingPathComponent("fake_server.py")
        try """
        import sys, json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        class H(BaseHTTPRequestHandler):
            def log_message(self, f, *a): sys.stderr.write((f % a) + "\\n"); sys.stderr.flush()
            def do_GET(self):
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "svc-model"}]}).encode())
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                for t in ["from ", "service"]:
                    self.wfile.write(("data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": t}}]}) + "\\n\\n").encode()); self.wfile.flush()
                self.wfile.write(b"data: [DONE]\\n\\n")
        print("fake server up", file=sys.stderr, flush=True)
        ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
        """.write(to: script, atomically: true, encoding: .utf8)
        // Find a free port.
        let probe = LoopbackServer { _ in .init(status: 200) }
        try await probe.start(port: 0); let port = Int(probe.port); probe.stop()

        let r = ProviderRouter(inner: DemoRuntime())
        r.configure([ProviderConfig(id: "svc", baseURL: "http://127.0.0.1:\(port)/v1", service: .init(command: "python3 '\(script.path)' \(port)", idleMinutes: 1, startupSeconds: 30))], home: home)
        let p = try XCTUnwrap(r.provider("svc"))
        let proc = try XCTUnwrap(p.process)
        XCTAssertEqual(proc.snapshot().state, .stopped, "not started until something asks")
        proc.idleSecondsOverride = 1
        let out = try await collect(r.generate(model: "svc/svc-model", request: LLMRequest(messages: [LLMMessage(role: .user, text: "hi")])))
        XCTAssertEqual(out.text, "from service")
        XCTAssertEqual(proc.snapshot().state, .running)
        XCTAssertNotNil(proc.snapshot().pid)
        XCTAssertTrue(ServiceLog.tail(home: home, service: "provider:svc", lines: 5).contains("fake server up"), "stderr goes to logs/provider-svc.log")
        let models = try await p.models()
        XCTAssertEqual(models.map(\.qualified), ["svc/svc-model"])
        let deadline = Date().addingTimeInterval(15)
        while proc.snapshot().state != .stopped, Date() < deadline { try await Task.sleep(nanoseconds: 300_000_000) }
        XCTAssertEqual(proc.snapshot().state, .stopped, "stops when idle")
        // And it comes back on the next request.
        let again = try await collect(r.generate(model: "svc/svc-model", request: LLMRequest(messages: [LLMMessage(role: .user, text: "hi")])))
        XCTAssertEqual(again.text, "from service")
        r.stopAll()
    }

    // MARK: runtime API

    @MainActor
    private func engine(_ llm: LanguageModelRuntime = DemoRuntime(), providers: String = "") async throws -> (ParlorEngine, String) {
        let home = tempHome(); try home.ensure()
        try #"{"parlorServer":{"enabled":true,"port":0},"runtimeAPI":{"enabled":true,"port":0}\#(providers)}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: llm)
        await e.start()
        let secret = try XCTUnwrap(e.createToken(name: "api-test", tier: .none))
        return (e, secret)
    }

    @MainActor
    private func http(_ e: ParlorEngine, _ method: String, _ path: String, token: String?, body: [String: Any]? = nil) async throws -> (Int, Data) {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(e.runtimeAPI.port)\(path)")!)
        req.httpMethod = method
        if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        if let body { req.httpBody = try JSONSerialization.data(withJSONObject: body) }
        let (d, r) = try await URLSession.shared.data(for: req)
        return ((r as! HTTPURLResponse).statusCode, d)
    }

    @MainActor
    func testRuntimeAPISchema() async throws {
        let (e, secret) = try await engine(providers: #","providers":[{"id":"fake","baseURL":"http://127.0.0.1:1/v1","models":["fake-a"]}]"#)
        XCTAssertGreaterThan(e.runtimeAPI.port, 0)
        func json(_ d: Data) -> [String: Any] { (try? JSONSerialization.jsonObject(with: d) as? [String: Any]) ?? [:] }

        // Auth and shape errors are OpenAI-shaped.
        var (s, d) = try await http(e, "GET", "/v1/models", token: nil)
        XCTAssertEqual(s, 401)
        XCTAssertEqual((json(d)["error"] as? [String: Any])?["code"] as? String, "invalid_api_key")
        (s, _) = try await http(e, "GET", "/v1/models", token: "plt_wrong")
        XCTAssertEqual(s, 401)
        (s, d) = try await http(e, "GET", "/v1/models", token: secret)
        XCTAssertEqual(s, 200)
        let ids = (json(d)["data"] as? [[String: Any]])?.compactMap { $0["id"] as? String } ?? []
        XCTAssertEqual(ids, ["fake/fake-a"], "models are listed as provider/model")
        XCTAssertEqual(json(d)["object"] as? String, "list")

        let chat: [String: Any] = ["model": "anything", "messages": [["role": "user", "content": "hello api"]]]
        (s, d) = try await http(e, "POST", "/v1/chat/completions", token: secret, body: chat)
        XCTAssertEqual(s, 200)
        let msg = ((json(d)["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])
        XCTAssertEqual((msg?["content"] as? String)?.trimmingCharacters(in: .whitespaces), "(demo) You said: hello api")
        XCTAssertEqual(((json(d)["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String), "stop")
        XCTAssertNotNil((json(d)["usage"] as? [String: Any])?["total_tokens"])

        // Tool calls: names come back exactly as the caller wrote them.
        var withTool = chat
        withTool["messages"] = [["role": "user", "content": "give me a briefing"]]
        withTool["tools"] = [["type": "function", "function": ["name": "briefing", "description": "d", "parameters": ["type": "object", "properties": [:]]]]]
        (s, d) = try await http(e, "POST", "/v1/chat/completions", token: secret, body: withTool)
        let choice = (json(d)["choices"] as? [[String: Any]])?.first
        XCTAssertEqual(choice?["finish_reason"] as? String, "tool_calls")
        let call = ((choice?["message"] as? [String: Any])?["tool_calls"] as? [[String: Any]])?.first
        XCTAssertEqual((call?["function"] as? [String: Any])?["name"] as? String, "briefing")
        XCTAssertEqual(call?["type"] as? String, "function")
        let callID = try XCTUnwrap(call?["id"] as? String)
        // ...and a tool result goes back in.
        var round2 = withTool
        round2["messages"] = [["role": "user", "content": "give me a briefing"],
                              ["role": "assistant", "content": NSNull(), "tool_calls": [["id": callID, "type": "function", "function": ["name": "briefing", "arguments": "{}"]]]],
                              ["role": "tool", "tool_call_id": callID, "content": "all quiet"]]
        (s, d) = try await http(e, "POST", "/v1/chat/completions", token: secret, body: round2)
        XCTAssertEqual(s, 200)
        XCTAssertTrue((((json(d)["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String ?? "").contains("all quiet"))

        // Streaming: SSE chunks, a finish chunk, usage on request, then [DONE].
        var streamed = chat
        streamed["stream"] = true
        streamed["stream_options"] = ["include_usage": true]
        (s, d) = try await http(e, "POST", "/v1/chat/completions", token: secret, body: streamed)
        XCTAssertEqual(s, 200)
        let lines = String(decoding: d, as: UTF8.self).components(separatedBy: "\n\n").filter { !$0.isEmpty }
        XCTAssertEqual(lines.last, "data: [DONE]")
        let chunks = lines.dropLast().compactMap { l -> [String: Any]? in json(Data(l.dropFirst(6).utf8)) }
        XCTAssertEqual(chunks.first.flatMap { (($0["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any])?["role"] as? String }, "assistant")
        let text = chunks.compactMap { (($0["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any])?["content"] as? String }.joined()
        XCTAssertEqual(text.trimmingCharacters(in: .whitespaces), "(demo) You said: hello api")
        XCTAssertTrue(chunks.contains { (($0["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String) == "stop" })
        XCTAssertNotNil(chunks.last?["usage"], "usage chunk when asked")
        XCTAssertEqual(chunks.last?["object"] as? String, "chat.completion.chunk")

        // Unsupported things say so with a 400.
        var n2 = chat; n2["n"] = 2
        (s, d) = try await http(e, "POST", "/v1/chat/completions", token: secret, body: n2)
        XCTAssertEqual(s, 400)
        (s, _) = try await http(e, "POST", "/v1/embeddings", token: secret, body: ["model": "x", "input": "y"])
        XCTAssertEqual(s, 400)
        var lp = chat; lp["logprobs"] = true
        (s, _) = try await http(e, "POST", "/v1/chat/completions", token: secret, body: lp)
        XCTAssertEqual(s, 400)
        (s, _) = try await http(e, "POST", "/v1/chat/completions", token: secret, body: ["messages": []])
        XCTAssertEqual(s, 400)
        (s, _) = try await http(e, "GET", "/v1/nothing", token: secret)
        XCTAssertEqual(s, 404)
    }

    @MainActor
    func testRuntimeAPIModelErrors() async throws {
        let (e, secret) = try await engine(FailingRuntime())
        let ask = { (model: String) async throws -> (Int, [String: Any]) in
            let (s, d) = try await self.http(e, "POST", "/v1/chat/completions", token: secret, body: ["model": model, "messages": [["role": "user", "content": "x"]]])
            return (s, (try? JSONSerialization.jsonObject(with: d) as? [String: Any]) ?? [:])
        }
        var (s, o) = try await ask("missing-model")
        XCTAssertEqual(s, 404)
        XCTAssertEqual((o["error"] as? [String: Any])?["code"] as? String, "model_not_found")
        (s, o) = try await ask("huge-model")
        XCTAssertEqual(s, 503, "a model that cannot load within the budget is a 503")
        XCTAssertEqual((o["error"] as? [String: Any])?["type"] as? String, "server_error")
    }

    /// The runtime API and the app's own chats share one queue.
    @MainActor
    func testRuntimeAPIGoesThroughThePromptQueue() async throws {
        final class Slow: LanguageModelRuntime, @unchecked Sendable {
            let lock = NSLock(); var running = 0; var peak = 0
            func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
                AsyncThrowingStream { c in
                    Task {
                        self.lock.lock(); self.running += 1; self.peak = max(self.peak, self.running); self.lock.unlock()
                        try? await Task.sleep(nanoseconds: 150_000_000)
                        c.yield(.text("ok"))
                        self.lock.lock(); self.running -= 1; self.lock.unlock()
                        c.finish()
                    }
                }
            }
            func loadedModel() async -> String? { nil }
            func unload() async {}
        }
        let slow = Slow()
        let (e, secret) = try await engine(slow)
        async let a = http(e, "POST", "/v1/chat/completions", token: secret, body: ["model": "m", "messages": [["role": "user", "content": "1"]]])
        async let b = http(e, "POST", "/v1/chat/completions", token: secret, body: ["model": "m", "messages": [["role": "user", "content": "2"]]])
        let (ra, rb) = try await (a, b)
        XCTAssertEqual(ra.0, 200); XCTAssertEqual(rb.0, 200)
        XCTAssertEqual(slow.peak, 1, "one generation at a time, like the app's own chats")
    }
}

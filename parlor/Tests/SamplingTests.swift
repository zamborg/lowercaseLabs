import XCTest
@testable import ParlorCore

/// Step 10: sampling is decided once, in the gateway: request > host defaults > machine LLM settings > engine default.
@MainActor
final class SamplingTests: XCTestCase {
    /// A fake OpenAI server whose reply is the sampling keys it received.
    static let echoServer = #"""
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            body = json.dumps({"data": [{"id": "m"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen = {k: req.get(k) for k in ("top_p", "max_tokens", "temperature", "repeat_penalty")}
            text = json.dumps(seen)
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            for chunk in ({"choices": [{"delta": {"content": text}}]}, {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}):
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
    s = HTTPServer(("127.0.0.1", 0), H)
    print(s.server_address[1], flush=True)
    s.serve_forever()
    """#

    private func setup(hostDefaults: [String: Any], config: [String: Any], machine: (inout ParlorSettings) -> Void = { _ in }) async throws -> (ParlorEngine, Process, Room) {
        let home = tempHome(); try home.ensure()
        try #"{"gateway":{"enabled":true,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let script = home.root.appendingPathComponent("echo_server.py")
        try Self.echoServer.write(to: script, atomically: true, encoding: .utf8)
        let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/python3"); p.arguments = [script.path]
        let pipe = Pipe(); p.standardOutput = pipe
        try p.run()
        let port = String(decoding: pipe.fileHandleForReading.availableData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        e.confirm = { _ in true }
        await e.start()
        e.updateSettings(machine)
        let host: [String: Any] = ["id": "lm", "engine": "openai-remote", "url": "http://127.0.0.1:\(port)/v1", "models": ["m"], "defaults": hostDefaults]
        let r = await e.commands.call("hosts.create", try JSONDecoder().decode(JSON.self, from: JSONSerialization.data(withJSONObject: ["draft": host])), caller: .local)
        XCTAssertFalse(r.isError, r.text)
        var a = try XCTUnwrap(e.agent("mabel"))
        a.hosts = ["chat": "lm/m"]; a.tools = []
        a.config = try JSONDecoder().decode([String: JSON].self, from: JSONSerialization.data(withJSONObject: config))
        e.save(a)
        return (e, p, e.room(for: "mabel"))
    }

    private func seen(_ room: Room) throws -> [String: Any] {
        let text = try XCTUnwrap(room.messages.last { $0.role == .assistant }?.text, room.errorText ?? "no reply")
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any], text)
    }

    func testHostDefaultsReachTheBackendWhenTheAgentIsSilent() async throws {
        let (e, p, room) = try await setup(hostDefaults: ["top_p": 0.5, "max_tokens": 123], config: [:])
        defer { p.terminate() }
        await room.sendAndWait("hi", agent: try XCTUnwrap(e.agent("mabel")))
        let s = try seen(room)
        XCTAssertEqual(s["top_p"] as? Double, 0.5); XCTAssertEqual(s["max_tokens"] as? Int, 123)
        await e.shutdown()
    }

    func testTheAgentsConfigBeatsTheHostDefaults() async throws {
        let (e, p, room) = try await setup(hostDefaults: ["top_p": 0.5], config: ["top_p": 0.9, "temperature": 0.25])
        defer { p.terminate() }
        await room.sendAndWait("hi", agent: try XCTUnwrap(e.agent("mabel")))
        let s = try seen(room)
        XCTAssertEqual(s["top_p"] as? Double, 0.9); XCTAssertEqual(s["temperature"] as? Double, 0.25)
        await e.shutdown()
    }

    func testBothSilentFallsBackToTheMachineSettings() async throws {
        let (e, p, room) = try await setup(hostDefaults: [:], config: [:]) { $0.llm.topP = 0.77; $0.llm.maxReplyTokens = 321; $0.llm.repetitionPenalty = 1.2 }
        defer { p.terminate() }
        await room.sendAndWait("hi", agent: try XCTUnwrap(e.agent("mabel")))
        let s = try seen(room)
        XCTAssertEqual(s["top_p"] as? Double, 0.77); XCTAssertEqual(s["max_tokens"] as? Int, 321); XCTAssertEqual(s["repeat_penalty"] as? Double, 1.2)
        await e.shutdown()
    }

    func testTurnStartCarriesLoopKnobsOnly() {
        let a = Agent(id: "x", name: "X", config: ["top_p": .double(0.9)])
        let input = TurnInput(agent: a, conversationID: "x", participants: [.user], transcript: [], summary: nil, tools: [],
                              settings: ResolvedSettings.resolve(agent: a, model: nil, machine: ParlorSettings()), lastPromptTokens: 0, priority: .user, names: [:], memory: "")
        let start = ExternalHarness.turnStart(input, turn: "t").objectValue
        let settings = start?["settings"]?.objectValue ?? [:]
        for k in ["top_p", "temperature", "repetition_penalty", "max_reply_tokens"] { XCTAssertNil(settings[k], k) }
        XCTAssertNotNil(settings["context"]); XCTAssertNotNil(settings["max_steps"])
    }
}

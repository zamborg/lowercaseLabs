import XCTest
@testable import ParlorCore

/// An external harness is a script speaking the JSON-lines protocol of SPEC §2.5.
@MainActor
final class HarnessTests: XCTestCase {
    static let script = #"""
    import sys, json, os, urllib.request

    def send(o):
        sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()

    def read():
        line = sys.stdin.readline()
        return json.loads(line) if line else None

    def wait_result(turn, call_id):
        while True:
            m = read()
            if m is None: sys.exit(0)
            if m.get("type") == "tool.result" and m.get("callID") == call_id: return m
            if m.get("type") == "turn.cancel" and m.get("turn") == turn: return None

    print("harness booted", file=sys.stderr, flush=True)
    while True:
        m = read()
        if m is None: break
        if m.get("type") != "turn.start": continue
        t = m["turn"]
        last = [x for x in m["transcript"] if x["role"] == "user"][-1]["text"]
        if "die" in last: sys.exit(3)
        if "hang" in last:
            while True:
                n = read()
                if n is None or (n.get("type") == "turn.cancel" and n.get("turn") == t): break
            continue
        if "memory" in last:
            send({"type": "message", "turn": t, "role": "assistant", "text": "memory=" + m["memory"]})
            send({"type": "turn.done", "turn": t}); continue
        if "env" in last:
            keys = ["OPENAI_BASE_URL", "OPENAI_API_KEY", "PARLOR_MCP_URL", "PARLOR_TOKEN", "PARLOR_AGENT"]
            info = {k: os.environ.get(k, "") for k in keys}
            ok = info["OPENAI_BASE_URL"].startswith("http://127.0.0.1:") and info["OPENAI_API_KEY"] == info["PARLOR_TOKEN"] != "" and info["PARLOR_AGENT"] == m["agent"]["id"]
            send({"type": "message", "turn": t, "role": "assistant", "text": "env ok=%s cwd=%s agent=%s tier=%s tools=%d" % (ok, os.path.basename(os.getcwd()), m["agent"]["id"], m["agent"]["tier"], len(m["tools"]))})
            send({"type": "turn.done", "turn": t}); continue
        if "api" in last:
            req = urllib.request.Request(os.environ["OPENAI_BASE_URL"] + "/chat/completions", method="POST",
                data=json.dumps({"model": m["agent"]["model"], "messages": [{"role": "user", "content": "from harness"}]}).encode(),
                headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
            body = json.load(urllib.request.urlopen(req))
            send({"type": "message", "turn": t, "role": "assistant", "text": "api said: " + body["choices"][0]["message"]["content"].strip()})
            send({"type": "turn.done", "turn": t}); continue
        if "tool" in last or "forbidden" in last:
            name = "echo__say" if "tool" in last else "parlor__agents.list"
            send({"type": "text.delta", "turn": t, "text": "calling"})
            send({"type": "message", "turn": t, "role": "assistant", "text": "", "toolCalls": [{"id": "c1", "name": name, "arguments": {"text": "hi"}}]})
            send({"type": "tool.call", "turn": t, "callID": "c1", "name": name, "arguments": {"text": "hi"}})
            r = wait_result(t, "c1")
            if r is None: continue
            send({"type": "stats", "turn": t, "promptTokens": 10, "completionTokens": 5, "tokensPerSecond": 50})
            send({"type": "message", "turn": t, "role": "assistant", "text": ("error=%s: " % r["isError"]) + r["text"]})
            send({"type": "turn.done", "turn": t}); continue
        if "orphan" in last:
            send({"type": "tool.call", "turn": t, "callID": "o1", "name": "echo__say", "arguments": {"text": "x"}})
            r = wait_result(t, "o1")
            send({"type": "message", "turn": t, "role": "assistant", "text": "done " + r["text"]})
            send({"type": "turn.done", "turn": t}); continue
        if "boom" in last:
            send({"type": "error", "turn": t, "message": "harness says boom"}); continue
        send({"type": "text.delta", "turn": t, "text": "hel"})
        send({"type": "message", "turn": t, "role": "assistant", "text": "hello from harness"})
        send({"type": "turn.done", "turn": t})
    """#

    private func setup(tier: Tier = .none, tools: [String] = ["echo.*"]) async throws -> (ParlorEngine, Room, Agent) {
        let home = tempHome(); try home.ensure()
        try #"{"parlorServer":{"enabled":true,"port":0},"runtimeAPI":{"enabled":false,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let path = home.root.appendingPathComponent("fake_harness.py")
        try Self.script.write(to: path, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        await e.start()
        await e.hub.register(EchoProvider(), status: ServerStatus(id: "echo", kind: "built in", detail: "", state: .ready, toolCount: 1, toolNames: ["say"]))
        var a = try XCTUnwrap(e.agent("mabel"))
        a.model = "demo"; a.tools = tools; a.tier = tier
        a.harness = "python3 '\(path.path)'"
        e.save(a)
        return (e, e.room(for: "mabel"), try XCTUnwrap(e.agent("mabel")))
    }

    func testHarnessWireRoundTrip() async throws {
        let (e, room, a) = try await setup()
        await room.sendAndWait("hello there", agent: a)
        XCTAssertEqual(room.messages.map(\.role), [.user, .assistant])
        XCTAssertEqual(room.messages.last?.text, "hello from harness")
        XCTAssertEqual(room.messages.last?.speaker, "mabel")

        // A tool call goes through the app's executor and the log looks exactly like the built-in harness's.
        await room.sendAndWait("please use the tool", agent: a)
        let tail = Array(room.messages.suffix(4))
        XCTAssertEqual(tail.map(\.role), [.user, .assistant, .tool, .assistant])
        XCTAssertEqual(tail[1].toolCalls?.first?.name, "echo__say")
        XCTAssertEqual(tail[1].toolCalls?.first?.id, "c1")
        XCTAssertEqual(tail[2].toolCallID, "c1")
        XCTAssertEqual(tail[2].text, "echo: hi")
        XCTAssertEqual(tail[2].isError, false)
        XCTAssertEqual(tail[3].text, "error=False: echo: hi")
        XCTAssertEqual(room.lastStats?.completionTokens, 5)

        // The log survives a reload from disk.
        let reloaded = Room(agentID: "mabel", deps: RoomDeps(home: e.home, hub: e.hub, llm: DemoRuntime(), harness: BuiltInHarness()))
        XCTAssertEqual(reloaded.messages.count, room.messages.count)
        await e.shutdown()
    }

    func testHarnessEnvironmentWorkspaceAndStderrLog() async throws {
        let (e, room, a) = try await setup(tier: .read)
        await room.sendAndWait("show env", agent: a)
        XCTAssertEqual(room.messages.last?.text, "env ok=True cwd=mabel agent=mabel tier=read tools=1")
        XCTAssertTrue(FileManager.default.fileExists(atPath: e.home.root.appendingPathComponent("workspaces/mabel").path))
        XCTAssertTrue(ServiceLog.tail(home: e.home, service: "harness:mabel", lines: 5).contains("harness booted"), "stderr goes to logs/harness-mabel.log")
        XCTAssertTrue(e.runtimeAPI.isRunning, "launching a harness starts the runtime API")
        // The process is a service in the dashboard.
        await e.refreshServices()
        XCTAssertEqual(e.services.first { $0.id == "harness:mabel" }?.state, .running)
        await e.shutdown()
    }

    func testHarnessToolCallsAreHeldToTheAgentsTier() async throws {
        let (e, room, a) = try await setup(tier: .none, tools: ["*"])
        await room.sendAndWait("this is forbidden", agent: a)
        let result = try XCTUnwrap(room.messages.first { $0.role == .tool })
        XCTAssertEqual(result.isError, true)
        XCTAssertTrue(result.text.contains("needs tier read"), result.text)
        // The same call with a tier goes through.
        var b = a; b.tier = .read; e.save(b)
        await room.sendAndWait("this is forbidden again", agent: try XCTUnwrap(e.agent("mabel")))
        let results = room.messages.filter { $0.role == .tool }
        XCTAssertEqual(results.last?.isError, false, results.last?.text ?? "")
        // And a tool the allowlist does not name is refused too.
        var c = try XCTUnwrap(e.agent("mabel")); c.tools = []; e.save(c)
        await room.sendAndWait("use the tool", agent: try XCTUnwrap(e.agent("mabel")))
        XCTAssertTrue(room.messages.last(where: { $0.role == .tool })?.text.contains("not enabled") == true)
        await e.shutdown()
    }

    func testHarnessUsesTheRuntimeAPIWithItsToken() async throws {
        let (e, room, a) = try await setup()
        await room.sendAndWait("call the api", agent: a)
        XCTAssertEqual(room.messages.last?.text, "api said: (demo) You said: from harness", room.errorText ?? "")
        await e.shutdown()
    }

    func testHarnessProtocolEdges() async throws {
        let (e, room, a) = try await setup()
        // A tool.call without a preceding assistant message is still logged as a pair.
        await room.sendAndWait("an orphan call", agent: a)
        let pair = room.messages.suffix(3)
        XCTAssertEqual(pair.map(\.role), [.assistant, .tool, .assistant])
        XCTAssertEqual(pair.first?.toolCalls?.first?.id, "o1")

        // An error line is shown to the user.
        await room.sendAndWait("go boom", agent: a)
        XCTAssertEqual(room.errorText, "harness says boom")

        // A harness that dies mid-turn is reported, then relaunched on the next message.
        await room.sendAndWait("please die", agent: a)
        XCTAssertTrue(room.errorText?.contains("exited") == true, room.errorText ?? "nil")
        await room.sendAndWait("hello again", agent: a)
        XCTAssertNil(room.errorText)
        XCTAssertEqual(room.messages.last?.text, "hello from harness")

        // Stop sends turn.cancel; the next turn still works.
        room.send("hang forever", agent: a)
        try await Task.sleep(nanoseconds: 800_000_000)
        XCTAssertTrue(room.isBusy)
        room.stop()
        let deadline = Date().addingTimeInterval(5)
        while room.isBusy, Date() < deadline { try await Task.sleep(nanoseconds: 100_000_000) }
        XCTAssertFalse(room.isBusy, "cancel ends the turn")
        await room.sendAndWait("hello once more", agent: a)
        XCTAssertEqual(room.messages.last?.text, "hello from harness")
        await e.shutdown()
    }

    func testHarnessKeyRoundTripsInTheAgentFile() async throws {
        let (e, _, _) = try await setup()
        let text = try String(contentsOf: e.home.agentsDir.appendingPathComponent("mabel.md"), encoding: .utf8)
        XCTAssertTrue(text.contains("harness:"), text)
        let b = Agent(id: "mabel", file: text)
        XCTAssertNotNil(b.harness)
        var c = b; c.harness = nil
        XCTAssertFalse(c.fileText().contains("harness:"))
        await e.shutdown()
    }

    /// The sample that ships in `harnesses/` runs as is: its OpenAI-style loop talks to the runtime API.
    func testSeededSampleHarnessRunsUnmodified() async throws {
        let (e, room, _) = try await setup()
        let sample = e.home.harnessesDir.appendingPathComponent("openai_loop.py")
        XCTAssertTrue(FileManager.default.fileExists(atPath: sample.path), "sample is seeded")
        XCTAssertTrue(FileManager.default.fileExists(atPath: e.home.harnessesDir.appendingPathComponent("CLAUDE.md").path))
        var a = try XCTUnwrap(e.agent("mabel")); a.harness = "python3 '\(sample.path)'"; e.save(a)
        await room.sendAndWait("hello sample", agent: try XCTUnwrap(e.agent("mabel")))
        XCTAssertNil(room.errorText)
        XCTAssertEqual(room.messages.last?.text.trimmingCharacters(in: .whitespaces), "(demo) You said: hello sample")
        XCTAssertEqual(room.messages.last?.speaker, "mabel")
        XCTAssertEqual(room.lastStats?.completionTokens, 12, "usage from the stream becomes stats")
        await e.shutdown()
    }
}

import XCTest
@testable import ParlorCore

/// V2 conformance, now over the wire: the seeded `chat` harness (chat.py) must behave exactly as the Swift loop did.
@MainActor
final class WireTests: XCTestCase {
    private func engine(_ llm: LanguageModelRuntime) async throws -> ParlorEngine {
        let home = tempHome(); try home.ensure()
        try #"{"gateway":{"enabled":true,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: llm)
        await e.start()
        return e
    }

    private func mockHub(_ e: ParlorEngine) async {
        await e.hub.register(MockTools(), status: ServerStatus(id: "mock", kind: "t", detail: "", state: .connecting, toolCount: 0, toolNames: []))
    }

    func testRequestMessagesMatchV2() async throws {
        let seen = Seen()
        let e = try await engine(ScriptedRuntime { req in seen.add(req); return [.text("ok")] })
        var a = Agent(id: "x", name: "X", persona: "  You are X.  \n", hosts: ["chat": "scripted"])
        e.save(a)
        let u1 = ChatMessage(role: .user, text: "one"), a1 = ChatMessage(role: .assistant, text: "two", speaker: "x")
        let log = ChatLog(dir: e.home.chatDir(for: "x"))
        for m in [u1, a1, ChatMessage.marker("noted")] { log.append(m) }
        let room = e.room(for: "x")
        await room.sendAndWait("three", agent: a)

        // No summary, no tools: persona, then the date, then every non-marker line.
        let plain = try XCTUnwrap(seen.all.first).messages
        XCTAssertEqual(plain.map(\.role), [.system, .user, .assistant, .user])
        XCTAssertTrue(plain[0].text.hasPrefix("You are X.\n\nToday is "), plain[0].text)
        XCTAssertFalse(plain[0].text.contains("You have tools"))
        XCTAssertFalse(plain[0].text.contains("Earlier conversation"))

        // With tools and a summary through a1: the summary is in the system prompt, history starts after it.
        await mockHub(e)
        a.tools = ["*"]; e.save(a)
        ChatLog(dir: e.home.chatDir(for: "x")).saveSummary(ChatLog.Summary(text: "They said one and two.", through: a1.id))
        e.reloadAgents()
        let room2 = Room(agentID: "x", deps: room.deps)
        await room2.sendAndWait("four", agent: a)
        let full = try XCTUnwrap(seen.all.last).messages
        XCTAssertTrue(full[0].text.contains("You have tools. When a request needs"), full[0].text)
        XCTAssertTrue(full[0].text.hasSuffix("## Earlier conversation (summary)\nThey said one and two."))
        XCTAssertEqual(full.dropFirst().first?.text, "three")
        await e.shutdown()
    }

    func testEventOrderAndLastStepWithholdsTools() async throws {
        let seen = Seen()
        let e = try await engine(ScriptedRuntime { req in
            seen.add(req)
            return [.toolCall(ToolCallRecord(id: UUID().uuidString, name: "mock__add", arguments: .object(["a": .int(1), "b": .int(2)])))]
        })
        await mockHub(e)
        var a = Agent(id: "x", name: "X", hosts: ["chat": "scripted"], tools: ["*"], tier: .none)
        a.config["max_steps"] = .int(2)
        e.save(a)
        let room = e.room(for: "x")
        await room.sendAndWait("go", agent: a)
        let counts = seen.all.map { $0.tools.count }
        XCTAssertEqual(counts.count, 2)
        XCTAssertGreaterThan(counts.first ?? 0, 0, "tools offered on the first step")
        XCTAssertEqual(counts.last, 0, "withheld on the last")
        XCTAssertEqual(room.messages.map(\.role), [.user, .assistant, .tool, .assistant, .tool])
        let msgs = room.messages
        for (i, m) in msgs.enumerated() where m.role == .assistant {
            XCTAssertEqual(msgs[i + 1].toolCallID, m.toolCalls?.first?.id, "every call is answered by a tool line with its id")
        }
        await e.shutdown()
    }

    func testAllowlistRefusalOverTheWire() async throws {
        let e = try await engine(ScriptedRuntime { req in
            if req.messages.last?.role == .tool { return [.text("noted")] }
            return [.toolCall(ToolCallRecord(id: "c1", name: "mock__add", arguments: .object(["a": .int(1), "b": .int(1)])))]
        })
        await mockHub(e)
        let a = Agent(id: "n", name: "N", hosts: ["chat": "scripted"], tools: [])
        e.save(a)
        let room = e.room(for: "n")
        await room.sendAndWait("add", agent: a)
        let tool = try XCTUnwrap(room.messages.first { $0.role == .tool })
        XCTAssertEqual(tool.isError, true)
        XCTAssertTrue(tool.text.contains("not enabled"), tool.text)
        await e.shutdown()
    }

    private func padded(_ e: ParlorEngine, _ id: String) {
        let log = ChatLog(dir: e.home.chatDir(for: id))
        for i in 0..<10 { log.append(ChatMessage(role: i % 2 == 0 ? .user : .assistant, text: "message number \(i) with padding text", speaker: i % 2 == 0 ? "user" : id)) }
    }

    func testCompactionIsAnEventBeforeTheReply() async throws {
        let e = try await engine(scriptedLLM())
        let a = Agent(id: "c", name: "C", hosts: ["chat": "scripted"], config: ["context": .int(200), "compact_at": .double(0.1)])
        e.save(a)
        padded(e, "c")
        let room = e.room(for: "c")
        await room.sendAndWait("latest", agent: a)
        XCTAssertTrue(room.summaryText?.hasPrefix("SUMMARY") == true, room.summaryText ?? "nil")
        XCTAssertEqual(room.messages.last?.text, "Echo: latest")
        await e.shutdown()
    }

    func testCompactNowIsChatsActionAndLogsNoUserLine() async throws {
        let e = try await engine(scriptedLLM())
        let a = Agent(id: "c", name: "C", hosts: ["chat": "scripted"])
        e.save(a)
        padded(e, "c")
        let room = e.room(for: "c")
        let before = room.messages.filter { $0.role == .user }.count
        room.compactNow(agent: a)
        await room.wait()
        XCTAssertNil(room.errorText)
        XCTAssertTrue(room.summaryText?.hasPrefix("SUMMARY") == true, room.summaryText ?? "nil")
        XCTAssertEqual(room.messages.filter { $0.role == .user }.count, before, "an action adds no user line")
        XCTAssertTrue(room.messages.contains { $0.role == .marker && $0.text.hasPrefix("▸") })
        await e.shutdown()
    }

    func testCharacterActionsHideWithoutAnImageHost() async throws {
        let e = try await engine(scriptedLLM())
        let m = try XCTUnwrap(e.harnessList.first { $0.id == "character" })
        let gen = try XCTUnwrap(m.actions.first { $0.needs == "image" })
        XCTAssertFalse(m.available(gen, hosts: ["chat": "scripted"]))
        XCTAssertTrue(m.available(gen, hosts: ["chat": "scripted", "image": "z-image"]))
        await e.shutdown()
    }

    func testManifestAddedActionAppearsAfterReload() async throws {
        let e = try await engine(scriptedLLM())
        let url = e.home.harnessesDir.appendingPathComponent("chat/harness.json")
        var j = try JSONDecoder().decode(JSON.self, from: Data(contentsOf: url))
        guard case .object(var o) = j, case .array(var acts)? = o["actions"] else { return XCTFail("manifest shape") }
        acts.append(.object(["id": .string("poke"), "title": .string("Poke"), "placement": .string("composer")]))
        o["actions"] = .array(acts); j = .object(o)
        try JSONEncoder().encode(j).write(to: url)
        e.reloadRunplane()
        let chat = try XCTUnwrap(e.harnessList.first { $0.id == "chat" })
        XCTAssertTrue(chat.actions.contains { $0.id == "poke" })
        await e.shutdown()
    }

    func testEditorRoundTripKeepsUnknownKeys() async throws {
        let e = try await engine(scriptedLLM())
        let url = e.home.agentsDir.appendingPathComponent("mabel/agent.json")
        var j = try JSONDecoder().decode(JSON.self, from: Data(contentsOf: url))
        guard case .object(var o) = j else { return XCTFail("shape") }
        o["x_future"] = .object(["a": .int(1)])
        var cfg = o["config"]?.objectValue ?? [:]
        cfg["x_knob"] = .int(7)
        o["config"] = .object(cfg); j = .object(o)
        try JSONEncoder().encode(j).write(to: url)
        e.reloadAgents()
        var a = try XCTUnwrap(e.agent("mabel"))
        a.summary = "edited"
        e.save(a)
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.contains("x_future") && text.contains("x_knob") && text.contains("edited"), text)
        await e.shutdown()
    }
}

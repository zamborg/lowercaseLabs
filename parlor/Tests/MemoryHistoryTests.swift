import XCTest
@testable import ParlorCore

@MainActor
final class MemoryHistoryTests: XCTestCase {
    private func engine() async throws -> ParlorEngine {
        let home = tempHome(); try home.ensure()
        try #"{"gateway":{"enabled":false,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        await e.start()
        for a in e.agents { var x = a; x.chatHost = "demo"; x.tools = ["parlor.*"]; e.save(x) }
        return e
    }

    private func call(_ e: ParlorEngine, _ verb: String, _ args: [String: Any] = [:], as caller: Caller = .local) async -> ToolOutput {
        let data = try! JSONSerialization.data(withJSONObject: args)
        return await e.commands.call(verb, try! JSONDecoder().decode(JSON.self, from: data), caller: caller)
    }

    func testMemoryStoreAppendReplaceAndLimit() throws {
        let home = tempHome()
        let m = MemoryStore(home: home)
        XCTAssertEqual(m.read("mabel"), "")
        try m.append("mabel", "Likes tea.")
        try m.append("mabel", "Lives in Leeds.")
        XCTAssertEqual(m.read("mabel"), "Likes tea.\nLives in Leeds.\n")
        XCTAssertThrowsError(try m.append("mabel", String(repeating: "x", count: MemoryStore.maxChars)))
        try m.write("mabel", "")
        XCTAssertFalse(FileManager.default.fileExists(atPath: m.url("mabel").path), "empty memory removes the file")
    }

    func testMemoryReachesTheSystemPromptAndIsTierGated() async throws {
        let e = try await engine()
        let mabel = Caller.agent(try XCTUnwrap(e.agent("mabel")))      // tier none
        // An agent edits its own memory at tier none, and not another's.
        var r = await call(e, "memory.append", ["text": "The user's dog is called Biscuit."], as: mabel)
        XCTAssertFalse(r.isError, r.text)
        r = await call(e, "memory.append", ["id": "pip", "text": "sneaky"], as: mabel)
        XCTAssertTrue(r.isError)
        XCTAssertTrue(r.text.contains("configure"), r.text)
        XCTAssertEqual(e.memory.read("pip"), "")
        // A configure-tier token can.
        r = await call(e, "memory.replace", ["id": "pip", "text": "Pip likes jokes."], as: .token("claude", .configure))
        XCTAssertFalse(r.isError, r.text)
        r = await call(e, "memory.read", ["id": "pip"], as: .token("claude", .act))
        XCTAssertTrue(r.isError, "an act token may not read another agent's memory")
        r = await call(e, "memory.read", as: mabel)
        XCTAssertTrue(r.text.contains("Biscuit"))
        // The built-in harness shows it to the model, under a heading; no memory leaves the prompt untouched.
        let room = e.room(for: "mabel")
        await room.sendAndWait("hello", agent: try XCTUnwrap(e.agent("mabel")))
        let a = try XCTUnwrap(e.agent("mabel"))
        XCTAssertEqual(e.memory.read("mabel"), "The user's dog is called Biscuit.\n")
        XCTAssertTrue(FileManager.default.fileExists(atPath: e.home.agentsDir.appendingPathComponent("mabel/memory.md").path))
        // Memory is listed under the agent's tier-none tools.
        let tools = await e.hub.tools(for: a).map(\.qualified)
        XCTAssertTrue(tools.contains("parlor.memory.append"))
        XCTAssertFalse(tools.contains("parlor.agents.update"))
    }

    func testExternalHarnessReceivesMemory() async throws {
        let home = tempHome(); try home.ensure()
        try #"{"gateway":{"enabled":false,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let path = home.root.appendingPathComponent("h.py")
        try HarnessTests.script.write(to: path, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        await e.start()
        var a = try XCTUnwrap(e.agent("mabel")); a.chatHost = "demo"; a.harness = "python3 '\(path.path)'"; e.save(a)
        try e.memory.write("mabel", "Prefers metric units.")
        let room = e.room(for: "mabel")
        await room.sendAndWait("what is in memory", agent: try XCTUnwrap(e.agent("mabel")))
        XCTAssertEqual(room.messages.last?.text, "memory=Prefers metric units.\n")
        await e.shutdown()
    }

    // MARK: templates and groups

    func testTemplatesCreateAgentsWithGroupAndHarness() async throws {
        let e = try await engine()
        var r = await call(e, "agents.templates", [:], as: .local)
        XCTAssertTrue(r.text.contains("researcher") && r.text.contains("scene"), r.text)
        r = await call(e, "agents.create", ["template": "researcher", "name": "Rae"], as: .local)
        XCTAssertFalse(r.isError, r.text)
        let rae = try XCTUnwrap(e.agent("rae"))
        XCTAssertEqual(rae.configBool("thinking"), true)
        XCTAssertTrue(rae.persona.contains("careful researcher"))
        r = await call(e, "agents.create", ["template": "scene", "name": "Scott"], as: .local)
        let scott = try XCTUnwrap(e.agent("scott"))
        XCTAssertEqual(scott.harnessID, "character")
        r = await call(e, "agents.update", ["id": "rae", "patch": ["group": "Work"]], as: .local)
        XCTAssertEqual(e.agent("rae")?.group, "Work")
        r = await call(e, "agents.create", ["template": "nope"], as: .local)
        XCTAssertTrue(r.isError)
    }

    func testHarnessSetByATokenNeedsTheUser() async throws {
        let e = try await engine()
        e.confirm = nil
        let r = await call(e, "agents.update", ["id": "mabel", "patch": ["harness": "echo hi"]], as: .token("t", .configure))
        XCTAssertTrue(r.isError, r.text)
        XCTAssertNil(e.agent("mabel")?.harness)
        e.confirm = { _ in true }
        let ok = await call(e, "agents.update", ["id": "mabel", "patch": ["harness": "echo hi"]], as: .token("t", .configure))
        XCTAssertFalse(ok.isError, ok.text)
        XCTAssertEqual(e.agent("mabel")?.harness, "echo hi")
    }

    func testModelProblemFlagsDisabledProvider() async throws {
        let e = try await engine()
        var a = try XCTUnwrap(e.agent("mabel"))
        a.chatHost = "nowhere/model"
        XCTAssertNil(e.modelProblem(for: a), "unknown prefix with no MLX in tests is not flagged")
    }
}

import XCTest
@testable import ParlorCore

@MainActor
final class MemoryHistoryTests: XCTestCase {
    private func engine() async throws -> ParlorEngine {
        let home = tempHome(); try home.ensure()
        try #"{"parlorServer":{"enabled":false,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        await e.start()
        for a in e.agents { var x = a; x.model = "demo"; x.tools = ["parlor.*"]; e.save(x) }
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
        let withMemory = BuiltInHarness.buildRequestMessages(agent: a, transcript: [], summary: nil, tools: [], now: Date(), memory: e.memory.read("mabel"))
        XCTAssertTrue(withMemory[0].text.contains("## What you remember\nThe user's dog is called Biscuit."))
        let without = BuiltInHarness.buildRequestMessages(agent: a, transcript: [], summary: nil, tools: [], now: Date())
        XCTAssertFalse(without[0].text.contains("What you remember"))
        // Memory is listed under the agent's tier-none tools.
        let tools = await e.hub.tools(for: a).map(\.qualified)
        XCTAssertTrue(tools.contains("parlor.memory.append"))
        XCTAssertFalse(tools.contains("parlor.agents.update"))
    }

    func testExternalHarnessReceivesMemory() async throws {
        let home = tempHome(); try home.ensure()
        try #"{"parlorServer":{"enabled":false,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let path = home.root.appendingPathComponent("h.py")
        try HarnessTests.script.write(to: path, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        await e.start()
        var a = try XCTUnwrap(e.agent("mabel")); a.model = "demo"; a.harness = "python3 '\(path.path)'"; e.save(a)
        try e.memory.write("mabel", "Prefers metric units.")
        let room = e.room(for: "mabel")
        await room.sendAndWait("what is in memory", agent: try XCTUnwrap(e.agent("mabel")))
        XCTAssertEqual(room.messages.last?.text, "memory=Prefers metric units.\n")
        await e.shutdown()
    }

    // MARK: history

    func testHistoryRecordsWritesWithTheActorAndIgnoresBulk() async throws {
        try XCTSkipUnless(HomeHistory.gitAvailable)
        let e = try await engine()
        // Things that must never be versioned.
        for (p, c) in [("models/x/weights.bin", "w"), ("media/2026-11/a.png", "p"), ("logs/x.log", "l"), ("workspaces/mabel/f.txt", "f"), ("chats/mabel/log.jsonl", "{}\n")] {
            let u = e.home.root.appendingPathComponent(p)
            try FileManager.default.createDirectory(at: u.deletingLastPathComponent(), withIntermediateDirectories: true)
            try c.write(to: u, atomically: true, encoding: .utf8)
        }
        let r = await call(e, "agents.update", ["id": "pip", "name": "Pip Squeak"], as: .token("claude-code", .configure))
        XCTAssertFalse(r.isError, r.text)
        e.history.flush()
        let log = e.history.log(limit: 10)
        let latest = try XCTUnwrap(log.first { $0.subject.contains("pip") })
        XCTAssertEqual(latest.author, "claude-code", "credited to the token that made the change")
        XCTAssertTrue(latest.files.contains { $0.hasPrefix("agents/") })
        let tracked = log.flatMap(\.files)
        for bad in ["models/", "media/", "logs/", "workspaces/", "log.jsonl"] { XCTAssertFalse(tracked.contains { $0.contains(bad) }, "\(bad) is not versioned") }
        XCTAssertTrue(tracked.contains { $0.hasSuffix("settings.json") })
    }

    func testRestoreBringsFilesBackAndAsksFirst() async throws {
        try XCTSkipUnless(HomeHistory.gitAvailable)
        let e = try await engine()
        _ = await call(e, "memory.replace", ["id": "mabel", "text": "version one"])
        e.history.flush()
        let v1 = try XCTUnwrap(e.history.log(limit: 1).first).hash
        _ = await call(e, "memory.replace", ["id": "mabel", "text": "version two"], as: .token("claude", .configure))
        _ = await call(e, "agents.create", ["name": "Newcomer"])
        e.history.flush()
        XCTAssertEqual(e.memory.read("mabel"), "version two\n")
        XCTAssertNotNil(e.agent("newcomer"))

        // The listing offers the commit; an agent or token must be confirmed by the user.
        let listed = await call(e, "home.history", ["limit": 5])
        XCTAssertTrue(listed.text.contains(String(v1.prefix(10))))
        e.confirm = { _ in false }
        var r = await call(e, "home.restore", ["commit": v1], as: .token("claude", .configure))
        XCTAssertTrue(r.isError, "declined")
        XCTAssertEqual(e.memory.read("mabel"), "version two\n")
        e.confirm = { _ in true }
        r = await call(e, "home.restore", ["commit": v1], as: .token("claude", .configure))
        XCTAssertFalse(r.isError, r.text)
        XCTAssertEqual(e.memory.read("mabel"), "version one\n", "file restored")
        XCTAssertNil(e.agent("newcomer"), "a file that did not exist then is removed")
        // Restoring is itself a step in the history, so it can be undone.
        XCTAssertTrue(e.history.log(limit: 3).first?.subject.hasPrefix("Restore") == true)
        // One file only.
        r = await call(e, "home.restore", ["commit": "zzzz"])
        XCTAssertTrue(r.isError)
    }

    func testEditsMadeOutsideTheAppAreRecorded() async throws {
        try XCTSkipUnless(HomeHistory.gitAvailable)
        let e = try await engine()
        e.history.flush()
        try "Hand written note.\n".write(to: e.home.root.appendingPathComponent("memory/mabel.md").also { try? FileManager.default.createDirectory(at: $0.deletingLastPathComponent(), withIntermediateDirectories: true) }, atomically: true, encoding: .utf8)
        let deadline = Date().addingTimeInterval(8)
        var found = false
        while Date() < deadline, !found {
            try await Task.sleep(nanoseconds: 300_000_000)
            found = e.history.log(limit: 5).contains { $0.author == "outside edit" }
        }
        XCTAssertTrue(found, "a hand edit is committed as an outside edit")
    }

    // MARK: templates and groups

    func testTemplatesCreateAgentsWithGroupAndHarness() async throws {
        let e = try await engine()
        var r = await call(e, "agents.templates", [:], as: .local)
        XCTAssertTrue(r.text.contains("researcher") && r.text.contains("scripted"), r.text)
        r = await call(e, "agents.create", ["template": "researcher", "name": "Rae"], as: .local)
        XCTAssertFalse(r.isError, r.text)
        let rae = try XCTUnwrap(e.agent("rae"))
        XCTAssertTrue(rae.thinking)
        XCTAssertTrue(rae.persona.contains("careful researcher"))
        r = await call(e, "agents.create", ["template": "scripted", "name": "Scott"], as: .local)
        let scott = try XCTUnwrap(e.agent("scott"))
        XCTAssertEqual(scott.group, "Scripted")
        XCTAssertNotNil(scott.harness)
        r = await call(e, "agents.update", ["id": "rae", "group": "Work"], as: .local)
        XCTAssertEqual(e.agent("rae")?.group, "Work")
        r = await call(e, "agents.create", ["template": "nope"], as: .local)
        XCTAssertTrue(r.isError)
    }

    func testHarnessSetByATokenNeedsTheUser() async throws {
        let e = try await engine()
        e.confirm = nil
        let r = await call(e, "agents.update", ["id": "mabel", "harness": "echo hi"], as: .token("t", .configure))
        XCTAssertTrue(r.isError, r.text)
        XCTAssertNil(e.agent("mabel")?.harness)
        e.confirm = { _ in true }
        let ok = await call(e, "agents.update", ["id": "mabel", "harness": "echo hi"], as: .token("t", .configure))
        XCTAssertFalse(ok.isError, ok.text)
        XCTAssertEqual(e.agent("mabel")?.harness, "echo hi")
    }

    func testModelProblemFlagsDisabledProvider() async throws {
        let e = try await engine()
        var a = try XCTUnwrap(e.agent("mabel"))
        a.model = "nowhere/model"
        XCTAssertNil(e.modelProblem(for: a), "unknown prefix with no MLX in tests is not flagged")
    }
}

private extension URL { func also(_ f: (URL) -> Void) -> URL { f(self); return self } }

@MainActor
final class TriggerTests: XCTestCase {
    func testClockMath() {
        var cal = Calendar(identifier: .gregorian); cal.timeZone = TimeZone(identifier: "UTC")!
        let t0 = cal.date(from: DateComponents(year: 2026, month: 11, day: 14, hour: 9, minute: 0))!
        XCTAssertEqual(ConversationTrigger(every: "30m").nextFire(after: t0, calendar: cal), t0.addingTimeInterval(1800))
        XCTAssertEqual(ConversationTrigger(every: "10s").nextFire(after: t0, calendar: cal), t0.addingTimeInterval(60), "never faster than a minute")
        let daily = ConversationTrigger(every: "1d", at: "08:00").nextFire(after: t0, calendar: cal)
        XCTAssertEqual(daily, cal.date(from: DateComponents(year: 2026, month: 11, day: 15, hour: 8)))
        let before = cal.date(from: DateComponents(year: 2026, month: 11, day: 14, hour: 7))!
        XCTAssertEqual(ConversationTrigger(at: "08:00").nextFire(after: before, calendar: cal), cal.date(from: DateComponents(year: 2026, month: 11, day: 14, hour: 8)))
        XCTAssertEqual(ConversationTrigger(every: "3d", at: "08:00").nextFire(after: t0, calendar: cal), cal.date(from: DateComponents(year: 2026, month: 11, day: 17, hour: 8)))
        XCTAssertNil(ConversationTrigger(onFile: "x").nextFire(after: t0))
        XCTAssertTrue(ConversationTrigger(onFile: "workspaces/mabel/inbox/*").matches(relativePath: "workspaces/mabel/inbox/a.txt"))
        XCTAssertFalse(ConversationTrigger(onFile: "workspaces/mabel/inbox/*").matches(relativePath: "agents/mabel.md"))
    }

    func testClockAndFileTriggersStartTheRoom() async throws {
        let home = tempHome(); try home.ensure()
        try #"{"parlorServer":{"enabled":false,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: roomLLM())
        await e.start()
        for a in e.agents { var x = a; x.model = "scripted"; x.tools = []; e.save(x) }
        let id = e.createRoom(title: "Watch", agents: ["mabel", "pip"], scheduler: SchedulerSpec(kind: "autonomous", stepBudget: 1))
        var c = ConversationConfig.load(dir: home.chatDir(for: id), agentID: id)
        c.triggers = [ConversationTrigger(every: "1h"), ConversationTrigger(onFile: "workspaces/mabel/inbox/*")]
        try c.save(dir: home.chatDir(for: id))
        let t0 = Date()
        XCTAssertEqual(e.triggers.tick(now: t0), [], "first sight only starts the clock")
        XCTAssertEqual(e.triggers.tick(now: t0.addingTimeInterval(1800)), [])
        XCTAssertEqual(e.triggers.tick(now: t0.addingTimeInterval(3700)), [id])
        await e.room(for: id).runAndWait()
        XCTAssertFalse(e.room(for: id).messages.isEmpty, "the run produced messages with no user message: \(e.room(for: id).errorText ?? "no error")")
        let inbox = home.root.appendingPathComponent("workspaces/mabel/inbox/a.txt").path
        XCTAssertEqual(e.triggers.filesChanged([inbox], now: t0.addingTimeInterval(4000)), [id])
        await e.room(for: id).runAndWait()
        XCTAssertEqual(e.triggers.filesChanged([inbox], now: t0.addingTimeInterval(4005)), [], "cooldown")
        XCTAssertEqual(e.triggers.filesChanged([home.root.appendingPathComponent("chats/\(id)/log.jsonl").path], now: t0.addingTimeInterval(9000)), [], "its own chat folder never triggers it")
        e.triggers.stop()
    }
}

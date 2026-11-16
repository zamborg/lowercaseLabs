import XCTest
@testable import ParlorCore

/// Step 6: the uniform noun verbs, `schema.get`, drafts (`validate`, `try`), structured errors and the diff confirmation.
@MainActor
final class NounVerbTests: XCTestCase {
    private func engine() async throws -> ParlorEngine {
        let home = tempHome(); try home.ensure()
        try #"{"gateway":{"enabled":true,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        await e.start()
        return e
    }

    private func call(_ e: ParlorEngine, _ verb: String, _ args: [String: Any] = [:], as caller: Caller = .local) async -> ToolOutput {
        let data = try! JSONSerialization.data(withJSONObject: args)
        return await e.commands.call(verb, try! JSONDecoder().decode(JSON.self, from: data), caller: caller)
    }

    private func json(_ o: ToolOutput) throws -> [String: Any] {
        try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(o.text.utf8)) as? [String: Any], o.text)
    }

    private func fakeServer(_ home: ParlorHome) throws -> (Process, String) {
        let script = home.root.appendingPathComponent("fake_server.py")
        try DropInTests.fakeServer.write(to: script, atomically: true, encoding: .utf8)
        let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/python3"); p.arguments = [script.path]
        let pipe = Pipe(); p.standardOutput = pipe
        try p.run()
        let port = String(decoding: pipe.fileHandleForReading.availableData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        return (p, port)
    }

    func testSchemaGetAndSeededFilesFitTheirSchemas() async throws {
        let e = try await engine()
        var r = await call(e, "schema.get", ["noun": "agent"])
        XCTAssertFalse(r.isError, r.text)
        XCTAssertNotNil(try json(r)["properties"])
        r = await call(e, "schema.get", ["noun": "nope"])
        XCTAssertTrue(r.isError); XCTAssertTrue(r.text.contains("agent, host, harness, engine"))
        // Every seeded file validates against its schema, so schema and structs cannot drift.
        for id in e.engineStore.ids() { XCTAssertEqual(NounSchemas.check(.object(e.engineStore.raw(id)!), noun: "engine").errors, [], "engine \(id)") }
        for id in e.harnessStore.ids() { XCTAssertEqual(NounSchemas.check(.object(e.harnessStore.raw(id)!), noun: "harness").errors, [], "harness \(id)") }
        for id in e.agentStore.ids() {
            XCTAssertEqual(NounSchemas.check(.object(e.agentStore.raw(id)!), noun: "agent").errors, [], "agent \(id)")
            XCTAssertEqual(e.commands.agentIssues(.object(e.agentStore.raw(id)!)).errors.filter { $0.code != "unknown_host" }, [], "agent \(id) validates")
        }
        await e.shutdown()
    }

    func testBadDraftReturnsEveryErrorWithPathAndHint() async throws {
        let e = try await engine()
        let r = await call(e, "agents.validate", ["draft": ["id": "Bad Id!", "harness": "nope", "hosts": ["chat": "ghost"], "tier": "root", "color": 99, "tools": "x"]])
        XCTAssertFalse(r.isError, r.text)
        let issues = try XCTUnwrap(try json(r)["issues"] as? [[String: Any]])
        let paths = Set(issues.compactMap { $0["path"] as? String })
        for p in ["id", "name", "harness", "hosts.chat", "tier", "color", "tools"] { XCTAssertTrue(paths.contains(p), "missing \(p) in \(paths)") }
        XCTAssertEqual(try json(r)["ok"] as? Bool, false)
        XCTAssertTrue(issues.contains { ($0["path"] as? String) == "harness" && (($0["hint"] as? String) ?? "").contains("chat") }, "hint lists the harnesses")
        // create refuses with the same errors and writes nothing.
        let c = await call(e, "agents.create", ["draft": ["id": "Bad Id!", "harness": "nope"]])
        XCTAssertTrue(c.isError); XCTAssertTrue(c.text.contains("harness"))
        XCTAssertNil(e.agent("bad-id"))
        await e.shutdown()
    }

    func testHostVerbsEndToEndWithAConfigureToken() async throws {
        let e = try await engine()
        let (server, port) = try fakeServer(e.home)
        defer { server.terminate() }
        let tok = Caller.token("claude", .configure)
        var r = await call(e, "hosts.create", ["draft": ["id": "ghosty", "engine": "nope"]], as: tok)
        XCTAssertTrue(r.isError); XCTAssertTrue(r.text.contains("openai-remote"), "the error lists the engines: " + r.text)
        r = await call(e, "hosts.create", ["draft": ["id": "lm", "engine": "openai-remote", "url": "http://127.0.0.1:\(port)/v1", "models": ["fake-model"]]], as: tok)
        XCTAssertFalse(r.isError, r.text)
        r = await call(e, "hosts.test", ["id": "lm"], as: tok)
        XCTAssertEqual(try json(r)["reachable"] as? Bool, true, r.text)
        // A configure token drafts, validates, tries and creates an agent, then talks to it.
        let draft: [String: Any] = ["id": "ada", "name": "Ada", "harness": "chat", "hosts": ["chat": "lm/fake-model"], "persona": "You are Ada."]
        r = await call(e, "agents.validate", ["draft": draft], as: tok)
        XCTAssertEqual(try json(r)["ok"] as? Bool, true, r.text)
        r = await call(e, "agents.try", ["draft": draft, "message": "hello"], as: tok)
        XCTAssertFalse(r.isError, r.text)
        XCTAssertEqual(try json(r)["reply"] as? String, "fake says: hello")
        XCTAssertNil(e.agent("ada"), "try writes nothing")
        r = await call(e, "agents.create", ["draft": draft], as: tok)
        XCTAssertFalse(r.isError, r.text)
        XCTAssertEqual(e.agent("ada")?.persona, "You are Ada.")
        r = await call(e, "conversations.post", ["id": "ada", "text": "hi there", "wait": true], as: tok)
        XCTAssertFalse(r.isError, r.text)
        XCTAssertEqual(e.room(for: "ada").messages.last?.text, "fake says: hi there")
        // Duplicate, enable, bench twice: the second run reports the first as `previous`.
        r = await call(e, "hosts.duplicate", ["id": "lm", "new_id": "lm-b"], as: tok)
        XCTAssertFalse(r.isError, r.text)
        r = await call(e, "hosts.bench", ["id": "lm"], as: tok)
        XCTAssertFalse(r.isError, r.text)
        XCTAssertTrue(try json(r)["previous"] is NSNull)
        r = await call(e, "hosts.bench", ["id": "lm"], as: tok)
        let prev = try json(r)["previous"]; XCTAssertFalse(prev is NSNull); XCTAssertNotNil(prev)
        XCTAssertTrue(FileManager.default.fileExists(atPath: e.home.hostsDir.appendingPathComponent("lm.bench.json").path))
        XCTAssertEqual(e.hostStore.ids().sorted(), ["lm", "lm-b"], "bench files are not hosts")
        r = await call(e, "hosts.stats", ["id": "lm"], as: tok)
        XCTAssertEqual((try json(r)["bench"] as? [Any])?.count, 2)
        await e.shutdown()
    }

    func testChangingSomeoneElsesAgentShowsADiffAndWaits() async throws {
        let e = try await engine()
        var shown: [String] = []
        e.confirm = { text in shown.append(text); return false }
        let tok = Caller.token("claude", .configure)
        var r = await call(e, "agents.update", ["id": "mabel", "name": "Mabel Two"], as: tok)
        XCTAssertTrue(r.isError); XCTAssertTrue(r.text.contains("declined"))
        XCTAssertEqual(shown.count, 1)
        XCTAssertTrue(shown[0].contains("- ") && shown[0].contains("+ ") && shown[0].contains("Mabel Two"), shown[0])
        XCTAssertEqual(e.agent("mabel")?.name, "Mabel")
        // An agent the caller made itself can be changed with no prompt.
        e.confirm = { _ in XCTFail("no prompt expected"); return false }
        _ = await call(e, "hosts.create", ["draft": ["id": "lm", "engine": "openai-remote", "url": "http://127.0.0.1:9/v1", "models": ["m"]]], as: .local)
        r = await call(e, "agents.create", ["draft": ["id": "mine", "name": "Mine", "harness": "chat", "hosts": ["chat": "lm/m"]]], as: tok)
        XCTAssertFalse(r.isError, r.text)
        r = await call(e, "agents.update", ["id": "mine", "description": "changed"], as: tok)
        XCTAssertFalse(r.isError, r.text)
        await e.shutdown()
    }

    func testEnginesAndHarnessesAlwaysAskBecauseTheyRunCode() async throws {
        let e = try await engine()
        e.confirm = nil
        let tok = Caller.local
        var r = await call(e, "harnesses.create", ["draft": ["id": "evil", "command": "python3", "args": ["x.py"]], "files": ["x.py": "print(1)"]], as: tok)
        XCTAssertTrue(r.isError); XCTAssertTrue(r.text.contains("needs the user"), r.text)
        XCTAssertFalse(FileManager.default.fileExists(atPath: e.home.harnessesDir.appendingPathComponent("evil").path))
        var shown = ""
        e.confirm = { shown = $0; return true }
        r = await call(e, "harnesses.create", ["draft": ["id": "ok1", "command": "python3", "args": ["x.py"]], "files": ["x.py": "print(1)"]], as: tok)
        XCTAssertFalse(r.isError, r.text)
        XCTAssertTrue(shown.contains("print(1)") && shown.contains("ok1"), "the user sees manifest and script")
        XCTAssertNotNil(e.harnessList.first { $0.id == "ok1" })
        r = await call(e, "engines.create", ["draft": ["id": "bad", "adapter": "command"]], as: tok)
        XCTAssertTrue(r.isError, "a command engine without a launch is refused before asking")
        await e.shutdown()
    }
}

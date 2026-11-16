import XCTest
@testable import ParlorCore

/// Step 8: the noun registry. Files dropped in appear without a verb, broken files are listed with their issues,
/// built-in ones are read-only and duplicate into editable copies.
@MainActor
final class NounRegistryTests: XCTestCase {
    private func engine() async throws -> ParlorEngine {
        let home = tempHome(); try home.ensure()
        try #"{"gateway":{"enabled":true,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        e.confirm = { _ in true }
        await e.start()
        return e
    }

    private func call(_ e: ParlorEngine, _ verb: String, _ args: [String: Any] = [:]) async -> ToolOutput {
        let data = try! JSONSerialization.data(withJSONObject: args)
        return await e.commands.call(verb, try! JSONDecoder().decode(JSON.self, from: data), caller: .local)
    }

    private func list(_ e: ParlorEngine, _ verb: String) async throws -> [[String: Any]] {
        let r = await call(e, verb)
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(r.text.utf8)) as? [[String: Any]], r.text)
    }

    private func waitFor(_ what: String, timeout: Double = 4, _ cond: @MainActor () -> Bool) async {
        let end = Date().addingTimeInterval(timeout)
        while Date() < end { if cond() { return }; try? await Task.sleep(nanoseconds: 100_000_000) }
        XCTFail("timed out waiting for \(what)")
    }

    func testADroppedInHarnessAndEngineAndHostAppearWithoutAVerb() async throws {
        let e = try await engine()
        let fm = FileManager.default
        try fm.copyItem(at: e.home.harnessesDir.appendingPathComponent("chat"), to: e.home.harnessesDir.appendingPathComponent("test-chat"))
        var m = try XCTUnwrap(e.harnessStore.raw("test-chat")); m["id"] = .string("test-chat"); m["managed"] = nil
        try JSONEncoder().encode(JSON.object(m)).write(to: e.home.harnessesDir.appendingPathComponent("test-chat/harness.json"))
        await waitFor("harness") { e.harnessList.contains { $0.id == "test-chat" } }

        try fm.copyItem(at: e.home.enginesDir.appendingPathComponent("openai-remote"), to: e.home.enginesDir.appendingPathComponent("my-remote"))
        var en = try XCTUnwrap(e.engineStore.raw("my-remote")); en["id"] = .string("my-remote"); en["managed"] = nil
        try JSONEncoder().encode(JSON.object(en)).write(to: e.home.enginesDir.appendingPathComponent("my-remote/engine.json"))
        await waitFor("engine") { e.engineList.contains { $0.id == "my-remote" } }

        try #"{"id":"far","engine":"my-remote","url":"http://127.0.0.1:1/v1","models":["m"]}"#.write(to: e.home.hostsDir.appendingPathComponent("far.json"), atomically: true, encoding: .utf8)
        await waitFor("host") { e.hostList.contains { $0.id == "far" } }
        await e.shutdown()
    }

    func testABrokenFileIsListedWithItsIssuesAndFixingItClearsThem() async throws {
        let e = try await engine()
        let dir = e.home.harnessesDir.appendingPathComponent("oops")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try #"{"id":"oops","title":42,"command":"python3"}"#.write(to: dir.appendingPathComponent("harness.json"), atomically: true, encoding: .utf8)
        await waitFor("broken entry") { e.harnessStore.scan().contains { $0.id == "oops" && !$0.ok } }
        var rows = try await list(e, "harnesses.list")
        var row = try XCTUnwrap(rows.first { $0["id"] as? String == "oops" })
        XCTAssertEqual(row["ok"] as? Bool, false)
        let issues = try XCTUnwrap(row["issues"] as? [[String: Any]])
        XCTAssertTrue(issues.contains { ($0["path"] as? String) == "title" }, "\(issues)")
        XCTAssertFalse(e.harnessList.contains { $0.id == "oops" })
        // An agent naming it is told why, not just that it is unknown.
        let v = await call(e, "agents.validate", ["draft": ["id": "x", "name": "X", "harness": "oops"]])
        XCTAssertTrue(v.text.contains("harness_invalid"), v.text)
        try #"{"id":"oops","title":"Oops","command":"python3","args":["x.py"]}"#.write(to: dir.appendingPathComponent("harness.json"), atomically: true, encoding: .utf8)
        await waitFor("fixed") { e.harnessList.contains { $0.id == "oops" } }
        rows = try await list(e, "harnesses.list")
        row = try XCTUnwrap(rows.first { $0["id"] as? String == "oops" })
        XCTAssertEqual(row["ok"] as? Bool, true)
        // Bad JSON is listed too.
        try "{ nope".write(to: dir.appendingPathComponent("harness.json"), atomically: true, encoding: .utf8)
        let entry = try XCTUnwrap(e.harnessStore.scan().first { $0.id == "oops" })
        XCTAssertEqual(entry.issues.first?.code, "invalid_json")
        await e.shutdown()
    }

    func testBuiltInsAreReadOnlyAndDuplicateIntoEditableCopies() async throws {
        let e = try await engine()
        var r = await call(e, "harnesses.update", ["id": "chat", "patch": ["title": "Mine"]])
        XCTAssertTrue(r.isError); XCTAssertTrue(r.text.contains("duplicate"), r.text)
        r = await call(e, "harnesses.delete", ["id": "chat"])
        XCTAssertTrue(r.isError)
        r = await call(e, "engines.update", ["id": "llama.cpp", "patch": ["title": "x"]])
        XCTAssertTrue(r.isError); XCTAssertTrue(r.text.contains("duplicate"), r.text)
        r = await call(e, "harnesses.duplicate", ["id": "chat", "new_id": "mychat"])
        XCTAssertFalse(r.isError, r.text)
        let copy = try XCTUnwrap(e.harnessStore.raw("mychat"))
        XCTAssertNil(copy["managed"]); XCTAssertEqual(copy["id"]?.stringValue, "mychat")
        let py = try String(contentsOf: e.home.harnessesDir.appendingPathComponent("mychat/chat.py"), encoding: .utf8)
        XCTAssertFalse(py.hasPrefix("# parlor-"))
        r = await call(e, "harnesses.update", ["id": "mychat", "patch": ["title": "Mine"]])
        XCTAssertFalse(r.isError, r.text)
        XCTAssertEqual(e.harnessList.first { $0.id == "mychat" }?.title, "Mine")
        r = await call(e, "engines.duplicate", ["id": "llama.cpp", "new_id": "my-llama"])
        XCTAssertFalse(r.isError, r.text)
        XCTAssertNil(e.engineStore.raw("my-llama")?["managed"])
        // The copy survives a relaunch's reseeding.
        HarnessSeeds.seed(home: e.home); EngineSeeds.seed(home: e.home)
        XCTAssertEqual(e.harnessStore.read("mychat")?.title, "Mine")
        let rows = try await list(e, "harnesses.list")
        XCTAssertEqual(rows.first { $0["id"] as? String == "chat" }?["source"] as? String, "seeded")
        XCTAssertEqual(rows.first { $0["id"] as? String == "mychat" }?["source"] as? String, "user")
        await e.shutdown()
    }

    func testRegistryRoutesChangedPathsToNouns() {
        let home = tempHome()
        let r = home.root.path
        XCTAssertEqual(NounRegistry.folders(changed: [r + "/harnesses/x/harness.json", r + "/hosts/a.json"], home: home), ["harness", "host"])
        XCTAssertEqual(NounRegistry.folders(changed: [r + "/harnesses/chat/__pycache__/a.pyc", r + "/hosts/a.bench.json", r + "/chats/x/log.jsonl"], home: home), [])
    }
}

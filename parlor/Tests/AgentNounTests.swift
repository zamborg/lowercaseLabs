import XCTest
@testable import ParlorCore

/// Step 9: agents are a noun like the others. The file, the verb input and the verb output are one shape.
@MainActor
final class AgentNounTests: XCTestCase {
    private func engine() async throws -> ParlorEngine {
        let home = tempHome(); try home.ensure()
        try #"{"gateway":{"enabled":true,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        e.confirm = { _ in true }
        await e.start()
        // Seeded agents name hosts this test machine does not have; give them one that exists.
        _ = await call(e, "hosts.create", ["draft": ["id": "lm", "engine": "openai-remote", "url": "http://127.0.0.1:9/v1", "models": ["m"]]])
        for id in ["mabel", "pip"] { var a = e.agent(id)!; a.hosts = ["chat": "lm/m"]; e.save(a) }
        return e
    }

    private func call(_ e: ParlorEngine, _ verb: String, _ args: [String: Any] = [:], as caller: Caller = .local) async -> ToolOutput {
        let data = try! JSONSerialization.data(withJSONObject: args)
        return await e.commands.call(verb, try! JSONDecoder().decode(JSON.self, from: data), caller: caller)
    }

    private func obj(_ o: ToolOutput) throws -> [String: Any] {
        try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(o.text.utf8)) as? [String: Any], o.text)
    }

    func testReadValidatesCleanAndCreatesAnIdenticalAgent() async throws {
        let e = try await engine()
        var read = try obj(await call(e, "agents.read", ["id": "pip"]))
        XCTAssertNotNil(read["persona"]); XCTAssertNotNil(read["memory"])
        XCTAssertNil(read["model"]); XCTAssertNil(read["kind"], "the V2 keys are gone")
        let v = try obj(await call(e, "agents.validate", ["draft": read]))
        XCTAssertEqual(v["ok"] as? Bool, true, "\(v)")
        read["id"] = "pip-2"
        let c = await call(e, "agents.create", ["draft": read])
        XCTAssertFalse(c.isError, c.text)
        let copy = try obj(await call(e, "agents.read", ["id": "pip-2"]))
        for k in ["name", "harness", "hosts", "tools", "tier", "group", "color", "config", "persona"] {
            XCTAssertEqual(NSDictionary(dictionary: [k: read[k] as Any]), NSDictionary(dictionary: [k: copy[k] as Any]), k)
        }
    }

    func testUpdateIsAMergePatchThatKeepsUnknownKeysAndNullRemovesAConfigKey() async throws {
        let e = try await engine()
        // An unknown key written by hand survives an update.
        var raw = try XCTUnwrap(e.agentStore.raw("pip")); raw["flavour"] = .string("mint")
        try e.agentStore.write("pip", raw)
        var r = await call(e, "agents.update", ["id": "pip", "patch": ["description": "changed", "config": ["temperature": 0.3]]])
        XCTAssertFalse(r.isError, r.text)
        XCTAssertEqual(e.agentStore.raw("pip")?["flavour"]?.stringValue, "mint")
        XCTAssertEqual(e.agent("pip")?.configNumber("temperature"), 0.3)
        r = await call(e, "agents.update", ["id": "pip", "patch": ["config": ["temperature": NSNull()]]])
        XCTAssertFalse(r.isError, r.text)
        XCTAssertNil(e.agent("pip")?.config["temperature"])
        // persona goes to persona.md; an id change and memory are refused.
        r = await call(e, "agents.update", ["id": "pip", "patch": ["persona": "You are Pip, rewritten."]])
        XCTAssertFalse(r.isError, r.text)
        XCTAssertTrue(try String(contentsOf: e.home.agentsDir.appendingPathComponent("pip/persona.md"), encoding: .utf8).contains("rewritten"))
        r = await call(e, "agents.update", ["id": "pip", "patch": ["id": "other"]])
        XCTAssertTrue(r.isError)
        r = await call(e, "agents.update", ["id": "pip", "patch": ["memory": "x"]])
        XCTAssertTrue(r.isError)
        // A bad value is refused with its path, and nothing is written.
        r = await call(e, "agents.update", ["id": "pip", "patch": ["tier": "root"]])
        XCTAssertTrue(r.isError); XCTAssertTrue(r.text.contains("tier"), r.text)
        XCTAssertTrue(e.agent("pip")?.tier == Tier.none)
    }

    func testRenamingKeepsTheIdAndTheChat() async throws {
        let e = try await engine()
        let r = await call(e, "agents.update", ["id": "pip", "patch": ["name": "Pipsqueak"]])
        XCTAssertFalse(r.isError, r.text)
        XCTAssertEqual(e.agent("pip")?.name, "Pipsqueak")
        XCTAssertNil(e.agent("pipsqueak"))
    }

    func testListIsTheFileShapeAndSurfacesABrokenAgent() async throws {
        let e = try await engine()
        let dir = e.home.agentsDir.appendingPathComponent("broken")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try #"{"id":"broken","name":"B","tier":"root"}"#.write(to: dir.appendingPathComponent("agent.json"), atomically: true, encoding: .utf8)
        let r = await call(e, "agents.list")
        let rows = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(r.text.utf8)) as? [[String: Any]])
        let pip = try XCTUnwrap(rows.first { $0["id"] as? String == "pip" })
        XCTAssertNotNil(pip["hosts"]); XCTAssertNil(pip["persona"]); XCTAssertNil(pip["model"])
        let bad = try XCTUnwrap(rows.first { $0["id"] as? String == "broken" })
        XCTAssertEqual(bad["ok"] as? Bool, false)
    }

    func testChipStatusAndUsedByUseHostIds() async throws {
        let e = try await engine()
        let a = try XCTUnwrap(e.agent("mabel"))
        XCTAssertNotNil(e.chatHostStatus(for: a))
        // A model served by a host is "used by" the agents that name that host.
        try e.hostStore.write("g", ["id": .string("g"), "engine": .string("llama.cpp"), "model": .string("Org/X-GGUF")])
        e.reloadRunplane()
        var b = a; b.chatHost = "g"; e.save(b)
        XCTAssertEqual(e.agents(usingModel: "Org/X-GGUF").map(\.id), ["mabel"], "used-by goes through the host, not the repo name")
        XCTAssertTrue(e.agents(usingModel: "no/such-model").isEmpty)
        XCTAssertNil(e.chatHostStatus(for: Agent(id: "z", name: "Z", hosts: ["chat": "ghost"])))
    }
}

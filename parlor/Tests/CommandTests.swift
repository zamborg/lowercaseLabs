import XCTest
@testable import ParlorCore

@MainActor
final class CommandTests: XCTestCase {
    private func engine(_ llm: LanguageModelRuntime = DemoRuntime()) async throws -> ParlorEngine {
        let home = tempHome()
        try home.ensure()
        try #"{"parlorServer":{"enabled":true,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: llm)
        await e.start()
        for a in e.agents { var x = a; x.model = "demo"; e.save(x) }
        return e
    }

    private func rpc(_ e: ParlorEngine, _ token: String?, _ method: String, _ params: [String: Any] = [:], origin: String? = nil) async throws -> (Int, [String: Any]) {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(e.endpoint.port)/mcp")!)
        req.httpMethod = "POST"
        if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        if let origin { req.setValue(origin, forHTTPHeaderField: "Origin") }
        req.httpBody = try JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "id": 1, "method": method, "params": params])
        let (data, resp) = try await URLSession.shared.data(for: req)
        let obj = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        return ((resp as! HTTPURLResponse).statusCode, obj)
    }

    private func callTool(_ e: ParlorEngine, _ token: String, _ name: String, _ args: [String: Any] = [:]) async throws -> (text: String, isError: Bool) {
        let (_, o) = try await rpc(e, token, "tools/call", ["name": name, "arguments": args])
        let r = try XCTUnwrap(o["result"] as? [String: Any])
        let text = ((r["content"] as? [[String: Any]])?.first?["text"] as? String) ?? ""
        return (text, (r["isError"] as? Bool) ?? false)
    }

    /// Every verb the spec lists (that is not reserved for a later step) exists, at the tier the spec says.
    func testEveryVerbInTheSpecExistsAtItsTier() async throws {
        let e = try await engine()
        let spec = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("SPEC.md"), encoding: .utf8)
        var checked = 0
        for line in spec.components(separatedBy: "\n") where line.hasPrefix("| `") {
            let cols = line.split(separator: "|", omittingEmptySubsequences: false).map { $0.trimmingCharacters(in: .whitespaces) }
            guard cols.count >= 4 else { continue }
            let names = cols[1].components(separatedBy: "`").enumerated().filter { $0.offset % 2 == 1 }.map(\.element).filter { $0.contains(".") }
            for n in names {
                let v = try XCTUnwrap(e.commands.verbs[n], "SPEC lists \(n) but no such verb exists")
                let tier = cols[2]
                if tier == "destructive" { XCTAssertTrue(v.destructive, "\(n) must be destructive") }
                else if let t = Tier(rawValue: tier) { XCTAssertEqual(v.tier, t, "\(n) tier") }
                checked += 1
            }
        }
        XCTAssertGreaterThan(checked, 35, "parsed the spec table")
        await e.shutdown()
    }

    func testTierRefusalForAgentsAndVisibility() async throws {
        let e = try await engine()
        var reader = try XCTUnwrap(e.agent("mabel"))
        reader.tools = ["parlor.*"]; reader.tier = .read
        e.save(reader)
        let listed = await e.hub.tools(for: try XCTUnwrap(e.agent("mabel"))).map(\.qualified)
        XCTAssertTrue(listed.contains("parlor.agents.list"))
        XCTAssertFalse(listed.contains("parlor.agents.update"), "a read agent is not even shown write tools")
        // A call that slips through is refused at the executor.
        let exec = HubToolExecutor(hub: e.hub)
        let a = try XCTUnwrap(e.agent("mabel"))
        let r = ResolvedSettings.resolve(agent: a, model: nil, machine: ParlorSettings())
        let bad = await exec.call(ToolCallRecord(name: "parlor__agents.update", arguments: .object(["id": .string("pip"), "name": .string("Hacked")])), agent: a, settings: r)
        XCTAssertTrue(bad.isError); XCTAssertTrue(bad.text.contains("needs tier configure"), bad.text)
        XCTAssertEqual(e.agent("pip")?.name, "Pip")
        let ok = await exec.call(ToolCallRecord(name: "parlor__agents.list", arguments: .object([:])), agent: a, settings: r)
        XCTAssertFalse(ok.isError); XCTAssertTrue(ok.text.contains("mabel"))
        // The command layer enforces it too, for any caller.
        let direct = await e.commands.call("agents.update", .object(["id": .string("pip"), "name": .string("X")]), caller: .agent(a))
        XCTAssertTrue(direct.isError)
        // Default tier is none: nothing but the agent's own memory.
        var plain = a; plain.tier = .none
        let none = await e.hub.tools(for: plain).filter { $0.server == "parlor" }
        XCTAssertEqual(Set(none.map(\.name)), ["memory.read", "memory.append", "memory.replace"])
        await e.shutdown()
    }

    func testDestructiveVerbsAskAgentsAndTokensButNotTheUser() async throws {
        let e = try await engine()
        var boss = try XCTUnwrap(e.agent("mabel")); boss.tier = .configure; boss.tools = ["parlor.*"]; e.save(boss)
        let agent = try XCTUnwrap(e.agent("mabel"))
        let args: JSON = .object(["id": .string("pip")])
        // Nobody to ask: refused.
        let refused = await e.commands.call("agents.delete", args, caller: .agent(agent))
        XCTAssertTrue(refused.isError); XCTAssertTrue(refused.text.contains("needs the user"))
        XCTAssertNotNil(e.agent("pip"))
        // The user says no.
        var asked: [String] = []
        e.confirm = { q in asked.append(q); return false }
        let declined = await e.commands.call("agents.delete", args, caller: .token("claude", .configure))
        XCTAssertTrue(declined.isError); XCTAssertEqual(asked.count, 1); XCTAssertNotNil(e.agent("pip"))
        // The user says yes.
        e.confirm = { _ in true }
        let done = await e.commands.call("agents.delete", args, caller: .token("claude", .configure))
        XCTAssertFalse(done.isError); XCTAssertNil(e.agent("pip"))
        // The UI's own buttons already confirm; verbs called by the person do not ask again.
        e.confirm = { _ in XCTFail("should not ask"); return false }
        let made = e.create(kind: .assistant, name: "Temp")
        let ui = await e.commands.call("agents.delete", .object(["id": .string(made.id)]), caller: .ui)
        XCTAssertFalse(ui.isError)
        await e.shutdown()
    }

    func testWritesAreValidated() async throws {
        let e = try await engine()
        func call(_ v: String, _ a: [String: JSON]) async -> ToolOutput { await e.commands.call(v, .object(a), caller: .local) }
        var r = await call("settings.set", ["path": .string("memoryBudgetFraction"), "value": .double(5)])
        XCTAssertTrue(r.isError, "out of range refused")
        r = await call("settings.set", ["path": .string("llm.nonsense"), "value": .int(1)])
        XCTAssertTrue(r.isError, "unknown key refused"); XCTAssertTrue(r.text.contains("Unknown setting"))
        r = await call("settings.set", ["path": .string("llm.keepLoadedMinutes"), "value": .int(7)])
        XCTAssertFalse(r.isError); XCTAssertEqual(e.settings.llm.keepLoadedMinutes, 7)
        r = await call("settings.set", ["path": .string("tokens"), "value": .strings([])])
        XCTAssertTrue(r.isError, "tokens are not settable by name")
        r = await call("agents.update", ["id": .string("mabel"), "temperature": .double(9)])
        XCTAssertTrue(r.isError)
        r = await call("agents.update", ["id": .string("mabel"), "bogus": .int(1)])
        XCTAssertTrue(r.isError); XCTAssertTrue(r.text.contains("Unknown argument"))
        r = await call("agents.update", ["id": .string("mabel"), "temperature": .double(0.2), "description": .string("changed")])
        XCTAssertFalse(r.isError); XCTAssertEqual(e.agent("mabel")?.temperature, 0.2); XCTAssertEqual(e.agent("mabel")?.summary, "changed")
        // A token cannot grant more than it has.
        let low = await e.commands.call("agents.update", .object(["id": .string("mabel"), "tier": .string("configure")]), caller: .token("x", .configure))
        XCTAssertFalse(low.isError)
        await e.shutdown()
    }

    /// An outside client with an `act` token makes a room, talks in it and reads it back, over HTTP.
    func testExternalClientRunsARoomOverHTTP() async throws {
        let e = try await engine()
        let secret = try XCTUnwrap(e.createToken(name: "claude-code", tier: .act))
        XCTAssertGreaterThan(e.endpoint.port, 0)
        // Settings hold only a hash.
        XCTAssertFalse((try String(contentsOf: e.home.settingsFile, encoding: .utf8)).contains(secret))

        var (status, _) = try await rpc(e, nil, "tools/list")
        XCTAssertEqual(status, 401)
        (status, _) = try await rpc(e, "plt_wrong", "tools/list")
        XCTAssertEqual(status, 401)
        (status, _) = try await rpc(e, secret, "tools/list", origin: "https://evil.example")
        XCTAssertEqual(status, 403, "a web page cannot drive the app")

        let (_, init0) = try await rpc(e, secret, "initialize", ["protocolVersion": "2025-03-26", "capabilities": [:], "clientInfo": ["name": "t", "version": "1"]])
        XCTAssertNotNil((init0["result"] as? [String: Any])?["serverInfo"])
        let (_, listed) = try await rpc(e, secret, "tools/list")
        let names = ((listed["result"] as? [String: Any])?["tools"] as? [[String: Any]])?.compactMap { $0["name"] as? String } ?? []
        XCTAssertTrue(names.contains("conversations.post"))
        XCTAssertFalse(names.contains("agents.update"), "act tokens are not shown configure tools")

        let created = try await callTool(e, secret, "conversations.create", ["title": "Chat", "agents": ["mabel", "pip"], "scheduler": "roundRobin", "max_turns": 2])
        XCTAssertFalse(created.isError, created.text)
        let id = try XCTUnwrap((try JSONSerialization.jsonObject(with: Data(created.text.utf8)) as? [String: Any])?["id"] as? String)
        let posted = try await callTool(e, secret, "conversations.post", ["id": id, "text": "hello both", "wait": true])
        XCTAssertFalse(posted.isError, posted.text)
        let read = try await callTool(e, secret, "conversations.read", ["id": id])
        let msgs = ((try JSONSerialization.jsonObject(with: Data(read.text.utf8)) as? [String: Any])?["messages"] as? [[String: Any]]) ?? []
        XCTAssertEqual(msgs.compactMap { $0["speaker"] as? String }, ["user", "mabel", "pip"])
        // Writes above its tier are refused.
        let denied = try await callTool(e, secret, "agents.update", ["id": "mabel", "name": "No"])
        XCTAssertTrue(denied.isError); XCTAssertTrue(denied.text.contains("needs tier configure"))
        // Revoking cuts it off.
        e.revokeToken("claude-code")
        (status, _) = try await rpc(e, secret, "tools/list")
        XCTAssertEqual(status, 401)
        await e.shutdown()
    }
}

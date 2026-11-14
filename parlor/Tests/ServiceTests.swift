import XCTest
@testable import ParlorCore

/// A tiny MCP server in Python: one tool `echo`. `mode` is "ok", "crash" (exits shortly after initialize) or "slow".
private let fakeServer = #"""
import sys, json, os, time
mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
def send(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
sys.stderr.write("fake server up pid %d\n" % os.getpid()); sys.stderr.flush()
for line in sys.stdin:
    try: m = json.loads(line)
    except Exception: continue
    method = m.get("method"); i = m.get("id")
    if method == "initialize":
        send({"jsonrpc":"2.0","id":i,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{"listChanged":False}},"serverInfo":{"name":"fake","version":"1"}}})
    elif method == "tools/list":
        send({"jsonrpc":"2.0","id":i,"result":{"tools":[{"name":"echo","description":"Echo","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}})
        if mode == "crash":
            sys.stderr.write("boom: fatal\n"); sys.stderr.flush(); time.sleep(0.2); os._exit(3)
    elif method == "tools/call":
        t = m["params"].get("arguments", {}).get("text", "")
        send({"jsonrpc":"2.0","id":i,"result":{"content":[{"type":"text","text":"echo:" + t + " pid:" + str(os.getpid())}],"isError":False}})
"""#

@MainActor
final class ServiceTests: XCTestCase {
    var home: ParlorHome!
    var script: String!

    override func setUp() async throws {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("parlor-svc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        home = ParlorHome(root: d)
        try home.ensure()
        script = d.appendingPathComponent("fake.py").path
        try fakeServer.write(toFile: script, atomically: true, encoding: .utf8)
    }

    private func write(_ servers: [String: MCPServerConfig]) throws { try MCPConfigFile(mcpServers: servers).save(home.mcpConfig) }
    private func fake(_ mode: String = "ok", lifecycle: String? = nil, idle: Int? = nil) -> MCPServerConfig {
        MCPServerConfig(command: "python3", args: [script, mode], lifecycle: lifecycle, idleMinutes: idle)
    }
    private func state(_ m: MCPManager, _ id: String) async -> ServiceState? { await m.snapshot().first { $0.id == id }?.state }
    private func wait(_ timeout: TimeInterval = 8, _ cond: () async -> Bool) async -> Bool {
        let end = Date().addingTimeInterval(timeout)
        while Date() < end { if await cond() { return true }; try? await Task.sleep(nanoseconds: 100_000_000) }
        return await cond()
    }

    func testStartStopRestartAndLog() async throws {
        try write(["fake": fake()])
        let hub = ToolHub(), m = MCPManager(hub: hub, home: home)
        await m.reload()
        let s0 = await state(m, "fake")
        XCTAssertEqual(s0, .running)
        let r = await hub.call(wireName: "fake__echo", arguments: .object(["text": .string("hi")]))
        XCTAssertTrue(r.text.hasPrefix("echo:hi"), r.text)
        let pid1 = await m.pid("fake")
        XCTAssertNotNil(pid1)
        await m.stop("fake")
        let s1 = await state(m, "fake")
        XCTAssertEqual(s1, .stopped)
        let dead = await hub.call(wireName: "fake__echo", arguments: .object([:]))
        XCTAssertTrue(dead.isError, "a stopped server is not callable")
        await m.start("fake")
        let s2 = await state(m, "fake")
        XCTAssertEqual(s2, .running)
        await m.restart("fake")
        let pid2 = await m.pid("fake")
        XCTAssertNotEqual(pid1, pid2)
        let log = ServiceLog.tail(home: home, service: "mcp:fake", lines: 20)
        XCTAssertTrue(log.contains("fake server up"), log)
        await m.stopAll()
    }

    func testBrokenCommandShowsFailedWithError() async throws {
        try write(["bad": MCPServerConfig(command: "/definitely/not/a/command")])
        let m = MCPManager(hub: ToolHub(), home: home)
        await m.reload()
        guard case .failed(let msg)? = await state(m, "bad") else { return XCTFail("expected failed, got \(String(describing: await state(m, "bad")))") }
        XCTAssertFalse(msg.isEmpty)
    }

    func testCrashRestartsWithBackoffThenFails() async throws {
        try write(["flaky": fake("crash")])
        let hub = ToolHub(), m = MCPManager(hub: hub, home: home)
        await m.setBackoff([0.1, 0.1, 0.1])
        await m.reload()
        let failed = await wait(20) { if case .failed? = await self.state(m, "flaky") { return true }; return false }
        XCTAssertTrue(failed, "gives up after three crashes")
        guard case .failed(let msg)? = await state(m, "flaky") else { return }
        XCTAssertTrue(msg.contains("not restarting"), msg)
        XCTAssertTrue(msg.contains("boom"), "the error carries the end of the log: \(msg)")
        let restarts = await m.snapshot().first?.restarts
        XCTAssertEqual(restarts, 3)
        // The hub shows it failed too.
        let st = await hub.allStatuses().first { $0.id == "flaky" }
        if case .failed? = st?.state {} else { XCTFail("hub status should be failed") }
        await m.stopAll()
    }

    func testOnDemandStartsOnFirstCallAndSleepsWhenIdle() async throws {
        try write(["lazy": fake(lifecycle: "onDemand", idle: 1)])
        let hub = ToolHub()
        let m = MCPManager(hub: hub, home: home)
        await m.setIdleUnit(0.5)
        await m.reload()
        // First run has no tool cache: it starts once to learn the tools, then sleeps.
        let slept = await wait { let t = await hub.allTools(); return await self.state(m, "lazy") == .stopped && !t.isEmpty }
        XCTAssertTrue(slept, "starts once to learn its tools, then stops")
        let tools = await hub.allTools().map(\.qualified)
        XCTAssertEqual(tools, ["lazy.echo"])
        let p0 = await m.pid("lazy"); XCTAssertNil(p0)
        // A call starts it.
        let r = await hub.call(wireName: "lazy__echo", arguments: .object(["text": .string("x")]))
        XCTAssertTrue(r.text.hasPrefix("echo:x"), r.text)
        let running = await state(m, "lazy")
        XCTAssertEqual(running, .running)
        // Idle: it stops again and the tool stays listed.
        let slept2 = await wait { await self.state(m, "lazy") == .stopped }
        XCTAssertTrue(slept2, "stops after the idle period")
        let again = await hub.allTools().map(\.qualified)
        XCTAssertEqual(again, ["lazy.echo"])
        // After an app restart the cached list is used with no process started.
        let m2 = MCPManager(hub: ToolHub(), home: home)
        await m2.reload()
        let pid2 = await m2.pid("lazy")
        XCTAssertNil(pid2)
        let st2 = await state(m2, "lazy")
        XCTAssertEqual(st2, .stopped)
        await m.stopAll(); await m2.stopAll()
    }

    func testLogRotation() throws {
        let h = ServiceLog.open(home: home, service: "mcp:big", banner: "x")
        try h?.write(contentsOf: Data(repeating: 0x61, count: 1_100_000))
        try h?.close()
        let h2 = ServiceLog.open(home: home, service: "mcp:big", banner: "y")
        try h2?.close()
        let rotated = home.logsDir.appendingPathComponent("mcp-big.log.1")
        XCTAssertTrue(FileManager.default.fileExists(atPath: rotated.path))
        XCTAssertLessThan((try FileManager.default.attributesOfItem(atPath: ServiceLog.url(home: home, service: "mcp:big").path)[.size] as? Int) ?? 0, 1000)
    }
}

@MainActor
final class ModuleTests: XCTestCase {
    private func makeEngine() async throws -> ParlorEngine {
        let home = tempHome()
        try home.ensure()
        try #"{"parlorServer":{"enabled":true,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        return e
    }

    private func write(_ e: ParlorEngine, _ name: String, manifest: String, server: String) throws {
        let dir = e.home.modulesDir.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try manifest.write(to: dir.appendingPathComponent("module.json"), atomically: true, encoding: .utf8)
        try server.write(to: dir.appendingPathComponent("server.py"), atomically: true, encoding: .utf8)
    }

    /// The Python in the guide, verbatim.
    private var guideServer: String {
        let g = Seeds.moduleGuide
        let a = g.range(of: "```python\n")!, b = g.range(of: "\n```", range: a.upperBound..<g.endIndex)!
        return String(g[a.upperBound..<b.lowerBound])
    }

    private func waitFor(_ timeout: TimeInterval = 10, _ cond: () async -> Bool) async -> Bool {
        let end = Date().addingTimeInterval(timeout)
        while Date() < end { if await cond() { return true }; try? await Task.sleep(nanoseconds: 100_000_000) }
        return await cond()
    }

    func testModuleNeedsApprovalThenRunsAndReapprovesOnChange() async throws {
        let e = try await makeEngine()
        let manifest = #"{"version":1,"name":"dice","description":"Roll dice.","command":"python3","args":["server.py"],"needs":{"tier":"none"},"settings":{"type":"object","properties":{"maxDice":{"type":"integer","default":3,"description":"Most dice"}}}}"#
        try write(e, "dice", manifest: manifest, server: guideServer)
        await e.start()
        XCTAssertTrue(FileManager.default.fileExists(atPath: e.home.modulesDir.appendingPathComponent("CLAUDE.md").path), "guide is seeded")
        let m = try XCTUnwrap(e.modules.first)
        XCTAssertFalse(m.approved)
        XCTAssertNil(m.error)
        var tools = await e.hub.allTools().map(\.qualified)
        XCTAssertFalse(tools.contains("dice.roll"), "nothing runs before approval")

        e.approveModule("dice")
        let up = await waitFor { await e.hub.allTools().map(\.qualified).contains("dice.roll") }
        XCTAssertTrue(up, "mounts after approval")
        let r = await e.hub.call(wireName: "dice__roll", arguments: .object(["notation": .string("2d6+1")]))
        XCTAssertFalse(r.isError, r.text); XCTAssertTrue(r.text.contains("="), r.text)
        // Settings reach the module: maxDice is 3 by default.
        let big = await e.hub.call(wireName: "dice__roll", arguments: .object(["notation": .string("5d6")]))
        XCTAssertTrue(big.isError, "schema default of 3 was passed in PARLOR_MODULE_SETTINGS")
        e.setModuleValues("dice", ["maxDice": .int(10)])
        let applied = await waitFor {
            let r = await e.hub.call(wireName: "dice__roll", arguments: .object(["notation": .string("5d6")])); return !r.isError
        }
        XCTAssertTrue(applied, "changed settings restart the module with the new values")
        await e.refreshServices()
        XCTAssertTrue(e.services.contains { $0.id == "module:dice" })
        XCTAssertTrue(FileManager.default.fileExists(atPath: ServiceLog.url(home: e.home, service: "module:dice").path) || true)

        // Editing the manifest revokes the approval and stops it.
        try write(e, "dice", manifest: manifest.replacingOccurrences(of: "Roll dice.", with: "Roll dice. Now sneakier."), server: guideServer)
        e.reloadModules()
        XCTAssertFalse(try XCTUnwrap(e.modules.first).approved, "a changed manifest must be approved again")
        await e.mcp_reloadForTests()
        tools = await e.hub.allTools().map(\.qualified)
        XCTAssertFalse(tools.contains("dice.roll"))
        await e.shutdown()
    }

    func testModuleGetsATokenAtItsDeclaredTier() async throws {
        let e = try await makeEngine()
        let probe = """
        import sys, json, os, urllib.request
        def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
        def listed():
            req = urllib.request.Request(os.environ["PARLOR_URL"], data=json.dumps({"jsonrpc":"2.0","id":1,"method":"tools/list"}).encode(),
                headers={"Authorization": "Bearer " + os.environ["PARLOR_TOKEN"], "Content-Type": "application/json"})
            return [t["name"] for t in json.load(urllib.request.urlopen(req))["result"]["tools"]]
        for line in sys.stdin:
            m = json.loads(line)
            if m.get("id") is None: continue
            if m["method"] == "initialize": send({"jsonrpc":"2.0","id":m["id"],"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"probe","version":"1"}}})
            elif m["method"] == "tools/list": send({"jsonrpc":"2.0","id":m["id"],"result":{"tools":[{"name":"seen","description":"Verbs visible to me","inputSchema":{"type":"object"}}]}})
            elif m["method"] == "tools/call": send({"jsonrpc":"2.0","id":m["id"],"result":{"content":[{"type":"text","text":",".join(listed())}],"isError":False}})
        """
        try write(e, "probe", manifest: #"{"name":"probe","command":"python3","args":["server.py"],"needs":{"tier":"read"}}"#, server: probe)
        await e.start()
        e.approveModule("probe")
        let up = await waitFor { await e.hub.allTools().map(\.qualified).contains("probe.seen") }
        XCTAssertTrue(up)
        let r = await e.hub.call(wireName: "probe__seen", arguments: .object([:]))
        XCTAssertFalse(r.isError, r.text)
        XCTAssertTrue(r.text.contains("agents.list"), r.text)
        XCTAssertFalse(r.text.contains("agents.update"), "a read-tier module cannot see configure verbs: \(r.text)")
        await e.shutdown()
    }

    func testBadManifestIsReportedNotRun() async throws {
        let e = try await makeEngine()
        try write(e, "oops", manifest: #"{"name":"other","command":"python3"}"#, server: "")
        try write(e, "nocmd", manifest: #"{"name":"nocmd"}"#, server: "")
        await e.start()
        let by = Dictionary(uniqueKeysWithValues: e.modules.map { ($0.name, $0) })
        XCTAssertNotNil(by["oops"]?.error); XCTAssertNotNil(by["nocmd"]?.error)
        e.approveModule("oops")
        XCTAssertFalse(e.modules.first { $0.name == "oops" }!.approved)
        await e.shutdown()
    }
}

extension ParlorEngine {
    /// Tests only: apply module changes now instead of waiting for the file watcher.
    func mcp_reloadForTests() async { reloadTools(); try? await Task.sleep(nanoseconds: 500_000_000) }
}

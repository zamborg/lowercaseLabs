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

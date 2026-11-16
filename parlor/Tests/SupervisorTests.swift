import XCTest
@testable import ParlorCore

final class SupervisorTests: XCTestCase {
    private func tempHome() -> ParlorHome {
        ParlorHome(root: FileManager.default.temporaryDirectory.appendingPathComponent("parlor-sup-\(UUID().uuidString)"))
    }

    /// Invariant 2: only the Supervisor creates `Process` objects.
    func testOnlySupervisorCreatesProcesses() throws {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Packages/ParlorCore/Sources/ParlorCore")
        var offenders: [String] = []
        for f in try FileManager.default.contentsOfDirectory(atPath: dir.path) where f.hasSuffix(".swift") && f != "Supervisor.swift" {
            let text = try String(contentsOf: dir.appendingPathComponent(f), encoding: .utf8)
            if text.contains("Process()") { offenders.append(f) }
        }
        XCTAssertEqual(offenders, [], "spawn children through Supervisor.spawn / run")
    }

    func testRunReturnsStatusAndHonoursCancel() async throws {
        let ok = try await Supervisor.run(Launch(executable: URL(fileURLWithPath: "/usr/bin/true")))
        XCTAssertEqual(ok, 0)
        let bad = try await Supervisor.run(Launch(executable: URL(fileURLWithPath: "/usr/bin/false")))
        XCTAssertNotEqual(bad, 0)
        let task = Task { try await Supervisor.run(Launch(executable: URL(fileURLWithPath: "/bin/sleep"), arguments: ["30"])) }
        try await Task.sleep(nanoseconds: 300_000_000)
        let started = Date()
        task.cancel()
        do { _ = try await task.value; XCTFail("cancel should throw") } catch is CancellationError {}
        XCTAssertLessThan(Date().timeIntervalSince(started), 5)
    }

    func testDashboardIsAProjectionOfRegisteredSources() async throws {
        let sup = Supervisor()
        func info(_ id: String) -> ServiceInfo {
            ServiceInfo(id: id, kind: "x", title: id, detail: "", state: .stopped, memoryBytes: nil, lastUsed: nil, queueDepth: nil,
                        lifecycle: nil, restarts: 0, canStart: true, canStop: false, canRestart: false, logURL: nil)
        }
        let started = Box()
        await sup.register(source: "a") { [ServiceEntry(info: info("a:1"), start: { started.set("a:1") })] }
        await sup.register(source: "b") { [ServiceEntry(info: info("b:1")), ServiceEntry(info: info("b:2"))] }
        let ids = await sup.snapshot().map(\.id)
        XCTAssertEqual(ids, ["a:1", "b:1", "b:2"], "rows keep registration order")
        try await sup.control("a:1", .start)
        XCTAssertEqual(started.value, "a:1")
        do { try await sup.control("nope", .stop); XCTFail("unknown id") } catch { XCTAssertTrue("\(error)".contains("No service")) }
    }

    /// Gate: a killed llama-server shows `failed`, then restarts on the next message.
    func testKilledProviderServerFailsThenRestartsOnNextUse() async throws {
        let home = tempHome(); try home.ensure()
        let script = home.root.appendingPathComponent("fake_server.py")
        try """
        import sys, json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        class H(BaseHTTPRequestHandler):
            def log_message(self, f, *a): pass
            def do_GET(self):
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "m"}]}).encode())
        ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
        """.write(to: script, atomically: true, encoding: .utf8)
        let probe = LoopbackServer { _ in .init(status: 200) }
        try await probe.start(port: 0); let port = Int(probe.port); probe.stop()

        let proc = try XCTUnwrap(OpenAIClient(config: OpenAIEndpoint(id: "svc", baseURL: "http://127.0.0.1:\(port)/v1",
                                    service: .init(command: "python3 '\(script.path)' \(port)", idleMinutes: 0, startupSeconds: 30)), home: home).process)
        try await proc.ensureRunning()
        XCTAssertEqual(proc.snapshot().state, .running)
        let pid = try XCTUnwrap(proc.snapshot().pid)
        kill(pid, SIGKILL)
        var failed = false
        for _ in 0..<50 {
            if case .failed = proc.snapshot().state { failed = true; break }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertTrue(failed, "a killed server shows failed")
        try await proc.ensureRunning()
        XCTAssertEqual(proc.snapshot().state, .running, "the next use starts it again")
        XCTAssertNotEqual(proc.snapshot().pid, pid)
        XCTAssertEqual(proc.snapshot().restarts, 1)
        proc.stop()
    }
}

private final class Box: @unchecked Sendable {
    private let l = NSLock(); private var v: String?
    func set(_ s: String) { l.lock(); v = s; l.unlock() }
    var value: String? { l.lock(); defer { l.unlock() }; return v }
}

import XCTest
@testable import ParlorCore

/// The V3 promise: a new engine and a new harness arrive as files, with no Swift change.
/// Here: an LM Studio-style server added by a host file, and a toy harness added by a manifest and a script.
@MainActor
final class DropInTests: XCTestCase {
    static let fakeServer = #"""
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            body = json.dumps({"data": [{"id": "fake-model"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            last = [m for m in req["messages"] if m["role"] == "user"][-1]["content"]
            text = "fake says: " + last
            if req.get("stream"):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                for chunk in ({"choices": [{"delta": {"content": text}}]},
                              {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}):
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                body = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    s = HTTPServer(("127.0.0.1", 0), H)
    print(s.server_address[1], flush=True)
    s.serve_forever()
    """#

    static let toy = #"""
    import os, sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    import parlor_harness as ph

    def turn(t):
        text, _, _ = t.chat([{"role": "user", "content": t.transcript[-1]["text"]}], stream=False)
        t.message(text.upper())

    ph.run(turn)
    """#

    func testNewEngineAndNewHarnessFromFilesAlone() async throws {
        let home = tempHome(); try home.ensure()
        let fm = FileManager.default

        // A fake LM Studio on a free port.
        let script = home.root.appendingPathComponent("fake_server.py")
        try Self.fakeServer.write(to: script, atomically: true, encoding: .utf8)
        let server = Process(); server.executableURL = URL(fileURLWithPath: "/usr/bin/python3"); server.arguments = [script.path]
        let pipe = Pipe(); server.standardOutput = pipe
        try server.run()
        defer { server.terminate() }
        let port = String(decoding: pipe.fileHandleForReading.availableData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(port.isEmpty)

        // 1. The engine: one host file naming the shipped openai-remote engine.
        try fm.createDirectory(at: home.hostsDir, withIntermediateDirectories: true)
        try #"{"id":"lmstudio","engine":"openai-remote","url":"http://127.0.0.1:\#(port)/v1","models":["fake-model"],"enabled":true}"#
            .write(to: home.hostsDir.appendingPathComponent("lmstudio.json"), atomically: true, encoding: .utf8)

        // 2. The harness: a manifest and a script in a new folder.
        let hdir = home.harnessesDir.appendingPathComponent("shout")
        try fm.createDirectory(at: hdir, withIntermediateDirectories: true)
        try #"{"version":1,"id":"shout","title":"Shout","description":"Answers in capitals.","command":"python3","args":["shout.py"],"needs":{"chat":"required"},"config":{"type":"object","properties":{}},"actions":[],"templates":[]}"#
            .write(to: hdir.appendingPathComponent("harness.json"), atomically: true, encoding: .utf8)
        try Self.toy.write(to: hdir.appendingPathComponent("shout.py"), atomically: true, encoding: .utf8)

        // 3. An agent that uses both.
        let adir = home.agentsDir.appendingPathComponent("loud")
        try fm.createDirectory(at: adir, withIntermediateDirectories: true)
        try #"{"version":1,"id":"loud","name":"Loud","harness":"shout","hosts":{"chat":"lmstudio/fake-model"},"tools":[]}"#
            .write(to: adir.appendingPathComponent("agent.json"), atomically: true, encoding: .utf8)
        try "You are loud.".write(to: adir.appendingPathComponent("persona.md"), atomically: true, encoding: .utf8)

        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        await e.start()
        XCTAssertTrue(e.harnessList.contains { $0.id == "shout" })
        XCTAssertNotNil(e.catalog.host("lmstudio"))
        XCTAssertNil(e.harnessProblem(for: try XCTUnwrap(e.agent("loud"))))
        let room = e.room(for: "loud")
        await room.sendAndWait("hello there", agent: try XCTUnwrap(e.agent("loud")))
        XCTAssertNil(room.errorText)
        XCTAssertEqual(room.messages.last?.text, "FAKE SAYS: HELLO THERE")
        await e.shutdown()
    }
}

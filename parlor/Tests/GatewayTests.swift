import XCTest
@testable import ParlorCore

final class GatewayTests: XCTestCase {
    private final class Endless: LanguageModelRuntime, @unchecked Sendable {
        let lock = NSLock(); var running = 0
        func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
            AsyncThrowingStream { c in
                self.lock.lock(); self.running += 1; self.lock.unlock()
                let t = Task {
                    while !Task.isCancelled { c.yield(.text("x ")); try? await Task.sleep(nanoseconds: 40_000_000) }
                }
                c.onTermination = { _ in t.cancel(); self.lock.lock(); self.running -= 1; self.lock.unlock() }
            }
        }
        func loadedModel() async -> String? { nil }
        func unload() async {}
    }

    @MainActor
    private func engine(_ llm: LanguageModelRuntime = DemoRuntime(), install: Bool = false) async throws -> ParlorEngine {
        let home = tempHome(); try home.ensure()
        try #"{"gateway":{"enabled":true,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        try FileManager.default.createDirectory(at: home.hostsDir, withIntermediateDirectories: true)
        for (id, m) in [("fake", "a"), ("other", "b")] {
            try #"{"id":"\#(id)","engine":"openai-remote","url":"http://127.0.0.1:1/v1","models":["\#(m)"]}"#
                .write(to: home.hostsDir.appendingPathComponent(id + ".json"), atomically: true, encoding: .utf8)
        }
        if install {
            let dir = home.modelsDir.appendingPathComponent("Aaa/sdxl-model")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Data(count: 4).write(to: dir.appendingPathComponent("m.safetensors"))
            try JSONEncoder().encode(ModelLibrary.Marker(repo: "Aaa/sdxl-model", kind: .image, format: .sdCheckpoint, file: "m.safetensors"))
                .write(to: dir.appendingPathComponent(".parlor-model.json"))
        }
        let e = ParlorEngine(home: home, runtime: llm)
        await e.start()
        return e
    }

    @MainActor
    private func http(_ e: ParlorEngine, _ method: String, _ path: String, token: String?, body: [String: Any]? = nil) async throws -> (Int, [String: Any]) {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(e.endpoint.port)\(path)")!)
        req.httpMethod = method
        if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        if let body { req.httpBody = try JSONSerialization.data(withJSONObject: body) }
        let (d, r) = try await URLSession.shared.data(for: req)
        return ((r as! HTTPURLResponse).statusCode, (try? JSONSerialization.jsonObject(with: d) as? [String: Any]) ?? [:])
    }

    @MainActor
    func testScopedTokenIsForbiddenOnOtherHosts() async throws {
        let e = try await engine(install: true)
        let scoped = e.endpoint.issueEphemeral(name: "scoped", tier: .none, hosts: ["fake"])
        let open = e.endpoint.issueEphemeral(name: "open", tier: .none)
        let other = try await http(e, "POST", "/v1/chat/completions", token: scoped, body: ["model": "other/b", "messages": [["role": "user", "content": "x"]]]).0
        XCTAssertEqual(other, 403)
        let ok = try await http(e, "POST", "/v1/chat/completions", token: scoped, body: ["model": "fake/a", "messages": [["role": "user", "content": "x"]]]).0
        XCTAssertNotEqual(ok, 403)
        // The model list shows only what the token may use.
        let (_, list) = try await http(e, "GET", "/v1/models", token: scoped)
        let ids = (list["data"] as? [[String: Any]])?.compactMap { $0["id"] as? String } ?? []
        XCTAssertTrue(ids.contains("fake/a")); XCTAssertFalse(ids.contains("other/b"))
        let (_, all) = try await http(e, "GET", "/v1/models", token: open)
        XCTAssertTrue(((all["data"] as? [[String: Any]])?.compactMap { $0["id"] as? String } ?? []).contains("other/b"))
        // Images obey the same scope.
        let (s, _) = try await http(e, "POST", "/v1/images/generations", token: scoped, body: ["model": "sdxl-model", "prompt": "a cat"])
        XCTAssertEqual(s, 403)
        await e.shutdown()
    }

    @MainActor
    func testImageRouteErrorsAreOpenAIShaped() async throws {
        let e = try await engine()
        let t = e.endpoint.issueEphemeral(name: "t", tier: .none)
        var (s, o) = try await http(e, "POST", "/v1/images/generations", token: nil, body: ["prompt": "x"])
        XCTAssertEqual(s, 401)
        (s, o) = try await http(e, "POST", "/v1/images/generations", token: t, body: ["model": "nope", "prompt": "x"])
        XCTAssertEqual(s, 404)
        XCTAssertEqual((o["error"] as? [String: Any])?["code"] as? String, "model_not_found")
        (s, o) = try await http(e, "POST", "/v1/images/generations", token: t, body: ["model": "nope"])
        XCTAssertEqual(s, 400)
        (s, _) = try await http(e, "POST", "/v1/embeddings", token: t, body: [:])
        XCTAssertEqual(s, 400)
        await e.shutdown()
    }

    @MainActor
    func testModelsListHostsWithCapabilities() async throws {
        let e = try await engine(install: true)
        let t = e.endpoint.issueEphemeral(name: "t", tier: .none)
        let (_, list) = try await http(e, "GET", "/v1/models", token: t)
        let items = list["data"] as? [[String: Any]] ?? []
        let sd = try XCTUnwrap(items.first { $0["id"] as? String == "sdxl-model" })
        let caps = (sd["parlor"] as? [String: Any])?["capabilities"] as? [String]
        XCTAssertEqual(caps, ["image"])
        await e.shutdown()
    }

    /// With the real runtime a bare model repo is not a reference: only hosts are.
    @MainActor
    func testBareRepoIsNotFoundOnTheRealRuntime() async throws {
        let home = tempHome(); try home.ensure()
        try #"{"gateway":{"enabled":true,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let e = ParlorEngine(home: home)
        await e.start()
        let t = e.endpoint.issueEphemeral(name: "t", tier: .none)
        let (s, o) = try await http(e, "POST", "/v1/chat/completions", token: t, body: ["model": "mlx-community/Qwen3-4B-4bit", "messages": [["role": "user", "content": "x"]]])
        XCTAssertEqual(s, 404)
        XCTAssertEqual((o["error"] as? [String: Any])?["code"] as? String, "model_not_found")
        await e.shutdown()
    }

    private final class Resident: LanguageModelRuntime, @unchecked Sendable {
        let lock = NSLock(); var loaded: String?
        init(_ m: String) { loaded = m }
        func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
            AsyncThrowingStream { $0.yield(.text("ok")); $0.finish() }
        }
        func loadedModel() async -> String? { lock.lock(); defer { lock.unlock() }; return loaded }
        func unload() async { lock.lock(); loaded = nil; lock.unlock() }
    }

    /// `hosts.stop` acts on that host's model only.
    @MainActor
    func testStoppingOneHostLeavesAnothersModelLoaded() async throws {
        let home = tempHome(); try home.ensure()
        for r in ["Aaa/alpha", "Bbb/beta"] {
            let dir = home.modelsDir.appendingPathComponent(r)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Data(count: 8).write(to: dir.appendingPathComponent("m.safetensors"))
            try JSONEncoder().encode(ModelLibrary.Marker(repo: r, kind: .language, format: .mlxLM, file: nil)).write(to: dir.appendingPathComponent(".parlor-model.json"))
        }
        let rt = Resident("Aaa/alpha")
        let e = ParlorEngine(home: home, runtime: rt)
        await e.start()
        let caller = Caller.local
        _ = await e.commands.call("hosts.stop", .object(["id": .string("beta")]), caller: caller)
        let still = await rt.loadedModel()
        XCTAssertEqual(still, "Aaa/alpha", "stopping beta must not unload alpha")
        _ = await e.commands.call("hosts.stop", .object(["id": .string("alpha")]), caller: caller)
        let gone = await rt.loadedModel()
        XCTAssertNil(gone)
        await e.shutdown()
    }

    /// A client that goes away frees the host: generation stops within a second.
    @MainActor
    func testClientDisconnectCancelsGeneration() async throws {
        let rt = Endless()
        let e = try await engine(rt)
        let t = e.endpoint.issueEphemeral(name: "t", tier: .none)
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(e.endpoint.port)/v1/chat/completions")!)
        req.httpMethod = "POST"
        req.setValue("Bearer \(t)", forHTTPHeaderField: "Authorization")
        req.httpBody = try JSONSerialization.data(withJSONObject: ["model": "fake/a", "stream": true, "messages": [["role": "user", "content": "go"]]] as [String: Any])
        let task = Task { () -> Int in
            let (bytes, _) = try await URLSession.shared.bytes(for: req)
            var n = 0
            for try await _ in bytes.lines { n += 1; if n > 3 { break } }
            return n
        }
        _ = try? await task.value
        let start = Date()
        while rt.running > 0, Date().timeIntervalSince(start) < 3 { try await Task.sleep(nanoseconds: 50_000_000) }
        XCTAssertEqual(rt.running, 0)
        XCTAssertLessThan(Date().timeIntervalSince(start), 1.5)
        await e.shutdown()
    }
}

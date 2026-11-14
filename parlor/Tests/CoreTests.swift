import XCTest
@testable import ParlorCore

// MARK: - helpers

/// Scripted stand-in for the language model so flows run with no weights.
final class ScriptedRuntime: LanguageModelRuntime, @unchecked Sendable {
    var handler: @Sendable (LLMRequest) -> [LLMEvent]
    private(set) var requests: [LLMRequest] = []
    init(_ handler: @escaping @Sendable (LLMRequest) -> [LLMEvent]) { self.handler = handler }

    func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
        requests.append(request)
        let events = handler(request)
        return AsyncThrowingStream { c in
            for e in events { c.yield(e) }
            c.finish()
        }
    }
    func loadedModel() async -> String? { "scripted" }
    func unload() async {}
}

struct MockTools: ToolProvider {
    let id = "mock"
    func listTools() async throws -> [ToolDef] {
        [ToolDef(server: "mock", name: "add", description: "Add two integers",
                 schema: .object(["type": .string("object"), "properties": .object(["a": .object(["type": .string("integer")]), "b": .object(["type": .string("integer")])])]))]
    }
    func call(tool: String, arguments: JSON) async throws -> ToolOutput {
        let a = arguments.objectValue?["a"]?.intValue ?? 0, b = arguments.objectValue?["b"]?.intValue ?? 0
        return ToolOutput(text: "\(a + b)")
    }
}

func tempHome() -> ParlorHome {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("parlor-test-\(UUID().uuidString)")
    return ParlorHome(root: url)
}

func scriptedLLM() -> ScriptedRuntime {
    ScriptedRuntime { req in
        let sys = req.messages.first?.text ?? ""
        if sys.contains("faithful, compact conversation summaries") {
            return [.text("SUMMARY: user and agent chatted about numbers.")]
        }
        guard let last = req.messages.last else { return [] }
        let stats = LLMStats(promptTokens: req.messages.reduce(0) { $0 + $1.text.count / 3 + 8 }, completionTokens: 5, tokensPerSecond: 50)
        if last.role == .tool { return [.text("The answer is \(last.text)."), .stats(stats)] }
        if last.text.hasPrefix("add"), !req.tools.isEmpty {
            return [.toolCall(ToolCallRecord(name: "mock__add", arguments: .object(["a": .int(2), "b": .int(3)]))), .stats(stats)]
        }
        return [.text("<think>hmm</think>Echo: \(last.text)"), .stats(stats)]
    }
}

// MARK: - tests

final class CoreTests: XCTestCase {
    func testFrontmatterRoundTripWithAwkwardPersona() {
        var a = Agent(id: "orla", name: "Captain: Orla", summary: "Keeper of the \"light\" # not a comment",
                      persona: "Line one: has a colon.\n\n- not a list\n---\nstill persona", kind: .character,
                      model: "mlx-community/Qwen3-4B-4bit", imageModel: nil, tools: ["image.generate", "zass.*"],
                      contextTokens: 8192, compactAt: 0.65, temperature: 0.9, thinking: true, color: 3)
        a.persona = a.persona.trimmingCharacters(in: .whitespacesAndNewlines)
        let back = Agent(id: "orla", file: a.fileText())
        XCTAssertEqual(back, a)
    }

    func testToolPermissionPatterns() {
        let a = Agent(id: "x", name: "X", tools: ["zass.*", "image.generate"])
        XCTAssertTrue(a.allows("zass.briefing"))
        XCTAssertTrue(a.allows("image.generate"))
        XCTAssertFalse(a.allows("image.edit"))
        XCTAssertFalse(a.allows("calendar.list"))
        XCTAssertTrue(Agent(id: "y", name: "Y", tools: ["*"]).allows("anything.at_all"))
    }

    func testChatLogPersistsAndRewrites() {
        let home = tempHome(); try? home.ensure()
        let log = ChatLog(dir: home.chatDir(for: "a"))
        let m1 = ChatMessage(role: .user, text: "hi"), m2 = ChatMessage(role: .assistant, text: "hello")
        log.append(m1); log.append(m2)
        XCTAssertEqual(ChatLog(dir: home.chatDir(for: "a")).load().map(\.text), ["hi", "hello"])
        log.rewrite([m2])
        XCTAssertEqual(log.load().map(\.text), ["hello"])
        log.saveSummary(.init(text: "s", through: m1.id))
        XCTAssertEqual(log.loadSummary()?.through, m1.id)
    }

    // The phase-1 gate: create agent → tool call → 60 messages → compaction → relaunch → prune.
    @MainActor func testFullFlow() async throws {
        let home = tempHome()
        let llm = scriptedLLM()
        let engine = ParlorEngine(home: home, runtime: llm)
        await engine.start()
        XCTAssertEqual(engine.agents.map(\.id).sorted(), ["mabel", "pip"], "first run seeds two agents")
        await engine.hub.register(MockTools(), status: ServerStatus(id: "mock", kind: "stdio", detail: "", state: .connecting, toolCount: 0, toolNames: []))

        // create an agent by writing a file the way Claude Code would, and see it appear live
        let file = home.agentsDir.appendingPathComponent("orla.md")
        try Agent(id: "orla", name: "Orla", persona: "You keep a lighthouse.", kind: .character, model: "scripted",
                  tools: ["mock.*"], contextTokens: 1200, compactAt: 0.5).fileText().write(to: file, atomically: true, encoding: .utf8)
        for _ in 0..<40 where engine.agent("orla") == nil { try await Task.sleep(nanoseconds: 100_000_000) }
        let orla = try XCTUnwrap(engine.agent("orla"), "file watcher picked up the new agent")

        // tool round trip
        let room = engine.room(for: "orla")
        await room.sendAndWait("add two and three", agent: orla)
        let kinds = room.messages.map(\.role)
        XCTAssertEqual(kinds, [.user, .assistant, .tool, .assistant])
        XCTAssertEqual(room.messages[2].text, "5")
        XCTAssertEqual(room.messages.last?.text, "The answer is 5.")
        XCTAssertEqual(room.messages[1].toolCalls?.first?.qualified, "mock.add")

        // permission: an agent without the tool is refused
        var blocked = orla; blocked.tools = []
        let before = room.messages.count
        await room.sendAndWait("add again", agent: blocked)
        XCTAssertFalse(room.messages.dropFirst(before).contains { $0.role == .tool }, "no tools offered, none called")

        // thinking blocks never reach the log
        XCTAssertFalse(room.messages.contains { $0.text.contains("<think>") })
        XCTAssertTrue(room.messages.contains { $0.text.hasPrefix("Echo: ") })

        // 60 messages → compaction fires and shrinks what the model sees
        for i in 0..<30 { await room.sendAndWait("chatter number \(i) with some padding words to take space", agent: orla) }
        XCTAssertNotNil(room.summaryText, "compaction produced a summary")
        XCTAssertTrue(room.messages.contains { $0.role == .marker && $0.text.contains("compacted") })
        let lastReq = try XCTUnwrap(llm.requests.last)
        XCTAssertTrue(lastReq.messages.first!.text.contains("SUMMARY:"), "summary reaches the system prompt")
        XCTAssertLessThan(lastReq.messages.count, room.messages.count, "history sent is shorter than the log")
        let count = room.messages.count
        let summary = room.summaryText

        // relaunch: new engine, same home → everything is still there
        await engine.shutdown()
        let engine2 = ParlorEngine(home: home, runtime: scriptedLLM())
        await engine2.start()
        let room2 = engine2.room(for: "orla")
        XCTAssertEqual(room2.messages.count, count)
        XCTAssertEqual(room2.summaryText, summary)

        // prune: deleting a tool call also removes its orphaned result
        let toolCallMsg = try XCTUnwrap(room2.messages.first { $0.toolCalls != nil })
        room2.delete(ids: [toolCallMsg.id])
        XCTAssertFalse(room2.messages.contains { $0.role == .tool }, "orphaned tool result removed")
        // cut-after keeps up to the chosen message
        let cutPoint = room2.messages[3]
        room2.cut(after: cutPoint.id)
        XCTAssertEqual(room2.messages.last?.id, cutPoint.id)
        XCTAssertEqual(ChatLog(dir: home.chatDir(for: "orla")).load().count, room2.messages.count, "prune persisted")
        room2.reset()
        XCTAssertTrue(room2.messages.isEmpty)
        XCTAssertNil(ChatLog(dir: home.chatDir(for: "orla")).loadSummary())
        await engine2.shutdown()
    }

    @MainActor func testToolsUpdatedMarkerAppearsMidChat() async throws {
        let home = tempHome()
        let engine = ParlorEngine(home: home, runtime: scriptedLLM())
        await engine.start()
        let a = Agent(id: "t", name: "T", model: "scripted", tools: ["*"])
        engine.save(a)
        let room = engine.room(for: "t")
        await room.sendAndWait("hello", agent: a)
        XCTAssertFalse(room.messages.contains { $0.role == .marker })
        await engine.hub.register(MockTools(), status: ServerStatus(id: "mock", kind: "stdio", detail: "", state: .connecting, toolCount: 0, toolNames: []))
        await room.sendAndWait("hello again", agent: a)
        XCTAssertTrue(room.messages.contains { $0.role == .marker && $0.text.contains("mock.add") })
        await engine.shutdown()
    }

    @MainActor func testDisableAndDeleteServers() async throws {
        let home = tempHome(); try home.ensure()
        try MCPConfigFile(mcpServers: ["flaky": MCPServerConfig(command: "/usr/bin/false")]).save(home.mcpConfig)
        let engine = ParlorEngine(home: home, runtime: scriptedLLM())
        await engine.start()
        XCTAssertNotNil(engine.configuredServers["flaky"])
        for _ in 0..<50 where !engine.servers.contains(where: { $0.id == "flaky" }) { try await Task.sleep(nanoseconds: 100_000_000) }
        XCTAssertTrue(engine.servers.contains { $0.id == "flaky" }, "enabled server is mounted (and here failing); startup did not block on it")

        engine.setServer("flaky", enabled: false)
        for _ in 0..<30 where engine.servers.contains(where: { $0.id == "flaky" }) { try await Task.sleep(nanoseconds: 100_000_000) }
        XCTAssertEqual(engine.configuredServers["flaky"]?.disabled, true, "kept in mcp.json, marked disabled")
        XCTAssertFalse(engine.servers.contains { $0.id == "flaky" }, "disabled server is unmounted")
        XCTAssertEqual(MCPConfigFile.load(home.mcpConfig).mcpServers["flaky"]?.command, "/usr/bin/false", "settings survive")

        engine.setServer("flaky", enabled: true)
        for _ in 0..<30 where !engine.servers.contains(where: { $0.id == "flaky" }) { try await Task.sleep(nanoseconds: 100_000_000) }
        XCTAssertTrue(engine.servers.contains { $0.id == "flaky" }, "re-enabled")

        engine.deleteServer("flaky")
        XCTAssertNil(engine.configuredServers["flaky"])
        XCTAssertNil(MCPConfigFile.load(home.mcpConfig).mcpServers["flaky"], "removed from disk")
        for _ in 0..<30 where engine.servers.contains(where: { $0.id == "flaky" }) { try await Task.sleep(nanoseconds: 100_000_000) }
        XCTAssertFalse(engine.servers.contains { $0.id == "flaky" })
        await engine.shutdown()
    }

    @MainActor func testRenameMovesFileAndChatAndDeleteTidies() async throws {
        let home = tempHome()
        let engine = ParlorEngine(home: home, runtime: scriptedLLM())
        let a = engine.create(kind: .character, name: "New character")
        let log = home.chatDir(for: a.id).appendingPathComponent("log.jsonl")
        try FileManager.default.createDirectory(at: home.chatDir(for: a.id), withIntermediateDirectories: true)
        try "{}\n".write(to: log, atomically: true, encoding: .utf8)
        var edited = a; edited.name = "Babe"
        engine.save(edited)
        XCTAssertEqual(engine.renameToMatchName(a.id), "babe")
        XCTAssertNil(engine.agent("new-character"))
        XCTAssertEqual(engine.agent("babe")?.name, "Babe")
        XCTAssertEqual(engine.selection, "babe")
        XCTAssertTrue(FileManager.default.fileExists(atPath: home.chatDir(for: "babe").appendingPathComponent("log.jsonl").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: home.agentsDir.appendingPathComponent("new-character.md").path))
        // An emptied chat folder for a removed agent is tidied up on reload.
        try FileManager.default.createDirectory(at: home.chatDir(for: "ghost"), withIntermediateDirectories: true)
        engine.reloadAgents()
        XCTAssertFalse(FileManager.default.fileExists(atPath: home.chatDir(for: "ghost").path))
    }

    /// Live network: guards against the Hugging Face search API changing under us again.
    func testHuggingFaceSearchReturnsSizes() async throws {
        let lib = ModelLibrary(home: tempHome())
        let r = try await lib.search("qwen3-4b", kind: .language, limit: 3)
        XCTAssertFalse(r.isEmpty)
        XCTAssertTrue(r.contains { ($0.bytes ?? 0) > 100_000_000 })
    }

    func testSettingsFileTolerantAndRoundTrips() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("settings-\(UUID().uuidString).json")
        try #"{"llm":{"keepLoadedMinutes":0,"kvBits":4},"image":{"precision":"int4"}}"#.write(to: url, atomically: true, encoding: .utf8)
        var s = ParlorSettings.load(url)
        XCTAssertEqual(s.llm.keepLoadedMinutes, 0)
        XCTAssertEqual(s.llm.kvBits, 4)
        XCTAssertFalse(s.llm.promptCacheActive)          // KV quantisation turns the prompt cache off
        XCTAssertEqual(s.llm.maxReplyTokens, 2048)       // untouched keys keep defaults
        XCTAssertEqual(s.loop.maxSteps, 8)
        s.image.fastSteps = 6
        s.save(url)
        XCTAssertEqual(ParlorSettings.load(url), s)
        XCTAssertEqual(ParlorSettings.load(URL(fileURLWithPath: "/nonexistent/x.json")), ParlorSettings())
        let i = ParlorSettings.Image()
        XCTAssertTrue(i.size(shape: "square", detailed: false) == (512, 512))
        XCTAssertTrue(i.size(shape: "portrait", detailed: false) == (448, 640))
        XCTAssertTrue(i.size(shape: "landscape", detailed: true) == (960, 640))
    }

    @MainActor func testSettingsReachTheLoopAndToolsAreClippedAndTimedOut() async throws {
        let home = tempHome()
        var seen: [LLMRequest] = []
        let llm = ScriptedRuntime { req in
            seen.append(req)
            if req.messages.last?.role == .tool { return [.text("done")] }
            return [.toolCall(ToolCallRecord(id: "1", name: "mock__big", arguments: .object([:])))]
        }
        let engine = ParlorEngine(home: home, runtime: llm)
        await engine.start()
        await engine.hub.register(SlowTools(), status: ServerStatus(id: "mock", kind: "test", detail: "", state: .connecting, toolCount: 0, toolNames: []))
        engine.updateSettings { $0.loop.toolResultChars = 400; $0.llm.maxReplyTokens = 77; $0.llm.topP = 0.8 }
        var a = engine.create(kind: .assistant, name: "Tuner")
        a.tools = ["*"]; engine.save(a)
        let room = engine.room(for: a.id)
        await room.sendAndWait("go", agent: engine.agent(a.id)!)
        let toolMsg = room.messages.first { $0.role == .tool }!
        XCTAssertLessThan(toolMsg.text.count, 600)
        XCTAssertTrue(toolMsg.text.contains("cut to save context"))
        XCTAssertEqual(seen.first?.maxTokens, 77)
        XCTAssertEqual(seen.first?.tuning.settings.topP, 0.8)
        // Summaries must not touch the shared prompt cache.
        let hubOut = await engine.hub.call(wireName: "mock__slow", arguments: .object([:]), timeout: 0.2)
        XCTAssertTrue(hubOut.isError); XCTAssertTrue(hubOut.text.contains("timed out"))
        await engine.shutdown()
    }

    /// Live network: a PyTorch-only repo is refused before anything is downloaded.
    func testPyTorchOnlyRepoIsRefusedUpFront() async throws {
        let lib = ModelLibrary(home: tempHome())
        do { _ = try await lib.preflight("Gryphe/MythoMax-L2-13b", kind: .language); XCTFail("should refuse") }
        catch { XCTAssertTrue(error.localizedDescription.contains("safetensors")) }
        let ok = try await lib.preflight("mlx-community/Qwen3-4B-4bit", kind: .language)
        XCTAssertGreaterThan(ok ?? 0, 1_000_000_000)
        let r = try await lib.search("Gryphe/MythoMax-L2-13b", kind: .language, limit: 3)
        XCTAssertNotNil(r.first { $0.repo == "Gryphe/MythoMax-L2-13b" }?.unsupported)
    }
}

struct SlowTools: ToolProvider {
    let id = "mock"
    func listTools() async throws -> [ToolDef] {
        let schema: JSON = .object(["type": .string("object"), "properties": .object([:])])
        return [ToolDef(server: "mock", name: "big", description: "Returns a lot of text", schema: schema),
                ToolDef(server: "mock", name: "slow", description: "Never returns in time", schema: schema)]
    }
    func call(tool: String, arguments: JSON) async throws -> ToolOutput {
        if tool == "slow" { try await Task.sleep(nanoseconds: 5_000_000_000); return ToolOutput(text: "late") }
        return ToolOutput(text: String(repeating: "x", count: 5000))
    }
}

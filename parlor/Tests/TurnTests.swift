import XCTest
@testable import ParlorCore
import MLXLMCommon

/// Thread-safe event sink for harness tests.
final class Events: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [TurnEvent] = []
    func add(_ e: TurnEvent) { lock.lock(); items.append(e); lock.unlock() }
    var all: [TurnEvent] { lock.lock(); defer { lock.unlock() }; return items }
}

/// Streams a partial reply, then hangs until cancelled.
final class HangingRuntime: LanguageModelRuntime, @unchecked Sendable {
    func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
        AsyncThrowingStream { c in
            c.yield(.text("partial"))
            let t = Task { try? await Task.sleep(nanoseconds: 60_000_000_000); c.finish() }
            c.onTermination = { _ in t.cancel() }
        }
    }
    func loadedModel() async -> String? { "hang" }
    func unload() async {}
}

final class TurnTests: XCTestCase {
    private func input(_ agent: Agent, transcript: [ChatMessage], summary: ChatLog.Summary? = nil,
                       tools: [ToolDef] = [], machine: ParlorSettings = ParlorSettings(), last: Int = 0) -> TurnInput {
        TurnInput(agent: agent, conversationID: agent.id, participants: [.agent(agent.id), .user], transcript: transcript,
                  summary: summary, tools: tools, settings: ResolvedSettings.resolve(agent: agent, model: nil, machine: machine),
                  lastPromptTokens: last, now: Date(timeIntervalSince1970: 1_800_000_000))
    }

    private func mockTool() -> ToolDef {
        ToolDef(server: "mock", name: "add", description: "Add", schema: .object([:]))
    }

    @MainActor func testBuiltInRequestMessagesGolden() {
        let a = Agent(id: "x", name: "X", persona: "  You are X.  \n", model: "m")
        let u1 = ChatMessage(role: .user, text: "one"), a1 = ChatMessage(role: .assistant, text: "two")
        let u2 = ChatMessage(role: .user, text: "three")
        let log = [u1, a1, ChatMessage.marker("noted"), u2]

        // No summary, no tools: persona, then the date, then every non-marker line.
        let plain = BuiltInHarness.buildRequestMessages(input(a, transcript: log))
        XCTAssertEqual(plain.map(\.role), [.system, .user, .assistant, .user])
        XCTAssertTrue(plain[0].text.hasPrefix("You are X.\n\nToday is "))
        XCTAssertFalse(plain[0].text.contains("You have tools"))
        XCTAssertFalse(plain[0].text.contains("Earlier conversation"))

        // With tools and a summary through a1: the summary is in the system prompt, history starts after it.
        let sum = ChatLog.Summary(text: "They said one and two.", through: a1.id)
        let full = BuiltInHarness.buildRequestMessages(input(a, transcript: log, summary: sum, tools: [mockTool()]))
        XCTAssertTrue(full[0].text.contains("You have tools. When a request needs"))
        XCTAssertTrue(full[0].text.hasSuffix("## Earlier conversation (summary)\nThey said one and two."))
        XCTAssertEqual(full.dropFirst().map(\.text), ["three"])

        // An empty persona falls back to the name.
        let anon = Agent(id: "y", name: "Why", persona: "")
        XCTAssertTrue(BuiltInHarness.buildRequestMessages(input(anon, transcript: [])).first!.text.hasPrefix("You are Why."))
    }

    func testTurnEventOrderAndLastStepWithholdsTools() async throws {
        let seen = Seen()
        let llm = ScriptedRuntime { req in
            seen.add(req)
            return [.toolCall(ToolCallRecord(id: UUID().uuidString, name: "mock__add", arguments: .object(["a": .int(1), "b": .int(2)])))]
        }
        let hub = ToolHub()
        await hub.register(MockTools(), status: ServerStatus(id: "mock", kind: "t", detail: "", state: .connecting, toolCount: 0, toolNames: []))
        var machine = ParlorSettings(); machine.loop.maxSteps = 2
        let agent = Agent(id: "x", name: "X", model: "m", tools: ["*"])
        let ev = Events()
        try await BuiltInHarness().run(input(agent, transcript: [ChatMessage(role: .user, text: "go")], tools: await hub.tools(for: agent), machine: machine),
                                       context: TurnContext(llm: llm, tools: HubToolExecutor(hub: hub))) { ev.add($0) }
        // Two steps: tools offered on the first, withheld on the last.
        XCTAssertEqual(seen.all.map { $0.tools.count }, [1, 0])
        var shape: [String] = []
        for e in ev.all {
            switch e {
            case .toolStarted: shape.append("start")
            case .message(let m): shape.append(m.role == .assistant ? "assistant(calls:\(m.toolCalls?.count ?? 0))" : "tool")
            default: break
            }
        }
        XCTAssertEqual(shape, ["assistant(calls:1)", "start", "tool", "assistant(calls:1)", "start", "tool"])
        // Every assistant tool call is answered by a tool line with the same id.
        let msgs = ev.all.compactMap { e -> ChatMessage? in if case .message(let m) = e { return m } else { return nil } }
        for (i, m) in msgs.enumerated() where m.role == .assistant {
            XCTAssertEqual(msgs[i + 1].toolCallID, m.toolCalls?.first?.id)
        }
    }

    func testToolExecutorEnforcesAllowlist() async {
        let hub = ToolHub()
        await hub.register(MockTools(), status: ServerStatus(id: "mock", kind: "t", detail: "", state: .connecting, toolCount: 0, toolNames: []))
        let exec = HubToolExecutor(hub: hub)
        let s = ResolvedSettings.resolve(agent: Agent(id: "n", name: "N"), model: nil, machine: ParlorSettings())
        let call = ToolCallRecord(name: "mock__add", arguments: .object(["a": .int(1), "b": .int(1)]))
        let denied = await exec.call(call, agent: Agent(id: "n", name: "N", tools: []), settings: s)
        XCTAssertTrue(denied.isError); XCTAssertTrue(denied.text.contains("not enabled"))
        let ok = await exec.call(call, agent: Agent(id: "n", name: "N", tools: ["mock.*"]), settings: s)
        XCTAssertFalse(ok.isError); XCTAssertEqual(ok.text, "2")
    }

    func testCompactionIsAnEventBeforeTheReply() async throws {
        let llm = scriptedLLM()
        let hub = ToolHub()
        let agent = Agent(id: "c", name: "C", model: "m", contextTokens: 200, compactAt: 0.1)
        var log: [ChatMessage] = []
        for i in 0..<10 { log.append(ChatMessage(role: i % 2 == 0 ? .user : .assistant, text: "message number \(i) with padding text")) }
        log.append(ChatMessage(role: .user, text: "latest"))
        let ev = Events()
        try await BuiltInHarness().run(input(agent, transcript: log), context: TurnContext(llm: llm, tools: HubToolExecutor(hub: hub))) { ev.add($0) }
        var order: [String] = []
        for e in ev.all {
            switch e {
            case .summary(let s, let folded): order.append("summary"); XCTAssertTrue(s.text.hasPrefix("SUMMARY")); XCTAssertGreaterThan(folded, 0)
            case .delta: if !order.contains("delta") { order.append("delta") }
            default: break
            }
        }
        XCTAssertEqual(order, ["summary", "delta"])
    }

    @MainActor func testCancelKeepsPartialReply() async throws {
        let engine = ParlorEngine(home: tempHome(), runtime: HangingRuntime())
        await engine.start()
        let a = Agent(id: "h", name: "H", model: "hang")
        engine.save(a)
        let room = engine.room(for: "h")
        room.send("hello", agent: a)
        for _ in 0..<100 where room.streaming != "partial" { try await Task.sleep(nanoseconds: 50_000_000) }
        XCTAssertEqual(room.streaming, "partial")
        room.stop()
        for _ in 0..<100 where room.isBusy { try await Task.sleep(nanoseconds: 50_000_000) }
        XCTAssertFalse(room.isBusy)
        XCTAssertEqual(room.messages.last?.role, .assistant)
        XCTAssertEqual(room.messages.last?.text, "partial")
        await engine.shutdown()
    }

    func testSettingsResolutionPrecedenceAndClamp() {
        var machine = ParlorSettings()
        machine.llm.topP = 0.95; machine.loop.maxSteps = 8; machine.llm.maxReplyTokens = 2048
        var agent = Agent(id: "r", name: "R", contextTokens: 32768, temperature: 0.9)
        // Nothing overridden: machine values, V1 agent values, no model.
        var r = ResolvedSettings.resolve(agent: agent, model: nil, machine: machine)
        XCTAssertEqual(r.topP, 0.95); XCTAssertEqual(r.maxSteps, 8); XCTAssertEqual(r.maxReplyTokens, 2048)
        XCTAssertEqual(r.temperature, 0.9); XCTAssertEqual(r.contextTokens, 32768)
        // Model defaults beat the machine; the agent beats the model; the model's max context clamps.
        var meta = ModelMetadata(); meta.topP = 0.8; meta.maxContext = 8192
        r = ResolvedSettings.resolve(agent: agent, model: meta, machine: machine)
        XCTAssertEqual(r.topP, 0.8); XCTAssertEqual(r.contextTokens, 8192)
        agent.overrides.topP = 0.5; agent.overrides.maxSteps = 3; agent.overrides.maxReplyTokens = 0
        r = ResolvedSettings.resolve(agent: agent, model: meta, machine: machine)
        XCTAssertEqual(r.topP, 0.5); XCTAssertEqual(r.llm.topP, 0.5)
        XCTAssertEqual(r.maxSteps, 3); XCTAssertNil(r.maxReplyTokens, "0 means no cap")
    }

    func testModelMetadataReadsV1AndExtendedFiles() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("meta-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        XCTAssertNil(ModelMetadata.load(dir: dir))
        try #"{"kind":"language","repo":"a/b"}"#.write(to: dir.appendingPathComponent(".parlor-model.json"), atomically: true, encoding: .utf8)
        XCTAssertEqual(ModelMetadata.load(dir: dir), ModelMetadata())
        try #"{"kind":"language","repo":"a/b","defaults":{"top_p":0.7,"context":4096},"capabilities":{"tools":false,"maxContext":8192}}"#
            .write(to: dir.appendingPathComponent(".parlor-model.json"), atomically: true, encoding: .utf8)
        let m = try XCTUnwrap(ModelMetadata.load(dir: dir))
        XCTAssertEqual(m.topP, 0.7); XCTAssertEqual(m.context, 4096); XCTAssertEqual(m.supportsTools, false); XCTAssertEqual(m.maxContext, 8192)
    }

    func testLogLinesWithoutSpeakerLoadAndUnknownKeysSurviveRewrite() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("log-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let v1 = #"{"id":"11111111-1111-1111-1111-111111111111","role":"assistant","text":"hi","createdAt":"2026-11-13T10:00:00Z","futureField":{"a":[1,2]}}"#
        try (v1 + "\n").write(to: dir.appendingPathComponent("log.jsonl"), atomically: true, encoding: .utf8)
        let log = ChatLog(dir: dir)
        var msgs = log.load()
        XCTAssertEqual(msgs.count, 1)
        XCTAssertNil(msgs[0].speaker)
        XCTAssertEqual(msgs[0].speaker(inChatOf: "mabel"), "mabel")
        XCTAssertEqual(ChatMessage(role: .user, text: "x").speaker(inChatOf: "mabel"), "user")
        msgs.append(ChatMessage(role: .user, text: "hello", speaker: "user"))
        log.rewrite(msgs)
        let again = log.load()
        XCTAssertEqual(again.count, 2)
        XCTAssertNotNil(again[0].extra?["futureField"], "unknown keys are kept on rewrite")
        XCTAssertEqual(again[1].speaker, "user")
    }

    func testAgentFileKeepsUnknownKeysAndV1FilesRoundTripUnchanged() {
        let v1 = Agent(id: "m", name: "M", summary: "d", persona: "Persona.", model: "x/y", tools: ["a.*"]).fileText()
        XCTAssertEqual(Agent(id: "m", file: v1).fileText(), v1, "opening and saving a V1 file changes nothing")
        let withExtra = v1.replacingOccurrences(of: "color: 0\n", with: "color: 0\ntier: act\nharness: my-loop --fast\n")
        let a = Agent(id: "m", file: withExtra)
        XCTAssertEqual(a.extra.map(\.key), ["tier", "harness"])
        let saved = a.fileText()
        XCTAssertTrue(saved.contains("tier: act")); XCTAssertTrue(saved.contains("harness: my-loop --fast"))
        var o = Agent(id: "o", file: v1); o.overrides.maxSteps = 4; o.overrides.maxReplyTokens = 0
        let back = Agent(id: "o", file: o.fileText())
        XCTAssertEqual(back.overrides.maxSteps, 4); XCTAssertEqual(back.overrides.maxReplyTokens, 0)
        XCTAssertTrue(o.fileText().contains("max_reply_tokens: none"))
    }

    @MainActor func testToolsMarkerSurvivesRelaunch() async throws {
        let home = tempHome()
        let first = ParlorEngine(home: home, runtime: scriptedLLM())
        await first.start()
        let a = Agent(id: "t", name: "T", model: "scripted", tools: ["*"])
        first.save(a)
        await first.room(for: "t").sendAndWait("hello", agent: a)
        await first.shutdown()
        XCTAssertNotNil(ChatLog(dir: home.chatDir(for: "t")).loadToolSnapshot())

        // New process, a tool has appeared in the meantime.
        let second = ParlorEngine(home: home, runtime: scriptedLLM())
        await second.start()
        await second.hub.register(MockTools(), status: ServerStatus(id: "mock", kind: "t", detail: "", state: .connecting, toolCount: 0, toolNames: []))
        let room = second.room(for: "t")
        await room.sendAndWait("hello again", agent: a)
        XCTAssertTrue(room.messages.contains { $0.role == .marker && $0.text.contains("mock.add") })
        await second.shutdown()
    }

    @MainActor func testStarterAgentsAreSeededOnce() async throws {
        let home = tempHome()
        let e1 = ParlorEngine(home: home, runtime: scriptedLLM())
        await e1.start()
        XCTAssertEqual(e1.agents.map(\.id).sorted(), ["mabel", "pip"])
        for a in e1.agents { e1.delete(a.id) }
        await e1.shutdown()
        let e2 = ParlorEngine(home: home, runtime: scriptedLLM())
        await e2.start()
        XCTAssertTrue(e2.agents.isEmpty, "deleting every agent does not resurrect the starters")
        await e2.shutdown()

        // An existing V1 home (agents, no marker) is adopted, not topped up.
        let v1 = tempHome()
        try v1.ensure()
        try Agent(id: "solo", name: "Solo").fileText().write(to: v1.agentsDir.appendingPathComponent("solo.md"), atomically: true, encoding: .utf8)
        let e3 = ParlorEngine(home: v1, runtime: scriptedLLM())
        await e3.start()
        XCTAssertEqual(e3.agents.map(\.id), ["solo"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: v1.seededMarker.path))
        await e3.shutdown()
    }

    func testConversationConfigDefaultsToOneToOneAndParsesAFile() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("conv-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let d = ConversationConfig.load(dir: dir, agentID: "mabel")
        XCTAssertEqual(d.agentIDs, ["mabel"]); XCTAssertEqual(d.scheduler.kind, "userDriven")
        try #"{"version":1,"participants":[{"agent":"mabel"},{"agent":"pip"},{"user":true}],"scheduler":{"kind":"roundRobin","maxTurns":6}}"#
            .write(to: dir.appendingPathComponent("conversation.json"), atomically: true, encoding: .utf8)
        let c = ConversationConfig.load(dir: dir, agentID: "mabel")
        XCTAssertEqual(c.agentIDs, ["mabel", "pip"]); XCTAssertEqual(c.scheduler.maxTurns, 6)
        // The user-driven scheduler answers only after the user speaks.
        let s = UserDrivenScheduler()
        XCTAssertEqual(s.next(config: d, log: [ChatMessage(role: .user, text: "hi")]), "mabel")
        XCTAssertNil(s.next(config: d, log: [ChatMessage(role: .user, text: "hi"), ChatMessage(role: .assistant, text: "yo")]))
    }
}

final class Seen: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [LLMRequest] = []
    func add(_ r: LLMRequest) { lock.lock(); items.append(r); lock.unlock() }
    var all: [LLMRequest] { lock.lock(); defer { lock.unlock() }; return items }
}

// MARK: - step 2: queue and cache slots

/// Records the order in which queued jobs actually start.
final class Order: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [String] = []
    func add(_ s: String) { lock.lock(); items.append(s); lock.unlock() }
    var all: [String] { lock.lock(); defer { lock.unlock() }; return items }
}

final class QueueTests: XCTestCase {
    func testQueueRunsOneAtATimeByPriorityThenLoadedModel() async throws {
        let q = LLMQueue()
        let order = Order()
        let first = try await q.acquire(model: "A", priority: .user)         // running
        var jobs: [Task<Void, Error>] = []
        func enqueue(_ name: String, _ model: String, _ p: LLMPriority) async {
            jobs.append(Task { let t = try await q.acquire(model: model, priority: p); order.add(name); await q.release(t) })
            while await q.depth < jobs.count { try? await Task.sleep(nanoseconds: 5_000_000) }
        }
        await enqueue("bg-A", "A", .background)
        await enqueue("room-B", "B", .room)
        await enqueue("user-B", "B", .user)
        await enqueue("user-A", "A", .user)
        let depth = await q.depth
        XCTAssertEqual(depth, 4)
        await q.release(first)
        for j in jobs { try await j.value }
        // Users first, and between equals the model that just ran (A) goes before B; then room; then background.
        XCTAssertEqual(order.all, ["user-A", "user-B", "room-B", "bg-A"])
    }

    func testQueueFairnessAndCancellation() async throws {
        let q = LLMQueue(fairness: 2)
        let order = Order()
        let running = try await q.acquire(model: "A", priority: .user)
        let low = Task { let t = try await q.acquire(model: "B", priority: .background); order.add("low"); await q.release(t) }
        while await q.depth < 1 { try? await Task.sleep(nanoseconds: 5_000_000) }
        // A stream of higher-priority jobs cannot starve the low one forever.
        var highs: [Task<Void, Error>] = []
        for i in 0..<5 {
            highs.append(Task { let t = try await q.acquire(model: "A", priority: .user); order.add("high\(i)"); await q.release(t) })
            while await q.depth < 2 + i { try? await Task.sleep(nanoseconds: 5_000_000) }
        }
        await q.release(running)
        try await low.value
        for h in highs { try await h.value }
        let idx = try XCTUnwrap(order.all.firstIndex(of: "low"))
        XCTAssertLessThanOrEqual(idx, 2, "the low-priority job runs after being passed over twice")

        // A cancelled waiter leaves the queue without running.
        let hold = try await q.acquire(model: "A", priority: .user)
        let doomed = Task { _ = try await q.acquire(model: "A", priority: .user) }
        while await q.depth < 1 { try? await Task.sleep(nanoseconds: 5_000_000) }
        doomed.cancel()
        _ = await doomed.result
        let d = await q.depth
        XCTAssertEqual(d, 0)
        await q.release(hold)
    }

    func testQueuedRuntimeSerialisesTwoStreams() async throws {
        let inFlight = Counter()
        let slow = SlowRuntime(counter: inFlight)
        let rt = QueuedRuntime(inner: slow)
        async let a: Void = { for try await _ in rt.generate(model: "m", request: LLMRequest(messages: [])) {} }()
        async let b: Void = { for try await _ in rt.generate(model: "m", request: LLMRequest(messages: [], priority: .room)) {} }()
        _ = try await (a, b)
        XCTAssertEqual(inFlight.maxSeen, 1, "never two generations at once")
        XCTAssertEqual(inFlight.total, 2)
    }

    func testSlotsKeepAlternatingConversationsApart() {
        let s = SlotCache(limit: 2)
        let a = Array(0..<100), b = Array(1000..<1100)
        // First prompt of each conversation takes its own slot.
        let ia = s.victim(); s.claim(ia, cache: [KVCacheSimple()]); s.finish(ia, tokens: a)
        let ib = s.victim(); s.claim(ib, cache: [KVCacheSimple()]); s.finish(ib, tokens: b)
        XCTAssertNotEqual(ia, ib)
        // The next turn of A (same start, longer) picks A's slot, not B's.
        let nextA = a + [7, 8, 9]
        XCTAssertEqual(s.candidates(for: nextA).first?.0, ia)
        XCTAssertEqual(s.candidates(for: nextA).first?.1, 100)
        let nextB = b + [5]
        XCTAssertEqual(s.candidates(for: nextB).first?.0, ib)
        // A third conversation evicts the least recently used slot (A's).
        XCTAssertEqual(s.victim(), ia)
        // Lowering the limit drops the oldest slot.
        s.limit = 1
        XCTAssertEqual(s.slots.count, 1)
    }

    func testSettingsCarryBudgetAndSlots() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("s-\(UUID().uuidString).json")
        try #"{"memoryBudgetFraction":0.4,"cacheSlotsPerModel":2}"#.write(to: url, atomically: true, encoding: .utf8)
        let s = ParlorSettings.load(url)
        XCTAssertEqual(s.memoryBudgetFraction, 0.4); XCTAssertEqual(s.cacheSlotsPerModel, 2)
        XCTAssertEqual(ParlorSettings().cacheSlotsPerModel, 4)
    }
}

final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var now = 0
    private(set) var maxSeen = 0
    private(set) var total = 0
    func enter() { lock.lock(); now += 1; total += 1; maxSeen = max(maxSeen, now); lock.unlock() }
    func leave() { lock.lock(); now -= 1; lock.unlock() }
}

final class SlowRuntime: LanguageModelRuntime, @unchecked Sendable {
    let counter: Counter
    init(counter: Counter) { self.counter = counter }
    func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
        AsyncThrowingStream { c in
            counter.enter()
            Task { try? await Task.sleep(nanoseconds: 80_000_000); c.yield(.text("x")); self.counter.leave(); c.finish() }
        }
    }
    func loadedModel() async -> String? { "slow" }
    func unload() async {}
}


// MARK: - step 3: rooms

/// Replies as whoever the system prompt says it is, mentioning a configurable next speaker.
func roomLLM(seen: Seen? = nil, mention: [String: String] = [:]) -> ScriptedRuntime {
    ScriptedRuntime { req in
        seen?.add(req)
        let sys = req.messages.first?.text ?? ""
        let who = sys.contains("You are Mabel") ? "Mabel" : (sys.contains("You are Pip") ? "Pip" : "Babe")
        let stats = LLMStats(promptTokens: 10, completionTokens: 3, tokensPerSecond: 50)
        let tail = mention[who].map { " \($0), your turn?" } ?? ""
        return [.text("\(who) speaking.\(tail)"), .stats(stats)]
    }
}

final class RoomTests: XCTestCase {
    @MainActor private func roomEngine(_ llm: LanguageModelRuntime) async -> ParlorEngine {
        let e = ParlorEngine(home: tempHome(), runtime: llm)
        await e.start()
        for id in ["mabel", "pip", "babe"] where e.agent(id) == nil {
            e.save(Agent(id: id, name: id.capitalized, persona: "You are \(id.capitalized).", model: "scripted"))
        }
        for a in e.agents { var x = a; x.model = "scripted"; e.save(x) }
        return e
    }

    @MainActor func testRoundRobinAttributesEveryLineAndFramesTheTranscript() async throws {
        let seen = Seen()
        let e = await roomEngine(roomLLM(seen: seen))
        let id = e.createRoom(title: "Settle it", agents: ["mabel", "pip"], scheduler: SchedulerSpec(kind: "roundRobin", maxTurns: 4))
        XCTAssertEqual(e.conversations.map(\.id), [id])
        let room = e.room(for: id)
        XCTAssertTrue(room.isGroup)
        await room.sendAndWait("Who is right?")
        let speakers = room.messages.filter { $0.role != .marker }.map { $0.speaker ?? "?" }
        XCTAssertEqual(speakers, ["user", "mabel", "pip", "mabel", "pip"])
        // Pip's first request sees Mabel's line as an attributed user line, not as its own.
        let pipReq = try XCTUnwrap(seen.all.first { ($0.messages.first?.text ?? "").contains("You are Pip") })
        XCTAssertTrue(pipReq.messages[0].text.contains("You are Pip, in a conversation with the user and Mabel"))
        let last = try XCTUnwrap(pipReq.messages.last)
        XCTAssertEqual(last.role, .user)
        XCTAssertEqual(last.text, "User: Who is right?\nMabel: Mabel speaking.")
        await e.shutdown()
    }

    @MainActor func testMentionSchedulerFollowsNamesAndStops() async throws {
        let e = await roomEngine(roomLLM(mention: ["Pip": "Babe"]))
        let id = e.createRoom(title: "Chain", agents: ["mabel", "pip", "babe"], scheduler: SchedulerSpec(kind: "mention", maxTurns: 6))
        let room = e.room(for: id)
        await room.sendAndWait("pip, what do you think?")
        // Pip is named first, Pip names Babe, Babe names nobody: the chain ends.
        XCTAssertEqual(room.messages.filter { $0.role == .assistant }.compactMap(\.speaker), ["pip", "babe"])
        XCTAssertEqual(MentionScheduler.firstMention(in: "Mabelle and Pip", ids: ["mabel", "pip"], names: ["mabel": "Mabel", "pip": "Pip"], excluding: nil), "pip", "whole words only")
        XCTAssertNil(MentionScheduler.firstMention(in: "Pip said so", ids: ["pip"], names: ["pip": "Pip"], excluding: "pip"))
        await e.shutdown()
    }

    @MainActor func testAutonomousRunStopsAtItsBudgetAndNeedsNoUserMessage() async throws {
        let e = await roomEngine(roomLLM())
        let id = e.createRoom(title: "Free", agents: ["mabel", "pip"], scheduler: SchedulerSpec(kind: "autonomous", stepBudget: 3))
        let room = e.room(for: id)
        XCTAssertTrue(room.messages.isEmpty)
        await room.runAndWait()
        XCTAssertEqual(room.messages.filter { $0.role == .assistant }.compactMap(\.speaker), ["mabel", "pip", "mabel"])
        XCTAssertFalse(room.messages.contains { $0.role == .user })
        // A model that just spoke is asked to carry on.
        let a = Agent(id: "mabel", name: "Mabel")
        let msgs = BuiltInHarness.buildRequestMessages(agent: a, transcript: [ChatMessage(role: .assistant, text: "hi", speaker: "mabel")], summary: nil, tools: [], now: Date(), others: [("pip", "Pip")])
        XCTAssertEqual(msgs.last?.text, "(Carry on the conversation.)")
        await e.shutdown()
    }

    @MainActor func testRoomsPersistReloadAndDelete() async throws {
        let e = await roomEngine(roomLLM())
        let id = e.createRoom(title: "Keep me", agents: ["mabel", "pip"], scheduler: SchedulerSpec(kind: "roundRobin", maxTurns: 2))
        e.reloadAgents()                       // must not tidy away a room folder
        XCTAssertEqual(e.conversations.count, 1)
        // A relaunch finds it again, with its log.
        await e.room(for: id).sendAndWait("hello all")
        await e.shutdown()
        let again = ParlorEngine(home: e.home, runtime: roomLLM())
        await again.start()
        XCTAssertEqual(again.conversations.map(\.id), [id])
        XCTAssertEqual(again.conversations.first?.title, "Keep me")
        XCTAssertEqual(again.room(for: id).messages.filter { $0.role == .assistant }.count, 2)
        // One-to-one chats are not rooms.
        XCTAssertFalse(again.conversations.contains { $0.id == "mabel" })
        again.deleteRoom(id)
        XCTAssertTrue(again.conversations.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: again.home.chatDir(for: id).path))
        await again.shutdown()
    }
}

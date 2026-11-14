import Foundation
import Observation

/// One conversation: a chat folder with a log, a summary and a scheduler. A V1 chat is the one-to-one case
/// (one agent, the user); a room has several agents and another scheduler.
@Observable @MainActor
public final class Room: Identifiable {
    /// The chat folder id. For a one-to-one chat this is also the agent's id.
    public nonisolated let agentID: String
    public nonisolated var id: String { agentID }
    public private(set) var messages: [ChatMessage] = []
    public private(set) var streaming: String = ""
    public private(set) var activity: String?
    public private(set) var isBusy = false
    public private(set) var errorText: String?
    /// Prompt size reported by the model on the last turn.
    public private(set) var contextUsed: Int = 0
    public private(set) var lastStats: LLMStats?
    /// Who is writing the reply that is streaming now.
    public private(set) var speakingID: String?

    let log: ChatLog
    public private(set) var config: ConversationConfig
    private var summary: ChatLog.Summary?
    private var task: Task<Void, Never>?
    private var toolSnapshot: Set<String>?
    private var currentSpeaker: String
    private var runPriority: LLMPriority = .user
    private unowned let deps: RoomDeps

    init(agentID: String, deps: RoomDeps) {
        self.agentID = agentID
        self.deps = deps
        self.currentSpeaker = agentID
        self.log = ChatLog(dir: deps.home.chatDir(for: agentID))
        self.config = ConversationConfig.load(dir: deps.home.chatDir(for: agentID), agentID: agentID)
        self.messages = log.load()
        self.summary = log.loadSummary()
    }

    public var summaryText: String? { summary?.text }
    /// More than one agent, or a scheduler that is not "the user drives".
    public var isGroup: Bool { config.isRoom }
    public var participantIDs: [String] { config.agentIDs }

    // MARK: sending

    /// The user speaks; the scheduler then decides who answers. `agent` is the one-to-one chat's agent.
    public func send(_ text: String, agent: Agent? = nil) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        start(userText: trimmed, fallback: agent)
    }

    /// Let the scheduler run without a user message (round robin, mentions, autonomous).
    public func run() { start(userText: nil, fallback: nil) }

    private func start(userText: String?, fallback: Agent?) {
        guard !isBusy else { return }
        errorText = nil
        isBusy = true
        runPriority = userText != nil ? .user : (config.scheduler.kind == "autonomous" ? .background : .room)
        task = Task { [weak self] in
            guard let self else { return }
            await self.conversation(userText: userText, fallback: fallback)
        }
    }

    /// Awaitable variants for the CLI and tests.
    public func sendAndWait(_ text: String, agent: Agent? = nil) async {
        send(text, agent: agent)
        await task?.value
    }

    public func runAndWait() async {
        run()
        await task?.value
    }

    public func stop() { task?.cancel() }

    private func resolve(_ id: String, _ fallback: Agent?) -> Agent? {
        if let f = fallback, f.id == id { return f }
        return deps.agentLookup?(id) ?? (agentID == id ? fallback : nil)
    }

    private func nameMap() -> [String: String] {
        var m: [String: String] = [:]
        for id in config.agentIDs { m[id] = deps.agentLookup?(id)?.name ?? id }
        return m
    }

    private func conversation(userText: String?, fallback: Agent?) async {
        defer { isBusy = false; streaming = ""; activity = nil; speakingID = nil; task = nil }
        if !isGroup, let primary = resolve(config.agentIDs.first ?? agentID, fallback) {
            // A one-to-one chat notes tool changes before the user's message, as before.
            noteToolChanges(await deps.hub.tools(for: primary))
        }
        if let userText { append(ChatMessage(role: .user, text: userText)) }
        let names = nameMap()
        let scheduler = config.scheduler.make()
        let cap = config.scheduler.hardCap(agentCount: config.agentIDs.count)
        var turns = 0
        while turns < cap, !Task.isCancelled {
            let state = SchedulerState(config: config, log: messages, names: names, turnsThisRun: turns)
            guard let next = scheduler.next(state), let agent = resolve(next, fallback) else { break }
            let ok = await turn(agent, names: names)
            turns += 1
            if !ok { break }
        }
    }

    /// One agent's reply. False when it was cancelled or failed.
    private func turn(_ agent: Agent, names: [String: String]) async -> Bool {
        currentSpeaker = agent.id
        speakingID = agent.id
        let tools = await deps.hub.tools(for: agent)
        await deps.image?.setModel(agent.imageModel)
        do {
            activity = "Thinking"
            let input = makeInput(agent: agent, tools: tools, names: names)
            let context = TurnContext(llm: deps.llm, tools: HubToolExecutor(hub: deps.hub))
            let harness = deps.harness
            try await harness.run(input, context: context) { [weak self] ev in
                Task { @MainActor in self?.handle(ev) }
            }
            // Let queued MainActor handlers drain before returning.
            await Task.yield()
            try? await Task.sleep(nanoseconds: 30_000_000)
            return true
        } catch is CancellationError {
            if !streaming.isEmpty { append(ChatMessage(role: .assistant, text: streaming)) }
            return false
        } catch {
            errorText = error.localizedDescription
            return false
        }
    }

    private func makeInput(agent: Agent, tools: [ToolDef], names: [String: String]? = nil) -> TurnInput {
        let meta = deps.library.flatMap { lib in
            lib.installedDirectory(for: agent.model).flatMap { ModelMetadata.load(dir: $0) }
        }
        return TurnInput(
            agent: agent, conversationID: id, participants: config.participants, transcript: messages,
            summary: summary, tools: tools,
            settings: ResolvedSettings.resolve(agent: agent, model: meta, machine: deps.settings),
            lastPromptTokens: contextUsed, priority: runPriority, names: names ?? nameMap())
    }

    private func handle(_ ev: TurnEvent) {
        switch ev {
        case .delta(let t): streaming = t; activity = nil
        case .status(let s): if streaming.isEmpty { activity = s }
        case .message(let m):
            if m.role == .assistant {
                streaming = ""
                if !m.text.isEmpty || m.toolCalls != nil { append(m) }
                activity = "Thinking"
            } else {
                append(m)
            }
        case .toolStarted(let c): activity = "Running \(c.qualified)"
        case .summary(let s, let folded): adopt(summary: s, folded: folded)
        case .stats(let s): lastStats = s; contextUsed = s.promptTokens + s.completionTokens
        }
    }

    private func adopt(summary s: ChatLog.Summary, folded: Int) {
        summary = s
        log.saveSummary(s)
        append(.marker("Earlier conversation compacted · \(folded) messages folded into a short summary"))
        contextUsed = 0
    }

    private func append(_ m: ChatMessage) {
        var m = m
        if m.speaker == nil, m.role != .marker { m.speaker = m.role == .user ? "user" : currentSpeaker }
        messages.append(m)
        log.append(m)
    }

    private func noteToolChanges(_ tools: [ToolDef]) {
        let now = Set(tools.map(\.qualified))
        defer { toolSnapshot = now; log.saveToolSnapshot(now) }
        guard let before = toolSnapshot ?? log.loadToolSnapshot() else { return }
        let added = now.subtracting(before).sorted(), removed = before.subtracting(now).sorted()
        guard !added.isEmpty || !removed.isEmpty else { return }
        var parts: [String] = []
        if !added.isEmpty { parts.append("added " + added.joined(separator: ", ")) }
        if !removed.isEmpty { parts.append("removed " + removed.joined(separator: ", ")) }
        append(.marker("Tools updated: " + parts.joined(separator: "; ")))
    }

    // MARK: context

    @discardableResult
    public func compact(agent: Agent) async throws -> Bool {
        activity = "Compacting"
        defer { activity = nil }
        let input = makeInput(agent: agent, tools: [])
        guard let (new, folded) = try await BuiltInHarness.compact(input: input, summary: summary, llm: deps.llm) else { return false }
        adopt(summary: new, folded: folded)
        return true
    }

    public func compactNow(agent: Agent) {
        guard !isBusy else { return }
        isBusy = true
        task = Task { [weak self] in
            guard let self else { return }
            defer { self.isBusy = false; self.task = nil }
            do { try await self.compact(agent: agent) } catch { self.errorText = error.localizedDescription }
        }
    }

    // MARK: pruning

    public func delete(ids: Set<UUID>) {
        messages.removeAll { ids.contains($0.id) }
        sanitize()
        persist()
    }

    /// Keep everything up to and including `id`.
    public func cut(after id: UUID) {
        guard let i = messages.firstIndex(where: { $0.id == id }) else { return }
        messages = Array(messages[...i])
        sanitize()
        persist()
    }

    public func reset() {
        messages = []
        summary = nil
        contextUsed = 0
        log.rewrite([])
        log.saveSummary(nil)
    }

    private func persist() {
        log.rewrite(messages)
        if let t = summary?.through, !messages.contains(where: { $0.id == t }) {
            // The folded range was pruned away; the summary no longer matches the log.
            summary = nil
            log.saveSummary(nil)
        }
        contextUsed = 0
    }

    /// Keep tool calls and their results paired after deletions.
    private func sanitize() {
        var calls = Set<String>()
        var answered = Set<String>()
        for m in messages {
            for c in m.toolCalls ?? [] { calls.insert(c.id) }
            if m.role == .tool, let id = m.toolCallID { answered.insert(id) }
        }
        messages.removeAll { m in
            if m.role == .tool, let id = m.toolCallID { return !calls.contains(id) }
            return false
        }
        for i in messages.indices where messages[i].role == .assistant {
            if let c = messages[i].toolCalls {
                let kept = c.filter { answered.contains($0.id) }
                messages[i].toolCalls = kept.isEmpty ? nil : kept
            }
        }
        messages.removeAll { $0.role == .assistant && $0.text.isEmpty && $0.toolCalls == nil }
    }

    /// Reload from disk (external edit).
    func reloadFromDisk() {
        guard !isBusy else { return }
        config = ConversationConfig.load(dir: deps.home.chatDir(for: agentID), agentID: agentID)
        messages = log.load()
        summary = log.loadSummary()
    }
}

/// Shared dependencies of every room.
@MainActor
public final class RoomDeps {
    public let home: ParlorHome
    public let hub: ToolHub
    public var llm: LanguageModelRuntime
    public var harness: Harness
    public var image: ImageProvider?
    public var library: ModelLibrary?
    public var settings = ParlorSettings()
    /// Finds an agent by id; set by the engine so a room can address everyone in it.
    public var agentLookup: (@MainActor (String) -> Agent?)?
    init(home: ParlorHome, hub: ToolHub, llm: LanguageModelRuntime, harness: Harness, library: ModelLibrary? = nil) {
        self.home = home; self.hub = hub; self.llm = llm; self.harness = harness; self.library = library
    }
}

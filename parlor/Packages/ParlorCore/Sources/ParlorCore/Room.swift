import Foundation
import Observation

/// One agent's chat room. One room per agent, always.
@Observable @MainActor
public final class Room: Identifiable {
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

    let log: ChatLog
    public let config: ConversationConfig
    private let scheduler: Scheduler = UserDrivenScheduler()
    private var summary: ChatLog.Summary?
    private var task: Task<Void, Never>?
    private var toolSnapshot: Set<String>?
    private unowned let deps: RoomDeps

    init(agentID: String, deps: RoomDeps) {
        self.agentID = agentID
        self.deps = deps
        self.log = ChatLog(dir: deps.home.chatDir(for: agentID))
        self.config = ConversationConfig.load(dir: deps.home.chatDir(for: agentID), agentID: agentID)
        self.messages = log.load()
        self.summary = log.loadSummary()
    }

    public var summaryText: String? { summary?.text }

    // MARK: sending

    public func send(_ text: String, agent: Agent) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isBusy else { return }
        errorText = nil
        isBusy = true
        task = Task { [weak self] in
            guard let self else { return }
            await self.turn(trimmed, agent: agent)
        }
    }

    /// Awaitable variant for the CLI and tests.
    public func sendAndWait(_ text: String, agent: Agent) async {
        send(text, agent: agent)
        await task?.value
    }

    public func stop() { task?.cancel() }

    private func turn(_ text: String, agent: Agent) async {
        defer { isBusy = false; streaming = ""; activity = nil; task = nil }
        let tools = await deps.hub.tools(for: agent)
        await deps.image?.setModel(agent.imageModel)
        noteToolChanges(tools)
        append(ChatMessage(role: .user, text: text))
        guard scheduler.next(config: config, log: messages) == agentID else { return }

        do {
            activity = "Thinking"
            let input = makeInput(agent: agent, tools: tools)
            let context = TurnContext(llm: deps.llm, tools: HubToolExecutor(hub: deps.hub))
            let harness = deps.harness
            try await harness.run(input, context: context) { [weak self] ev in
                Task { @MainActor in self?.handle(ev) }
            }
            // Let queued MainActor handlers drain before returning.
            await Task.yield()
            try? await Task.sleep(nanoseconds: 30_000_000)
        } catch is CancellationError {
            if !streaming.isEmpty { append(ChatMessage(role: .assistant, text: streaming)) }
        } catch {
            errorText = error.localizedDescription
        }
    }

    private func makeInput(agent: Agent, tools: [ToolDef]) -> TurnInput {
        let meta = deps.library.flatMap { lib in
            lib.installedDirectory(for: agent.model).flatMap { ModelMetadata.load(dir: $0) }
        }
        return TurnInput(
            agent: agent, conversationID: id, participants: config.participants, transcript: messages,
            summary: summary, tools: tools,
            settings: ResolvedSettings.resolve(agent: agent, model: meta, machine: deps.settings),
            lastPromptTokens: contextUsed)
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
        if m.speaker == nil, m.role != .marker { m.speaker = m.role == .user ? "user" : agentID }
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
    init(home: ParlorHome, hub: ToolHub, llm: LanguageModelRuntime, harness: Harness, library: ModelLibrary? = nil) {
        self.home = home; self.hub = hub; self.llm = llm; self.harness = harness; self.library = library
    }
}

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
    private var summary: ChatLog.Summary?
    private var task: Task<Void, Never>?
    private var toolSnapshot: Set<String>?
    private unowned let deps: RoomDeps

    init(agentID: String, deps: RoomDeps) {
        self.agentID = agentID
        self.deps = deps
        self.log = ChatLog(dir: deps.home.chatDir(for: agentID))
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

        do {
            try await compactIfNeeded(agent: agent)
            let history = buildHistory(agent: agent, tools: tools)
            activity = "Thinking"
            let loop = deps.loop
            let llm = deps.llm
            let hub = deps.hub
            try await loop.run(agent: agent, history: history, tools: tools, llm: llm, hub: hub, settings: deps.settings) { [weak self] ev in
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

    private func handle(_ ev: LoopEvent) {
        switch ev {
        case .delta(let t): streaming = t; activity = nil
        case .assistant(let m):
            streaming = ""
            if !m.text.isEmpty || m.toolCalls != nil { append(m) }
            activity = "Thinking"
        case .toolStarted(let c): activity = "Running \(c.qualified)"
        case .tool(let m): append(m)
        case .stats(let s): lastStats = s; contextUsed = s.promptTokens + s.completionTokens
        }
    }

    private func append(_ m: ChatMessage) {
        messages.append(m)
        log.append(m)
    }

    private func noteToolChanges(_ tools: [ToolDef]) {
        let now = Set(tools.map(\.qualified))
        defer { toolSnapshot = now }
        guard let before = toolSnapshot ?? lastKnownTools() else { return }
        let added = now.subtracting(before).sorted(), removed = before.subtracting(now).sorted()
        guard !added.isEmpty || !removed.isEmpty else { return }
        var parts: [String] = []
        if !added.isEmpty { parts.append("added " + added.joined(separator: ", ")) }
        if !removed.isEmpty { parts.append("removed " + removed.joined(separator: ", ")) }
        append(.marker("Tools updated: " + parts.joined(separator: "; ")))
    }

    /// After a relaunch, recover the last tool set from the log's own markers is not possible,
    /// so a fresh process starts from the current set without a marker.
    private func lastKnownTools() -> Set<String>? { nil }

    // MARK: context

    func buildHistory(agent: Agent, tools: [ToolDef]) -> [LLMMessage] {
        var system = agent.persona.trimmingCharacters(in: .whitespacesAndNewlines)
        if system.isEmpty { system = "You are \(agent.name)." }
        // Date only: a clock in the system prompt would change every minute and defeat the prompt cache.
        let f = DateFormatter(); f.dateStyle = .full; f.timeStyle = .none
        system += "\n\nToday is \(f.string(from: Date()))."
        if !tools.isEmpty {
            system += "\nYou have tools. When a request needs information or an action from a tool, call the tool immediately in this reply. Never say you will look something up or do something without actually calling the tool. Never ask permission to use a tool for a read-only lookup. After the tool result arrives, answer in plain text."
        }
        if let s = summary { system += "\n\n## Earlier conversation (summary)\n\(s.text)" }

        var live = messages.filter { $0.role != .marker }
        if let through = summary?.through, let i = live.lastIndex(where: { $0.id == through }) {
            live = Array(live[(i + 1)...])
        }
        var out: [LLMMessage] = [LLMMessage(role: .system, text: system)]
        for m in live {
            switch m.role {
            case .user: out.append(LLMMessage(role: .user, text: m.text))
            case .assistant: out.append(LLMMessage(role: .assistant, text: m.text, toolCalls: m.toolCalls ?? []))
            case .tool: out.append(LLMMessage(role: .tool, text: m.text, toolCallID: m.toolCallID, toolName: m.toolName))
            case .marker: break
            }
        }
        return out
    }

    private func compactIfNeeded(agent: Agent) async throws {
        let history = buildHistory(agent: agent, tools: [])
        let used = max(contextUsed, Compactor.estimateTokens(history))
        guard Double(used) > agent.compactAt * Double(agent.contextTokens) else { return }
        try await compact(agent: agent)
    }

    @discardableResult
    public func compact(agent: Agent) async throws -> Bool {
        activity = "Compacting"
        defer { activity = nil }
        guard let s = try await Compactor(keepRecent: deps.settings.loop.keepRecent, summaryTokens: deps.settings.loop.summaryTokens).compact(agent: agent, messages: messages, existing: summary, llm: deps.llm) else { return false }
        let folded = messages.filter { $0.role != .marker }.firstIndex(where: { $0.id == s.through }).map { $0 + 1 } ?? 0
        summary = s
        log.saveSummary(s)
        append(.marker("Earlier conversation compacted · \(folded) messages folded into a short summary"))
        contextUsed = 0
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
    public var loop: AgentLoop
    public var image: ImageProvider?
    public var settings = ParlorSettings()
    init(home: ParlorHome, hub: ToolHub, llm: LanguageModelRuntime, loop: AgentLoop) {
        self.home = home; self.hub = hub; self.llm = llm; self.loop = loop
    }
}

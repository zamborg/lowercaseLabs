import Foundation
import Observation

/// Everything the UI and CLI talk to. Owns the store, tool hub, model runtime and rooms.
@Observable @MainActor
public final class ParlorEngine {
    public let home: ParlorHome
    public let library: ModelLibrary
    public let hub = ToolHub()
    public private(set) var agents: [Agent] = []
    public private(set) var servers: [ServerStatus] = []
    /// Every server in `mcp.json`, including disabled ones.
    public private(set) var configuredServers: [String: MCPServerConfig] = [:]
    public private(set) var tools: [ToolDef] = []
    public var selection: String?
    public private(set) var loadedModel: String?
    public private(set) var imageWorkerState: String = "idle"
    /// Speed, memory and token settings (`settings.json`). Change through `updateSettings`.
    public private(set) var settings = ParlorSettings()

    private var rooms: [String: Room] = [:]
    private let deps: RoomDeps
    private var mcp: MCPManager!
    private var watcher: FolderWatcher?
    private var reloadTask: Task<Void, Never>?
    private let mlx: MLXRuntime?
    private let queue = LLMQueue()
    /// Generations waiting behind the one running.
    public private(set) var queueDepth = 0
    public private(set) var image: ImageProvider?

    /// Pass a runtime to substitute the model (tests); nil uses MLX.
    public init(home: ParlorHome = .default, runtime: LanguageModelRuntime? = nil, harness: Harness = BuiltInHarness()) {
        self.home = home
        let lib = ModelLibrary(home: home)
        self.library = lib
        if let runtime {
            self.mlx = nil
            self.deps = RoomDeps(home: home, hub: hub, llm: QueuedRuntime(inner: runtime, queue: queue), harness: harness, library: lib)
        } else {
            let m = MLXRuntime(library: lib)
            self.mlx = m
            self.deps = RoomDeps(home: home, hub: hub, llm: QueuedRuntime(inner: m, queue: queue), harness: harness, library: lib)
        }
        self.mcp = MCPManager(hub: hub, home: home)
    }

    public var runtime: LanguageModelRuntime { deps.llm }

    // MARK: settings

    /// Change settings from the console or code; saved to `settings.json` and applied at once.
    public func updateSettings(_ change: (inout ParlorSettings) -> Void) {
        var s = settings
        change(&s)
        guard s != settings else { return }
        settings = s
        s.save(home.settingsFile)
        Task { await self.applySettings() }
    }

    public func resetSettings() { updateSettings { $0 = ParlorSettings() } }

    private func applySettings() async {
        deps.settings = settings
        await mlx?.setKeepAlive(minutes: settings.llm.keepLoadedMinutes)
        await mlx?.configure(budgetFraction: settings.memoryBudgetFraction, slots: settings.cacheSlotsPerModel)
        await image?.configure(settings.image)
    }

    // MARK: lifecycle

    public func start() async {
        try? home.ensure()
        seedIfEmpty()
        reloadAgents()
        if selection == nil { selection = agents.first?.id }
        let img = ImageProvider(home: home, library: library)
        image = img
        deps.image = img
        settings = ParlorSettings.load(home.settingsFile)
        if !FileManager.default.fileExists(atPath: home.settingsFile.path) { settings.save(home.settingsFile) }
        await applySettings()
        await hub.register(img, status: ServerStatus(id: "image", kind: "built in", detail: "local image model worker", state: .connecting, toolCount: 0, toolNames: []))
        _ = await hub.onChange { [weak self] in
            Task { @MainActor in await self?.syncFromHub() }
        }
        loadConfig()
        // Servers start in the background: a slow or broken one must not hold up the window.
        Task { await self.mcp.reload(); await self.syncFromHub() }
        await syncFromHub()
        watcher = FolderWatcher(path: home.root) { [weak self] paths in
            Task { @MainActor in self?.filesChanged(paths) }
        }
    }

    public func shutdown() async {
        watcher?.stop()
        await mcp.stopAll()
        await image?.shutdown()
        await deps.llm.unload()
    }

    private func seedIfEmpty() {
        let fm = FileManager.default
        let guide = home.root.appendingPathComponent("CLAUDE.md")
        if !fm.fileExists(atPath: guide.path) { try? Seeds.folderGuide.write(to: guide, atomically: true, encoding: .utf8) }
        if !fm.fileExists(atPath: home.mcpConfig.path) { try? MCPConfigFile().save(home.mcpConfig) }
        // Starter agents are created once. An existing home with agents is adopted without seeding, and
        // deleting every agent later does not bring them back.
        guard !fm.fileExists(atPath: home.seededMarker.path) else { return }
        let existing = (try? fm.contentsOfDirectory(atPath: home.agentsDir.path))?.filter { $0.hasSuffix(".md") } ?? []
        if existing.isEmpty {
            for a in [Seeds.mabel(), Seeds.pip()] { write(a) }
        }
        try? "seeded\n".write(to: home.seededMarker, atomically: true, encoding: .utf8)
    }

    // MARK: agents

    public func reloadAgents() {
        let fm = FileManager.default
        let files = ((try? fm.contentsOfDirectory(atPath: home.agentsDir.path)) ?? []).filter { $0.hasSuffix(".md") }.sorted()
        var out: [Agent] = []
        for f in files {
            let id = String(f.dropLast(3))
            if let text = try? String(contentsOf: home.agentsDir.appendingPathComponent(f), encoding: .utf8) {
                out.append(Agent(id: id, file: text))
            }
        }
        out.sort { ($0.kind == .assistant ? 0 : 1, $0.name.lowercased()) < ($1.kind == .assistant ? 0 : 1, $1.name.lowercased()) }
        if out != agents { agents = out }
        if let s = selection, !agents.contains(where: { $0.id == s }) { selection = agents.first?.id }
        // A chat folder only lingers for a deleted agent when it never got a message: tidy it.
        let live = Set(agents.map(\.id))
        for (id, _) in rooms where !live.contains(id) { rooms[id] = nil }
        if let dirs = try? fm.contentsOfDirectory(atPath: home.chatsDir.path) {
            for d in dirs where !live.contains(d) {
                let dir = home.chatsDir.appendingPathComponent(d)
                if (try? fm.contentsOfDirectory(atPath: dir.path))?.isEmpty == true { try? fm.removeItem(at: dir) }
            }
        }
    }

    public func agent(_ id: String) -> Agent? { agents.first { $0.id == id } }

    private func write(_ a: Agent) {
        try? home.ensure()
        let url = home.agentsDir.appendingPathComponent("\(a.id).md")
        let text = a.fileText()
        if (try? String(contentsOf: url, encoding: .utf8)) != text {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    public func save(_ a: Agent) {
        write(a)
        reloadAgents()
    }

    @discardableResult
    public func create(kind: Agent.Kind, name: String? = nil) -> Agent {
        let base = name ?? (kind == .assistant ? "New assistant" : "New character")
        var id = Agent.slug(base)
        var n = 2
        while agents.contains(where: { $0.id == id }) { id = Agent.slug(base) + "-\(n)"; n += 1 }
        let a = Agent(
            id: id, name: base, summary: "", persona: kind == .assistant
                ? "You are \(base), a helpful personal assistant. Be concise and direct."
                : "You are \(base). Describe who this character is and how they speak.",
            kind: kind, model: kind == .assistant ? Defaults.languageModel : Defaults.characterModel,
            imageModel: Defaults.imageModel, tools: kind == .assistant ? ["*"] : ["image.generate"],
            contextTokens: kind == .assistant ? 16384 : 8192, temperature: kind == .assistant ? 0.5 : 0.9,
            color: agents.count % 6)
        save(a)
        selection = a.id
        return a
    }

    public func delete(_ id: String) {
        try? FileManager.default.removeItem(at: home.agentsDir.appendingPathComponent("\(id).md"))
        try? FileManager.default.removeItem(at: home.chatDir(for: id))
        rooms[id] = nil
        reloadAgents()
    }

    /// Makes the file name and chat folder follow the agent's display name (`agents/<slug>.md`).
    /// Returns the (possibly unchanged) id. Skipped while the agent is mid-reply.
    @discardableResult
    public func renameToMatchName(_ id: String) -> String {
        guard let a = agent(id), rooms[id]?.isBusy != true else { return id }
        let base = Agent.slug(a.name)
        guard !base.isEmpty, base != id else { return id }
        var newID = base; var n = 2
        while agents.contains(where: { $0.id == newID }) || FileManager.default.fileExists(atPath: home.chatDir(for: newID).path) {
            newID = base + "-\(n)"; n += 1
        }
        let fm = FileManager.default
        var moved = a; moved.id = newID
        write(moved)
        try? fm.removeItem(at: home.agentsDir.appendingPathComponent("\(id).md"))
        if fm.fileExists(atPath: home.chatDir(for: id).path) {
            try? fm.moveItem(at: home.chatDir(for: id), to: home.chatDir(for: newID))
        }
        rooms[id] = nil
        let wasSelected = selection == id
        reloadAgents()
        if wasSelected { selection = newID }
        return newID
    }

    public func duplicate(_ id: String) {
        guard var a = agent(id) else { return }
        a.name += " copy"
        var newID = Agent.slug(a.name); var n = 2
        while agents.contains(where: { $0.id == newID }) { newID = Agent.slug(a.name) + "-\(n)"; n += 1 }
        a.id = newID
        save(a)
        selection = newID
    }

    // MARK: rooms

    public func room(for agentID: String) -> Room {
        if let r = rooms[agentID] { return r }
        let r = Room(agentID: agentID, deps: deps)
        rooms[agentID] = r
        return r
    }

    public func send(_ text: String, to agentID: String) {
        guard let a = agent(agentID) else { return }
        room(for: agentID).send(text, agent: a)
        Task { await self.refreshLoaded() }
    }

    public func refreshLoaded() async {
        loadedModel = await deps.llm.loadedModel()
        queueDepth = await queue.depth
        if let image { imageWorkerState = await image.stateDescription() }
    }

    // MARK: tools

    public func restartServer(_ id: String) { Task { await mcp.restart(id) } }
    public func reloadTools() { Task { await mcp.reload(); self.loadConfig() } }

    public func saveMCPConfig(_ file: MCPConfigFile) throws {
        try file.save(home.mcpConfig)
        configuredServers = file.mcpServers
        reloadTools()
    }

    public func loadConfig() { configuredServers = MCPConfigFile.load(home.mcpConfig).mcpServers }

    /// Turn a server off or on without losing its settings (`"disabled": true` in mcp.json).
    public func setServer(_ id: String, enabled: Bool) {
        var f = MCPConfigFile.load(home.mcpConfig)
        guard var cfg = f.mcpServers[id] else { return }
        cfg.disabled = enabled ? nil : true
        f.mcpServers[id] = cfg
        try? saveMCPConfig(f)
    }

    /// Remove a server from mcp.json and stop it.
    public func deleteServer(_ id: String) {
        var f = MCPConfigFile.load(home.mcpConfig)
        f.mcpServers[id] = nil
        try? saveMCPConfig(f)
    }

    private func syncFromHub() async {
        servers = await hub.allStatuses()
        tools = await hub.allTools()
    }

    // MARK: file changes

    /// `realpath` resolves /var → /private/var, which Foundation's own resolver does not.
    nonisolated static func realPath(_ p: String) -> String {
        guard let r = realpath(p, nil) else { return p }
        defer { free(r) }
        return String(cString: r)
    }

    private func filesChanged(_ paths: [String]) {
        // FSEvents reports resolved paths (/private/var/…), so compare against resolved prefixes too.
        let prefixes = [home.agentsDir.path, Self.realPath(home.agentsDir.path)]
        let relevant = paths.contains { p in prefixes.contains { p.hasPrefix($0) } || p.hasSuffix("/mcp.json") }
        _ = relevant
        let chats = paths.filter { $0.contains("/chats/") }
        reloadTask?.cancel()
        reloadTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 200_000_000)
            if relevant {
                self.reloadAgents()
                if paths.contains(where: { $0.hasSuffix("/mcp.json") }) { await self.mcp.reload(); self.loadConfig() }
            }
            if paths.contains(where: { $0.hasSuffix("/settings.json") }) {
                let fresh = ParlorSettings.load(self.home.settingsFile)
                if fresh != self.settings { self.settings = fresh; await self.applySettings() }
            }
            for p in chats {
                // chats/<id>/log.jsonl edited from outside
                let comps = p.components(separatedBy: "/chats/")
                if let id = comps.last?.components(separatedBy: "/").first { self.rooms[id]?.reloadFromDisk() }
            }
        }
    }
}

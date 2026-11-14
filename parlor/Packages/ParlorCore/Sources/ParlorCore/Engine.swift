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
    /// Group rooms (conversations with several agents or another scheduler). One-to-one chats are the agents.
    public private(set) var conversations: [ConversationInfo] = []
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
    /// Everything supervised: the language model, the image worker and each mounted server.
    public private(set) var services: [ServiceInfo] = []
    /// The command layer: every control as a named, tiered verb.
    public private(set) var commands: CommandCenter!
    /// The `parlor` MCP endpoint on 127.0.0.1.
    public private(set) var endpoint: ParlorMCPEndpoint!
    public private(set) var serverError: String?
    /// Modules found in `modules/`, approved or not.
    public private(set) var modules: [ModuleInfo] = []
    private var moduleTokens: [String: String] = [:]
    /// Asked before an agent's or token's destructive verb runs. nil means nobody can answer, so it is refused.
    public var confirm: (@MainActor (String) async -> Bool)?

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
        deps.agentLookup = { [weak self] id in self?.agent(id) }
        self.commands = CommandCenter(engine: self)
        self.endpoint = ParlorMCPEndpoint(engine: self)
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
        await syncServer()
    }

    /// Start, stop or move the `parlor` server to match settings.
    private func syncServer() async {
        let want = settings.parlorServer
        if !want.enabled { endpoint.stop(); serverError = nil; return }
        if endpoint.isRunning, want.port == 0 || endpoint.port == want.port { return }
        do { try await endpoint.start(port: want.port); serverError = nil }
        catch { endpoint.stop(); serverError = error.localizedDescription }
    }

    // MARK: modules

    public func reloadModules() {
        let found = Modules.scan(home: home, approvals: settings.moduleApprovals)
        if found != modules { modules = found }
    }

    /// Launch configs of approved modules, with a per-process token at the tier the manifest asks for.
    func moduleConfigs() -> [String: MCPServerConfig] {
        var out: [String: MCPServerConfig] = [:]
        for m in modules where m.approved && m.manifest != nil {
            var token: String?
            if m.manifest!.tier != .none, endpoint.isRunning {
                if moduleTokens[m.name] == nil { moduleTokens[m.name] = endpoint.issueEphemeral(name: "module:\(m.name)", tier: m.manifest!.tier) }
                token = moduleTokens[m.name]
            }
            let url = endpoint.isRunning ? "http://127.0.0.1:\(endpoint.port)/mcp" : nil
            if let c = Modules.config(for: m, parlorURL: url, token: token) { out[m.name] = c }
        }
        return out
    }

    /// The user reviewed the manifest and allows this module to run.
    public func approveModule(_ name: String) {
        guard let m = modules.first(where: { $0.name == name }), m.manifest != nil else { return }
        updateSettings { $0.moduleApprovals[name] = m.hash }
        reloadModules()
        Task { await self.mcp.reload(); await self.refreshServices() }
    }

    public func revokeModule(_ name: String) {
        updateSettings { $0.moduleApprovals[name] = nil }
        reloadModules()
        Task { await self.mcp.reload(); await self.refreshServices() }
    }

    public func setModuleValues(_ name: String, _ values: [String: JSON]) {
        guard let m = modules.first(where: { $0.name == name }) else { return }
        Modules.saveValues(m, values)
        reloadModules()
        Task { await self.mcp.reload(); await self.refreshServices() }
    }

    // MARK: access tokens

    /// Make a token. The secret is returned once; only its hash is stored.
    public func createToken(name: String, tier: Tier, expires: Date? = nil) -> String? {
        let n = name.trimmingCharacters(in: .whitespaces)
        guard !n.isEmpty, !settings.tokens.contains(where: { $0.name == n }) else { return nil }
        let (secret, token) = APIToken.make(name: n, tier: tier, expires: expires)
        updateSettings { $0.tokens.append(token) }
        return secret
    }

    public func revokeToken(_ name: String) { updateSettings { $0.tokens.removeAll { $0.name == name } } }

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
        let provider = CommandProvider(center: commands)
        await hub.setTierRequirements(commands.tiers())
        await hub.register(provider, status: ServerStatus(id: "parlor", kind: "built in", detail: "controls for the app itself", state: .connecting, toolCount: 0, toolNames: []))
        await hub.register(img, status: ServerStatus(id: "image", kind: "built in", detail: "local image model worker", state: .connecting, toolCount: 0, toolNames: []))
        _ = await hub.onChange { [weak self] in
            Task { @MainActor in await self?.syncFromHub() }
        }
        loadConfig()
        let guide = home.modulesDir.appendingPathComponent("CLAUDE.md")
        if !FileManager.default.fileExists(atPath: guide.path) {
            try? FileManager.default.createDirectory(at: home.modulesDir, withIntermediateDirectories: true)
            try? Seeds.moduleGuide.write(to: guide, atomically: true, encoding: .utf8)
        }
        reloadModules()
        await mcp.setModuleProvider { [weak self] in await self?.moduleConfigs() ?? [:] }
        // Servers start in the background: a slow or broken one must not hold up the window.
        Task { await self.mcp.reload(); await self.syncFromHub() }
        await syncFromHub()
        watcher = FolderWatcher(path: home.root) { [weak self] paths in
            Task { @MainActor in self?.filesChanged(paths) }
        }
    }

    public func shutdown() async {
        watcher?.stop()
        endpoint.stop()
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
        reloadConversations()
        let live = Set(agents.map(\.id)).union(conversations.map(\.id))
        for (id, _) in rooms where !live.contains(id) { rooms[id] = nil }
        if let dirs = try? fm.contentsOfDirectory(atPath: home.chatsDir.path) {
            for d in dirs where !live.contains(d) {
                let dir = home.chatsDir.appendingPathComponent(d)
                if (try? fm.contentsOfDirectory(atPath: dir.path))?.isEmpty == true { try? fm.removeItem(at: dir) }
            }
        }
    }

    public func agent(_ id: String) -> Agent? { agents.first { $0.id == id } }

    // MARK: rooms

    /// Rescan `chats/` for folders whose `conversation.json` makes them a room.
    public func reloadConversations() {
        let fm = FileManager.default
        var out: [ConversationInfo] = []
        for d in (try? fm.contentsOfDirectory(atPath: home.chatsDir.path)) ?? [] {
            let dir = home.chatDir(for: d)
            guard fm.fileExists(atPath: dir.appendingPathComponent("conversation.json").path) else { continue }
            let c = ConversationConfig.load(dir: dir, agentID: d)
            if c.isRoom { out.append(ConversationInfo(id: d, config: c)) }
        }
        out.sort { $0.title.lowercased() < $1.title.lowercased() }
        if out != conversations { conversations = out }
    }

    /// Create a room from existing agents. Returns its id (also its chat folder).
    @discardableResult
    public func createRoom(title: String, agents ids: [String], scheduler: SchedulerSpec) -> String {
        let slug = Agent.slug(title.isEmpty ? "room" : title)
        let suffix = String(UUID().uuidString.prefix(4)).lowercased()
        let id = "\(slug)-\(suffix)"
        let config = ConversationConfig(participants: ids.map { .agent($0) } + [.user], scheduler: scheduler, title: title.isEmpty ? "Room" : title)
        try? home.ensure()
        try? config.save(dir: home.chatDir(for: id))
        reloadConversations()
        return id
    }

    public func deleteRoom(_ id: String) {
        guard conversations.contains(where: { $0.id == id }) else { return }
        rooms[id] = nil
        try? FileManager.default.removeItem(at: home.chatDir(for: id))
        reloadConversations()
    }

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

    /// `id` is an agent (one-to-one chat) or a room.
    public func send(_ text: String, to id: String) {
        if conversations.contains(where: { $0.id == id }) {
            room(for: id).send(text)
        } else if let a = agent(id) {
            room(for: id).send(text, agent: a)
        } else { return }
        Task { await self.refreshLoaded() }
    }

    public func refreshLoaded() async {
        loadedModel = await deps.llm.loadedModel()
        queueDepth = await queue.depth
        if let image { imageWorkerState = await image.stateDescription() }
    }

    // MARK: tools

    public func restartServer(_ id: String) { Task { await mcp.restart(id); await refreshServices() } }
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

    // MARK: services

    /// Rebuild the dashboard rows. Called by the System screen on a timer and after every control action.
    public func refreshServices() async {
        var out: [ServiceInfo] = []
        let now = Date()
        // Language models.
        var loaded: [String] = []
        if let mlx { loaded = await mlx.loadedModels() } else if let m = await deps.llm.loadedModel() { loaded = [m] }
        let bytes = await mlx?.residentBytes()
        let depth = await queue.depth
        out.append(ServiceInfo(
            id: "llm", kind: "language model", title: "Language models",
            detail: loaded.isEmpty ? "loads on the first message" : loaded.map { $0.split(separator: "/").last.map(String.init) ?? $0 }.joined(separator: ", "),
            state: loaded.isEmpty ? .stopped : (depth > 0 ? .running : .idle),
            memoryBytes: bytes.flatMap { $0 > 0 ? UInt64($0) : nil }, lastUsed: nil, queueDepth: depth, lifecycle: "onDemand",
            restarts: 0, canStart: false, canStop: !loaded.isEmpty, canRestart: false, logURL: nil))
        // Image worker.
        if let image {
            let s = await image.serviceSnapshot()
            out.append(ServiceInfo(
                id: "image", kind: "image", title: "Image worker",
                detail: s.model.map { "loaded · " + ($0.split(separator: "/").last.map(String.init) ?? $0) } ?? "starts when an agent draws",
                state: s.state, memoryBytes: s.pid.flatMap { ProcessMemory.footprint(pid: $0) }, lastUsed: s.lastUsed, queueDepth: nil,
                lifecycle: "onDemand", restarts: 0, canStart: !s.state.isAlive, canStop: s.state.isAlive, canRestart: false,
                logURL: ServiceLog.url(home: home, service: "image")))
        }
        // Mounted MCP servers.
        for m in await mcp.snapshot() {
            var state = m.state
            if state == .running, let t = m.lastUsed, now.timeIntervalSince(t) > 30 { state = .idle }
            let cfg = m.config
            let pid = await mcp.pid(m.id)
            out.append(ServiceInfo(
                id: (m.isModule ? "module:" : "mcp:") + m.id, kind: m.isModule ? "module" : "mcp", title: m.id,
                detail: cfg.url ?? ([cfg.command ?? ""] + (cfg.args ?? [])).joined(separator: " "),
                state: state, memoryBytes: pid.flatMap { ProcessMemory.footprint(pid: $0) }, lastUsed: m.lastUsed, queueDepth: nil,
                lifecycle: cfg.isOnDemand ? "onDemand" : "always", restarts: m.restarts,
                canStart: !state.isAlive && cfg.disabled != true, canStop: state.isAlive, canRestart: cfg.disabled != true,
                logURL: ServiceLog.url(home: home, service: (m.isModule ? "module:" : "mcp:") + m.id)))
        }
        let ps = settings.parlorServer
        let psState: ServiceState = !ps.enabled ? .stopped : (serverError.map { .failed($0) } ?? (endpoint.isRunning ? .running : .starting))
        out.append(ServiceInfo(
            id: "parlorServer", kind: "server", title: "Parlor server",
            detail: ps.enabled ? "http://127.0.0.1:\(ps.port)/mcp · \(settings.tokens.count) token\(settings.tokens.count == 1 ? "" : "s")" : "off",
            state: psState, memoryBytes: nil, lastUsed: nil, queueDepth: nil, lifecycle: "always", restarts: 0,
            canStart: !ps.enabled, canStop: ps.enabled, canRestart: ps.enabled, logURL: nil))
        services = out
    }

    public enum ServiceAction { case start, stop, restart }

    public func controlService(_ id: String, _ action: ServiceAction) async throws {
        if id == "parlorServer" {
            switch action {
            case .start: updateSettings { $0.parlorServer.enabled = true }
            case .stop: updateSettings { $0.parlorServer.enabled = false }
            case .restart: endpoint.stop()
            }
            await syncServer(); await refreshServices(); return
        }
        let isMCP = id.hasPrefix("mcp:") || id.hasPrefix("module:")
        let name = id.split(separator: ":", maxSplits: 1).last.map(String.init) ?? id
        guard id == "llm" || id == "image" || isMCP else { throw CommandError("No service \"\(id)\". Use services.list.") }
        if isMCP, !(await mcp.snapshot().contains { $0.id == name }) { throw CommandError("No service \"\(id)\". Use services.list.") }
        switch action {
        case .start:
            if id == "image" { try await image?.start() } else if isMCP { await mcp.start(name) }
        case .stop:
            if id == "llm" { await unloadModels() } else if id == "image" { await image?.stop() } else { await mcp.stop(name) }
        case .restart:
            if id == "image" { await image?.stop(); try await image?.start() } else if isMCP { await mcp.restart(name) }
        }
        await refreshServices(); await refreshLoaded()
    }

    public func startService(_ id: String) { Task { try? await controlService(id, .start) } }
    public func stopService(_ id: String) { Task { try? await controlService(id, .stop) } }
    public func restartService(_ id: String) { Task { try? await controlService(id, .restart) } }

    // MARK: language model control

    public func loadedModels() async -> [String] {
        if let mlx { return await mlx.loadedModels() }
        return (await deps.llm.loadedModel()).map { [$0] } ?? []
    }

    public func unloadModels() async { await deps.llm.unload(); await refreshLoaded() }

    /// Warm a model by asking it for one token.
    public func loadModel(_ repo: String) async throws {
        let req = LLMRequest(messages: [LLMMessage(role: .user, text: "Hi")], temperature: 0, maxTokens: 1, priority: .background)
        for try await _ in deps.llm.generate(model: repo, request: req) {}
        await refreshLoaded()
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
            if paths.contains(where: { $0.contains("/modules/") }) { self.reloadModules(); await self.mcp.reload() }
            if paths.contains(where: { $0.hasSuffix("/settings.json") }) {
                let fresh = ParlorSettings.load(self.home.settingsFile)
                if fresh != self.settings { self.settings = fresh; await self.applySettings() }
            }
            if chats.contains(where: { $0.hasSuffix("conversation.json") }) { self.reloadConversations() }
            for p in chats {
                // chats/<id>/log.jsonl edited from outside
                let comps = p.components(separatedBy: "/chats/")
                if let id = comps.last?.components(separatedBy: "/").first { self.rooms[id]?.reloadFromDisk() }
            }
        }
    }
}

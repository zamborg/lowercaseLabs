import Foundation
import Observation

/// Everything the UI and CLI talk to. Owns the store, tool hub, model runtime and rooms.
@Observable @MainActor
public final class ParlorEngine {
    public let home: ParlorHome
    public let library: ModelLibrary
    public let hub = ToolHub()
    public let supervisor = Supervisor()
    public internal(set) var engineList: [EngineManifest] = []
    public internal(set) var hostList: [HostInfo] = []
    @ObservationIgnored public lazy var engineStore = NounStore<EngineManifest>(dir: home.enginesDir, layout: .folder("engine.json"))
    @ObservationIgnored public lazy var hostStore = NounStore<HostSpec>(dir: home.hostsDir, layout: .file)
    @ObservationIgnored public lazy var harnessStore = NounStore<HarnessManifest>(dir: home.harnessesDir, layout: .folder("harness.json"))
    @ObservationIgnored public lazy var agentStore = NounStore<AgentRecord>(dir: home.agentsDir, layout: .folder("agent.json"))
    public internal(set) var harnessList: [HarnessManifest] = []
    var servicesRegistered = false
    public internal(set) var agents: [Agent] = []
    public internal(set) var servers: [ServerStatus] = []
    /// Every server in `mcp.json`, including disabled ones.
    public internal(set) var configuredServers: [String: MCPServerConfig] = [:]
    public internal(set) var tools: [ToolDef] = []
    public var selection: String?
    /// Group rooms (conversations with several agents or another scheduler). One-to-one chats are the agents.
    public internal(set) var conversations: [ConversationInfo] = []
    public internal(set) var loadedModel: String?
    public internal(set) var imageWorkerState: String = "idle"
    /// Speed, memory and token settings (`settings.json`). Change through `updateSettings`.
    public internal(set) var settings = ParlorSettings()

    var rooms: [String: Room] = [:]
    let deps: RoomDeps
    var mcp: MCPManager!
    var watcher: FolderWatcher?
    var reloadTask: Task<Void, Never>?
    let mlx: MLXRuntime?
    /// Sends `provider/model` names to their provider and everything else to MLX.
    public let router: ProviderRouter
    /// The routes the gateway serves under `/v1` (the one server also serves `/mcp`).
    public internal(set) var gateway: GatewayRoutes!
    var gatewayForHarness = false
    /// Runs agents that have a `harness:` command.
    public let externalHarness: ExternalHarness
    var harnessTokens: [String: String] = [:]
    public var memory: MemoryStore { MemoryStore(home: home) }
    /// One queue per host.
    let queues = HostQueues()
    /// Generations waiting behind the one running.
    public internal(set) var queueDepth = 0
    public internal(set) var image: ImageProvider?
    /// Everything supervised: the language model, the image worker and each mounted server.
    public internal(set) var services: [ServiceInfo] = []
    /// The command layer: every control as a named, tiered verb.
    public internal(set) var commands: CommandCenter!
    /// The `parlor` MCP endpoint on 127.0.0.1.
    public internal(set) var endpoint: ParlorMCPEndpoint!
    public internal(set) var serverError: String?
    /// Modules found in `modules/`, approved or not.
    public internal(set) var modules: [ModuleInfo] = []
    var moduleTokens: [String: String] = [:]
    /// Asked before an agent's or token's destructive verb runs. nil means nobody can answer, so it is refused.
    public var confirm: (@MainActor (String) async -> Bool)?

    /// Pass a runtime to substitute the model (tests); nil uses MLX.
    public init(home: ParlorHome = .default, runtime: LanguageModelRuntime? = nil) {
        self.home = home
        let lib = ModelLibrary(home: home)
        self.library = lib
        let base: LanguageModelRuntime
        if let runtime { self.mlx = nil; base = runtime } else { let m = MLXRuntime(library: lib); self.mlx = m; base = m }
        let router = ProviderRouter(inner: base)
        self.router = router
        let envBox = EnvBox()
        let external = ExternalHarness(home: home) { agent in await envBox.provider?(agent) ?? [:] }
        self.externalHarness = external
        self.deps = RoomDeps(home: home, hub: hub, llm: QueuedRuntime(inner: router, queues: queues, key: { router.queueKey($0) }),
                             external: external, library: lib)
        envBox.provider = { [weak self] agent in await self?.harnessEnvironment(agent) ?? [:] }
        self.mcp = MCPManager(hub: hub, home: home)
        deps.agentLookup = { [weak self] id in self?.agent(id) }
        deps.launch = { [weak self] a in self?.launch(for: a) }
        deps.gatewayURL = { [weak self] in
            guard let self, self.endpoint.isRunning else { return "" }
            return "http://127.0.0.1:\(self.endpoint.port)/v1"
        }
        self.commands = CommandCenter(engine: self)
        self.endpoint = ParlorMCPEndpoint(engine: self)
        self.gateway = GatewayRoutes(engine: self)
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

    func applySettings() async {
        deps.settings = settings
        await mlx?.setKeepAlive(minutes: settings.llm.keepLoadedMinutes)
        await mlx?.configure(budgetFraction: settings.memoryBudgetFraction, slots: settings.cacheSlotsPerModel)
        await image?.configure(settings.image)
        reloadRunplane()
        await syncServer()
    }

    /// Bring the gateway up for a harness even when it is switched off, on the configured port or any free one.
    func ensureGateway() async {
        if endpoint.isRunning { return }
        gatewayForHarness = true
        if (try? await endpoint.start(port: settings.gateway.port)) == nil {
            do { try await endpoint.start(port: 0); serverError = nil } catch { serverError = error.localizedDescription }
        } else { serverError = nil }
    }

    /// What a launched harness gets in its environment (SPEC §2.5). One secret serves both servers, at the agent's tier.
    func harnessEnvironment(_ agent: Agent) async -> [String: String] {
        await ensureGateway()
        // The token reaches only the hosts the agent names, so a harness cannot go round its tool list to other models.
        // A name that is not a known host (a hand-written provider, a test runtime) leaves the token unscoped.
        let cat = catalog
        let refs = agent.hosts.values.filter { !$0.isEmpty }
        let resolved = refs.compactMap { cat.resolve($0)?.id }
        let scope: [String]? = (!refs.isEmpty && resolved.count == refs.count) ? Array(Set(resolved)).sorted() : nil
        let key = agent.id + ":" + agent.tier.rawValue + ":" + (scope?.joined(separator: ",") ?? "*")
        if harnessTokens[key] == nil { harnessTokens[key] = endpoint.issueEphemeral(name: "harness:" + agent.id, tier: agent.tier, hosts: scope) }
        let token = harnessTokens[key]!
        let parlorPort = endpoint.isRunning ? endpoint.port : settings.gateway.port
        let base = "http://127.0.0.1:\(parlorPort)/v1"
        return [
            "PARLOR_GATEWAY_URL": base, "OPENAI_BASE_URL": base, "OPENAI_API_KEY": token,
            "PARLOR_MCP_URL": "http://127.0.0.1:\(parlorPort)/mcp", "PARLOR_URL": "http://127.0.0.1:\(parlorPort)/mcp",
            "PARLOR_TOKEN": token, "PARLOR_AGENT": agent.id, "PARLOR_HOME": home.root.path,
            "PARLOR_WORKSPACE": externalHarness.workspace(for: agent.id).path]
    }

    // MARK: lifecycle

    public func start() async {
        try? home.ensure()
        settings = ParlorSettings.load(home.settingsFile)
        if !FileManager.default.fileExists(atPath: home.settingsFile.path) { settings.save(home.settingsFile) }
        EngineSeeds.seed(home: home)
        HarnessSeeds.seed(home: home)
        library.onChange = { [weak self] in Task { @MainActor in self?.reloadRunplane() } }
        reloadRunplane()
        seedIfEmpty()
        reloadAgents()
        if selection == nil { selection = agents.first?.id }
        let img = ImageProvider(home: home, library: library,
                                catalog: { [weak self] in await MainActor.run { self?.catalog ?? Catalog(models: [], engines: [], hosts: []) } },
                                hostFor: { [weak self] agent in await self?.defaultImageHost(for: agent) })
        image = img
        deps.image = img
        await applySettings()
        let provider = CommandProvider(center: commands)
        await hub.setTierRequirements(commands.tiers())
        await hub.register(provider, status: ServerStatus(id: "parlor", kind: "built in", detail: "controls for the app itself", state: .connecting, toolCount: 0, toolNames: []))
        await hub.register(img, status: ServerStatus(id: "image", kind: "built in", detail: "local image model worker", state: .connecting, toolCount: 0, toolNames: []))
        _ = await hub.onChange { [weak self] in
            Task { @MainActor in await self?.syncFromHub() }
        }
        loadConfig()
        seedModuleFiles()
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
        externalHarness.stopAll()
        router.stopAll()
        await mcp.stopAll()
        await image?.shutdown()
        await deps.llm.unload()
    }

    func seedIfEmpty() {
        let fm = FileManager.default
        let guide = home.root.appendingPathComponent("CLAUDE.md")
        if !fm.fileExists(atPath: guide.path) { try? Seeds.folderGuide.write(to: guide, atomically: true, encoding: .utf8) }
        if !fm.fileExists(atPath: home.mcpConfig.path) { try? MCPConfigFile().save(home.mcpConfig) }
        // Starter agents are created once. An existing home with agents is adopted without seeding, and
        // deleting every agent later does not bring them back.
        guard !fm.fileExists(atPath: home.seededMarker.path) else { return }
        if agentStore.ids().isEmpty {
            for var a in [Seeds.mabel(), Seeds.pip(), Seeds.builder()] {
                a.chatHost = availableModel(a.chatHost)
                if let img = a.imageHost { a.imageHost = availableImageHost(img) }
                write(a)
            }
        }
        try? "seeded\n".write(to: home.seededMarker, atomically: true, encoding: .utf8)
    }

    // MARK: file changes

    /// `realpath` resolves /var → /private/var, which Foundation's own resolver does not.
    nonisolated static func realPath(_ p: String) -> String {
        guard let r = realpath(p, nil) else { return p }
        defer { free(r) }
        return String(cString: r)
    }

    func filesChanged(_ paths: [String]) {
        // FSEvents reports resolved paths (/private/var/…); `folders` compares both spellings.
        let folders = NounRegistry.folders(changed: paths, home: home)
        let chats = paths.filter { $0.contains("/chats/") }
        reloadTask?.cancel()
        reloadTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 200_000_000)
            if !folders.isDisjoint(with: ["engine", "host", "harness"]) { self.reloadRunplane() }
            if !folders.isEmpty { self.reloadAgents() }
            if paths.contains(where: { $0.hasSuffix("/mcp.json") }) { await self.mcp.reload(); self.loadConfig() }
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

/// Lets the harness host ask the engine for an environment without the engine existing yet.
final class EnvBox: @unchecked Sendable {
    var provider: (@Sendable (Agent) async -> [String: String])?
}

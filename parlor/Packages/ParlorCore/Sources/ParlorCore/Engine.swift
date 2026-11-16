import Foundation
import Observation

/// Everything the UI and CLI talk to. Owns the store, tool hub, model runtime and rooms.
@Observable @MainActor
public final class ParlorEngine {
    public let home: ParlorHome
    public let library: ModelLibrary
    public let hub = ToolHub()
    public let supervisor = Supervisor()
    public private(set) var engineList: [EngineManifest] = []
    public private(set) var hostList: [HostInfo] = []
    @ObservationIgnored public lazy var engineStore = NounStore<EngineManifest>(dir: home.enginesDir, layout: .folder("engine.json"))
    @ObservationIgnored public lazy var hostStore = NounStore<HostSpec>(dir: home.hostsDir, layout: .file)
    @ObservationIgnored public lazy var harnessStore = NounStore<HarnessManifest>(dir: home.harnessesDir, layout: .folder("harness.json"))
    @ObservationIgnored public lazy var agentStore = NounStore<AgentRecord>(dir: home.agentsDir, layout: .folder("agent.json"))
    public private(set) var harnessList: [HarnessManifest] = []
    private var servicesRegistered = false
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
    let deps: RoomDeps
    private var mcp: MCPManager!
    private var watcher: FolderWatcher?
    private var reloadTask: Task<Void, Never>?
    let mlx: MLXRuntime?
    /// Sends `provider/model` names to their provider and everything else to MLX.
    public let router: ProviderRouter
    /// The routes the gateway serves under `/v1` (the one server also serves `/mcp`).
    public private(set) var gateway: GatewayRoutes!
    private var gatewayForHarness = false
    /// Runs agents that have a `harness:` command.
    public let externalHarness: ExternalHarness
    private var harnessTokens: [String: String] = [:]
    public var memory: MemoryStore { MemoryStore(home: home) }
    /// One queue per host.
    let queues = HostQueues()
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

    private func applySettings() async {
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

    /// Models providers answered with on their last test, for pickers (`provider/model`).
    public private(set) var providerModelCache: [String: [String]] = [:]

    /// Every provider model an agent could use, configured or learned from a test.
    public var providerModelChoices: [String] {
        var out: [String] = []
        for c in settings.providers where c.enabled && c.problem == nil {
            let names = (c.models?.isEmpty == false) ? c.models!.map { c.id + "/" + $0 } : (providerModelCache[c.id] ?? [])
            out += names
        }
        return out
    }

    // MARK: engines, hosts and the catalog

    /// Everything that decides availability, as one value. Rebuilt from what is loaded; nothing else answers "is it there?".
    public var catalog: Catalog {
        var states: [String: ServiceState] = [:]
        for h in hostList where h.spec.engine != "mlx" {
            if let proc = router.provider(h.id)?.process { states[h.id] = proc.snapshot().state }
        }
        var dirs: [String: URL] = [:]
        for e in engineList { dirs[e.id] = e.directory }
        return Catalog(models: library.installed, engines: engineList, hosts: hostList, states: states, engineDirs: dirs)
    }

    /// Reload engines and hosts from disk, recompute the auto hosts, and re-point the runtime at them.
    public func reloadRunplane() {
        var engines = engineStore.list()
        for i in engines.indices { engines[i].directory = home.enginesDir.appendingPathComponent(engines[i].id) }
        engineList = engines
        var harnesses = harnessStore.list()
        for i in harnesses.indices { harnesses[i].directory = home.harnessesDir.appendingPathComponent(harnesses[i].id) }
        harnessList = harnesses
        hostList = HostRegistry.build(explicit: hostStore.list(), installed: library.installed, engines: engines)
        let cat = catalog
        var alias: [String: String] = [:]
        for m in library.installed {
            guard let h = cat.host(forModel: m.repo), let e = cat.engine(h.spec.engine), e.adapter == "http-openai" else { continue }
            alias[m.repo] = h.id + "/" + h.id
        }
        router.setAliases(alias)
        router.configure(settings.providers + derivedProviders(cat), home: home)
    }

    /// Providers for hosts that run on an http-openai engine. Derived on every load and never written to settings.
    private func derivedProviders(_ cat: Catalog) -> [ProviderConfig] {
        var out: [ProviderConfig] = []
        let taken = Set(settings.providers.map(\.id))
        var used = Set(settings.providers.compactMap { URL(string: $0.baseURL)?.port })
        used.insert(11500)
        for h in cat.hosts.sorted(by: { $0.id < $1.id }) where !taken.contains(h.id) {
            guard let e = cat.engine(h.spec.engine), e.adapter == "http-openai" else { continue }
            if e.remote {
                guard let url = h.spec.url, !url.isEmpty else { continue }
                out.append(ProviderConfig(id: h.id, baseURL: url, keyRef: h.spec.keyRef, models: h.spec.models, locality: .remote, enabled: h.spec.enabled))
                continue
            }
            guard h.spec.enabled, let launch = e.launch, let repo = h.spec.model, let m = cat.model(repo),
                  let bin = cat.binary(for: e) ?? EngineBinary.find(launch.command, engineDir: e.directory) else { continue }
            var port = 8100 + Int(Self.stableHash(h.id) % 800)
            while used.contains(port) { port += 1 }
            used.insert(port)
            var values: [String: JSON] = ["host": .string(h.id), "port": .int(port), "model.dir": .string(m.directory.path)]
            if let f = m.file { values["model.file"] = .string(m.directory.appendingPathComponent(f).path) }
            for (k, v) in SchemaCheck.defaults(e.settings) { values["settings." + k] = v }
            for (k, v) in h.spec.settings { values["settings." + k] = v }
            let (args, missing) = ArgTemplate.expand(launch.args, values: values)
            guard missing.isEmpty else { continue }
            let cmd = ([bin.path] + args).map { "'" + $0.replacingOccurrences(of: "'", with: "'\\''") + "'" }.joined(separator: " ")
            out.append(ProviderConfig(id: h.id, baseURL: "http://127.0.0.1:\(port)/v1", models: [h.id],
                                      service: .init(command: cmd, idleMinutes: e.lifecycle.idleMinutes, startupSeconds: e.lifecycle.startupSeconds)))
        }
        return out
    }

    static func stableHash(_ s: String) -> UInt64 {
        var h: UInt64 = 14695981039346656037
        for b in s.utf8 { h = (h ^ UInt64(b)) &* 1099511628211 }
        return h
    }

    public func setProviderEnabled(_ id: String, _ on: Bool) {
        updateSettings { if let i = $0.providers.firstIndex(where: { $0.id == id }) { $0.providers[i].enabled = on } }
    }

    /// Ask a provider for its models (starting its server if it has one). A readable line either way.
    public func testProvider(_ id: String) async -> String {
        guard let p = router.provider(id) else {
            if let c = settings.providers.first(where: { $0.id == id }), let why = c.problem { return "Misconfigured: \(why)" }
            return "No such provider."
        }
        guard p.config.enabled else { return "Off. Turn it on first." }
        do {
            let ms = try await p.check()
            providerModelCache[id] = ms.map(\.qualified)
            await refreshServices()
            return ms.isEmpty ? "Reachable, but it lists no models." : "Reachable · " + ms.prefix(4).map(\.id).joined(separator: ", ") + (ms.count > 4 ? " +\(ms.count - 4)" : "")
        } catch { return "Failed: " + error.localizedDescription }
    }

    /// Every model the runtime API offers, as `provider/model`.
    public func runtimeModelNames() async -> [String] {
        var out: [String] = []
        if mlx != nil { out += library.installed(kind: .language).map { "mlx/" + $0.repo } }
        for p in router.all where p.config.enabled {
            if let m = p.config.models, !m.isEmpty { out += m.map { p.config.id + "/" + $0 } }
            else if p.process == nil || p.process!.snapshot().state.isAlive, let ms = try? await p.models() { out += ms.map(\.qualified) }
        }
        return out
    }

    /// Whether a name could be served: an enabled provider's, an installed MLX model, or anything under a substitute runtime.
    func runtimeKnowsModel(_ name: String) -> Bool {
        if router.route(name) != nil { return true }
        guard mlx != nil else { return true }
        let repo = name.hasPrefix("mlx/") ? String(name.dropFirst(4)) : name
        return catalog.isInstalled(repo)
    }

    /// The image host an agent draws with: its `image_model` through the catalog, else the first usable image host (Z-Image first).
    func defaultImageHost(for agentID: String?) -> String? {
        let cat = catalog
        if let id = agentID, let a = agent(id), let m = a.imageModel, !m.isEmpty,
           let h = cat.resolve(m), cat.capabilities(of: h).contains("image") { return h.id }
        let hs = cat.hosts(capability: "image").filter { cat.status(host: $0.id).usable }
        return (hs.first { $0.spec.engine == "z-image" } ?? hs.first)?.id
    }

    /// Start, stop or move the `parlor` server to match settings.
    private func syncServer() async {
        let want = settings.gateway
        if !want.enabled && !gatewayForHarness { endpoint.stop(); serverError = nil; return }
        if endpoint.isRunning, want.port == 0 || endpoint.port == want.port { return }
        do { try await endpoint.start(port: want.port); serverError = nil }
        catch { endpoint.stop(); serverError = error.localizedDescription }
    }

    // MARK: modules

    /// The authoring guide and the Python helper, rewritten when this app version has newer ones.
    private func seedModuleFiles() {
        let fm = FileManager.default
        try? fm.createDirectory(at: home.modulesDir, withIntermediateDirectories: true)
        for (name, text) in [("CLAUDE.md", Seeds.moduleGuide), ("parlor_module.py", Seeds.moduleHelper)] {
            let url = home.modulesDir.appendingPathComponent(name)
            let current = try? String(contentsOf: url, encoding: .utf8)
            if current == nil || !(current!.contains(Seeds.moduleMarker)) { try? text.write(to: url, atomically: true, encoding: .utf8) }
        }
    }

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
        freshStartIfNeeded()
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

    private func seedIfEmpty() {
        let fm = FileManager.default
        let guide = home.root.appendingPathComponent("CLAUDE.md")
        if !fm.fileExists(atPath: guide.path) { try? Seeds.folderGuide.write(to: guide, atomically: true, encoding: .utf8) }
        if !fm.fileExists(atPath: home.mcpConfig.path) { try? MCPConfigFile().save(home.mcpConfig) }
        // Starter agents are created once. An existing home with agents is adopted without seeding, and
        // deleting every agent later does not bring them back.
        guard !fm.fileExists(atPath: home.seededMarker.path) else { return }
        if agentStore.ids().isEmpty {
            for var a in [Seeds.mabel(), Seeds.pip(), Seeds.builder()] {
                a.model = availableModel(a.model)
                if a.imageModel != nil { a.imageModel = availableImageHost(a.imageModel!) }
                write(a)
            }
        }
        try? "seeded\n".write(to: home.seededMarker, atomically: true, encoding: .utf8)
    }

    /// V3 starts clean: the V2 state moves aside (never deleted) and a marker says so. A home that already has the marker,
    /// or that has nothing yet, is left alone.
    func freshStartIfNeeded() {
        let fm = FileManager.default
        guard !fm.fileExists(atPath: home.v3Marker.path) else { return }
        let v2 = ["agents", "chats", "memory", "workspaces", "harnesses", ".git", "settings.json"]
        // V2 agents were `agents/<id>.md` files; that is the signal. Anything else is already V3-shaped.
        let hasV2 = ((try? fm.contentsOfDirectory(atPath: home.agentsDir.path)) ?? []).contains { $0.hasSuffix(".md") }
        if hasV2 {
            let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd-HHmmss"
            let dest = home.root.appendingPathComponent("old-v2-" + f.string(from: Date()))
            try? fm.createDirectory(at: dest, withIntermediateDirectories: true)
            for name in v2 where fm.fileExists(atPath: home.root.appendingPathComponent(name).path) {
                try? fm.moveItem(at: home.root.appendingPathComponent(name), to: dest.appendingPathComponent(name))
            }
            // The seeding marker described the old agents: starter agents are made again.
            try? fm.removeItem(at: home.seededMarker)
        }
        try? home.ensure()
        try? "v3\n".write(to: home.v3Marker, atomically: true, encoding: .utf8)
    }

    // MARK: agents

    public func reloadAgents() {
        let fm = FileManager.default
        var out: [Agent] = []
        for id in agentStore.ids() {
            guard let r = agentStore.read(id) else { continue }
            let persona = (try? String(contentsOf: home.agentsDir.appendingPathComponent(id).appendingPathComponent("persona.md"), encoding: .utf8)) ?? ""
            out.append(r.agent(persona: persona.trimmingCharacters(in: .whitespacesAndNewlines)))
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
        let dir = home.agentsDir.appendingPathComponent(a.id)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // The record replaces every key it knows (so a removed setting is removed) and keeps keys it does not.
        var obj = agentStore.raw(a.id) ?? [:]
        for k in AgentRecord.knownKeys { obj[k] = nil }
        if let data = try? JSONEncoder().encode(AgentRecord(a)), let enc = try? JSONDecoder().decode(JSON.self, from: data).objectValue {
            for (k, v) in enc { obj[k] = v }
        }
        let persona = a.persona.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        let personaURL = dir.appendingPathComponent("persona.md")
        let changed = agentStore.raw(a.id) != obj || (try? String(contentsOf: personaURL, encoding: .utf8)) != persona
        if changed {
            try? agentStore.write(a.id, obj)
            try? persona.write(to: personaURL, atomically: true, encoding: .utf8)
        }
    }

    public func save(_ a: Agent) {
        write(a)
        reloadAgents()
    }

    @discardableResult
    public func create(kind: Agent.Kind, name: String? = nil) -> Agent {
        let h = kind == .assistant ? "chat" : "character"
        let t = harnessList.first { $0.id == h }?.templates.first { $0.id == (kind == .assistant ? "assistant" : "character") }
        return create(harness: h, template: t, name: name)
    }

    /// A new agent on a harness, from one of its templates, with a free id. Hosts fall back to ones that exist.
    @discardableResult
    public func create(harness hid: String, template t: HarnessManifest.Template?, name: String? = nil) -> Agent {
        let manifest = harnessList.first { $0.id == hid }
        let base = name ?? "New " + (t?.title ?? manifest?.title ?? hid).lowercased()
        var id = Agent.slug(base)
        var n = 2
        while agents.contains(where: { $0.id == id }) { id = Agent.slug(base) + "-\(n)"; n += 1 }
        let character = hid == "character"
        var a = Agent(id: id, name: base, persona: (t?.persona ?? "You are {name}.").replacingOccurrences(of: "{name}", with: base),
                      kind: character ? .character : .assistant,
                      model: availableModel(character ? Defaults.characterModel : Defaults.languageModel),
                      imageModel: (t?.image ?? false) ? availableImageHost(Defaults.imageModel) : nil,
                      tools: t?.tools ?? [], color: agents.count % 6, harness: hid == "chat" ? nil : hid, config: t?.config ?? [:])
        if let g = t?.group { a.group = g }
        save(a)
        selection = a.id
        return a
    }

    /// A host id for `preferred` if it exists, else the first installed MLX chat host, else `preferred` unchanged.
    func availableModel(_ preferred: String) -> String {
        let cat = catalog
        if let h = cat.resolve(preferred) { return h.id }
        guard mlx != nil else { return preferred }
        return cat.hosts(capability: "chat").first { $0.spec.engine == "mlx" }?.id ?? preferred
    }

    /// The same for the image host.
    func availableImageHost(_ preferred: String) -> String {
        let cat = catalog
        if let h = cat.resolve(preferred) { return h.id }
        let hs = cat.hosts(capability: "image")
        return (hs.first { $0.spec.engine == "z-image" } ?? hs.first)?.id ?? preferred
    }

    /// The shell line and resolved config for an agent's harness: schema defaults under the agent's own config.
    func launch(for a: Agent) -> (command: String, config: [String: JSON])? {
        if a.hasInlineHarness { return (a.harnessID, a.config) }
        guard let m = harnessList.first(where: { $0.id == a.harnessID }) else { return nil }
        var cfg = m.configDefaults
        for (k, v) in a.config { cfg[k] = v }
        return (m.launchCommand, cfg)
    }

    /// Why an agent cannot run right now, if so: its harness is missing, or a needed host is.
    public func harnessProblem(for a: Agent) -> String? {
        if !a.hasInlineHarness {
            guard let m = harnessList.first(where: { $0.id == a.harnessID }) else {
                return "Harness \"\(a.harnessID)\" is not in harnesses/. Available: " + harnessList.map(\.id).joined(separator: ", ") + "."
            }
            for (need, level) in m.needs where level == "required" && (a.hosts[need] ?? "").isEmpty {
                return "\(a.name) needs a \(need) host."
            }
        }
        return nil
    }

    /// Why an agent's model cannot answer right now, if so. Decided by the catalog.
    public func modelProblem(for a: Agent) -> String? {
        guard mlx != nil else { return nil }
        let cat = catalog
        if let (p, _) = router.route(a.model), cat.host(p.config.id) == nil { return p.config.enabled ? nil : "Provider \(p.config.id) is off." }
        guard let h = cat.resolve(a.model) else {
            let repo = a.model.hasPrefix("mlx/") ? String(a.model.dropFirst(4)) : a.model
            return "\(repo.split(separator: "/").last.map(String.init) ?? repo) is not downloaded."
        }
        return cat.problem(host: h.id)
    }

    public func delete(_ id: String) {
        externalHarness.stop(agent: id)
        try? FileManager.default.removeItem(at: home.agentsDir.appendingPathComponent(id))
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
        externalHarness.stop(agent: id)
        // The folder moves whole, so memory.md and anything else in it goes with the agent.
        try? fm.moveItem(at: home.agentsDir.appendingPathComponent(id), to: home.agentsDir.appendingPathComponent(newID))
        write(moved)
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
        queueDepth = await queues.depth
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

    /// The dashboard and the `services.*` verbs are projections of what each owner registers with the Supervisor.
    /// Nothing here switches on an id prefix.
    private func registerServicesIfNeeded() async {
        guard !servicesRegistered else { return }
        servicesRegistered = true
        await supervisor.register(source: "llm") { [weak self] in await self?.llmEntries() ?? [] }
        await supervisor.register(source: "image") { [weak self] in await self?.imageEntries() ?? [] }
        await supervisor.register(source: "mcp") { [weak self] in await self?.mcpEntries() ?? [] }
        await supervisor.register(source: "servers") { [weak self] in await self?.serverEntries() ?? [] }
        await supervisor.register(source: "providers") { [weak self] in await self?.providerEntries() ?? [] }
        await supervisor.register(source: "harnesses") { [weak self] in await self?.harnessEntries() ?? [] }
    }

    /// Rebuild the dashboard rows. Called by the System screen on a timer and after every control action.
    public func refreshServices() async {
        await registerServicesIfNeeded()
        services = await supervisor.snapshot()
    }

    private func llmEntries() async -> [ServiceEntry] {
        var loaded: [String] = []
        if let mlx { loaded = await mlx.loadedModels() } else if let m = await deps.llm.loadedModel() { loaded = [m] }
        let bytes = await mlx?.residentBytes()
        let depth = await queues.depth
        let info = ServiceInfo(
            id: "llm", kind: "language model", title: "Language models",
            detail: loaded.isEmpty ? "loads on the first message" : loaded.map { $0.split(separator: "/").last.map(String.init) ?? $0 }.joined(separator: ", "),
            state: loaded.isEmpty ? .stopped : (depth > 0 ? .running : .idle),
            memoryBytes: bytes.flatMap { $0 > 0 ? UInt64($0) : nil }, lastUsed: nil, queueDepth: depth, lifecycle: "onDemand",
            restarts: 0, canStart: false, canStop: !loaded.isEmpty, canRestart: false, logURL: nil)
        return [ServiceEntry(info: info, stop: { [weak self] in await self?.unloadModels() })]
    }

    private func imageEntries() async -> [ServiceEntry] {
        guard let image else { return [] }
        let s = await image.serviceSnapshot()
        let info = ServiceInfo(
            id: "image", kind: "image", title: "Image worker",
            detail: s.model.map { "loaded · " + ($0.split(separator: "/").last.map(String.init) ?? $0) } ?? "starts when an agent draws",
            state: s.state, memoryBytes: s.pid.flatMap { ProcessMemory.footprint(pid: $0) }, lastUsed: s.lastUsed, queueDepth: nil,
            lifecycle: "onDemand", restarts: 0, canStart: !s.state.isAlive, canStop: s.state.isAlive, canRestart: false,
            logURL: ServiceLog.url(home: home, service: "image"))
        return [ServiceEntry(info: info,
                             start: { try await image.start() },
                             stop: { await image.stop() },
                             restart: { await image.stop(); try await image.start() })]
    }

    private func mcpEntries() async -> [ServiceEntry] {
        var out: [ServiceEntry] = []
        let now = Date()
        let mcp = self.mcp!
        for m in await mcp.snapshot() {
            var state = m.state
            if state == .running, let t = m.lastUsed, now.timeIntervalSince(t) > 30 { state = .idle }
            let cfg = m.config
            let pid = await mcp.pid(m.id)
            let sid = (m.isModule ? "module:" : "mcp:") + m.id
            let name = m.id
            let info = ServiceInfo(
                id: sid, kind: m.isModule ? "module" : "mcp", title: m.id,
                detail: cfg.url ?? ([cfg.command ?? ""] + (cfg.args ?? [])).joined(separator: " "),
                state: state, memoryBytes: pid.flatMap { ProcessMemory.footprint(pid: $0) }, lastUsed: m.lastUsed, queueDepth: nil,
                lifecycle: cfg.isOnDemand ? "onDemand" : "always", restarts: m.restarts,
                canStart: !state.isAlive && cfg.disabled != true, canStop: state.isAlive, canRestart: cfg.disabled != true,
                logURL: ServiceLog.url(home: home, service: sid))
            out.append(ServiceEntry(info: info,
                                    start: { await mcp.start(name) }, stop: { await mcp.stop(name) }, restart: { await mcp.restart(name) }))
        }
        return out
    }

    private func serverEntries() -> [ServiceEntry] {
        let ps = settings.gateway
        let psState: ServiceState = !ps.enabled && !endpoint.isRunning ? .stopped : (serverError.map { .failed($0) } ?? (endpoint.isRunning ? .running : .starting))
        let server = ServiceEntry(
            info: ServiceInfo(
                id: "gateway", kind: "server", title: "Gateway",
                detail: endpoint.isRunning ? "http://127.0.0.1:\(endpoint.port) · /v1 (OpenAI) and /mcp · \(settings.tokens.count) token\(settings.tokens.count == 1 ? "" : "s")" : (ps.enabled ? "starting" : "off"),
                state: psState, memoryBytes: nil, lastUsed: nil, queueDepth: nil, lifecycle: "always", restarts: 0,
                canStart: !endpoint.isRunning, canStop: endpoint.isRunning, canRestart: endpoint.isRunning, logURL: nil),
            start: { [weak self] in guard let self else { return }; self.updateSettings { $0.gateway.enabled = true }; await self.syncServer() },
            stop: { [weak self] in guard let self else { return }; self.updateSettings { $0.gateway.enabled = false }; self.gatewayForHarness = false; await self.syncServer() },
            restart: { [weak self] in guard let self else { return }; self.endpoint.stop(); await self.syncServer() })
        return [server]
    }

    /// Providers that run a server process (llama.cpp and friends).
    private func providerEntries() -> [ServiceEntry] {
        router.all.compactMap { p -> ServiceEntry? in
            guard let proc = p.process else { return nil }
            let s = proc.snapshot()
            let id = p.config.id
            let enabled = p.config.enabled
            let info = ServiceInfo(
                id: "provider:" + id, kind: "provider", title: id,
                detail: s.external ? "already running · \(p.config.baseURL)" : (p.config.service?.command ?? ""),
                state: s.state, memoryBytes: s.pid.flatMap { ProcessMemory.footprint(pid: $0) }, lastUsed: s.lastUsed, queueDepth: nil,
                lifecycle: "onDemand", restarts: s.restarts, canStart: !s.state.isAlive && enabled, canStop: s.state.isAlive, canRestart: enabled,
                logURL: ServiceLog.url(home: home, service: "provider:" + id))
            let start: @MainActor @Sendable () async throws -> Void = {
                guard enabled else { throw CommandError("Provider \(id) is off.") }
                try await proc.ensureRunning()
            }
            return ServiceEntry(info: info, start: start, stop: { proc.stop() }, restart: { proc.stop(); try await start() })
        }
    }

    /// External harness processes. They start with the agent's next message, so the only verb is stop.
    private func harnessEntries() -> [ServiceEntry] {
        let ext = externalHarness
        return ext.running().map { id in
            let info = ServiceInfo(
                id: "harness:" + id, kind: "harness", title: agent(id)?.name ?? id,
                detail: agent(id)?.harness ?? "", state: .running,
                memoryBytes: ext.pid(agent: id).flatMap { ProcessMemory.footprint(pid: $0) }, lastUsed: nil, queueDepth: nil,
                lifecycle: "onDemand", restarts: 0, canStart: false, canStop: true, canRestart: true,
                logURL: ServiceLog.url(home: home, service: "harness:" + id))
            return ServiceEntry(info: info, stop: { ext.stop(agent: id) }, restart: { ext.stop(agent: id) })
        }
    }

    public func controlService(_ id: String, _ action: ServiceAction) async throws {
        await registerServicesIfNeeded()
        defer { Task { await self.refreshServices(); await self.refreshLoaded() } }
        try await supervisor.control(id, action)
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

/// Lets the harness host ask the engine for an environment without the engine existing yet.
final class EnvBox: @unchecked Sendable {
    var provider: (@Sendable (Agent) async -> [String: String])?
}

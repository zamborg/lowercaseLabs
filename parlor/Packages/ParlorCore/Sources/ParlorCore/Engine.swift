import Foundation
import Observation

/// Everything the UI and CLI talk to. Owns the store, tool hub, model runtime and rooms.
@Observable @MainActor
public final class ParlorEngine {
    public let home: ParlorHome
    public let library: ModelLibrary
    public let hub = ToolHub()
    public let supervisor = Supervisor()
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
    private let deps: RoomDeps
    private var mcp: MCPManager!
    private var watcher: FolderWatcher?
    @ObservationIgnored lazy var triggers = TriggerScheduler(engine: self)
    private var reloadTask: Task<Void, Never>?
    private let mlx: MLXRuntime?
    /// Sends `provider/model` names to their provider and everything else to MLX.
    public let router: ProviderRouter
    /// The OpenAI-compatible endpoint (SPEC §4) on 127.0.0.1.
    public private(set) var runtimeAPI: RuntimeAPIEndpoint!
    public private(set) var runtimeError: String?
    private var runtimeForHarness = false
    /// Runs agents that have a `harness:` command.
    public let externalHarness: ExternalHarness
    private var harnessTokens: [String: String] = [:]
    /// Git history of the hand-editable state (`~/parlor/.git`).
    public let history: HomeHistory
    /// Who the next write is credited to in history. The command gate sets it for the length of a verb.
    var actor = "ui"
    public var memory: MemoryStore { MemoryStore(home: home) }
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
        self.history = HomeHistory(root: home.root)
        let lib = ModelLibrary(home: home)
        self.library = lib
        let base: LanguageModelRuntime
        if let runtime { self.mlx = nil; base = runtime } else { let m = MLXRuntime(library: lib); self.mlx = m; base = m }
        let router = ProviderRouter(inner: base)
        self.router = router
        let envBox = EnvBox()
        let external = ExternalHarness(home: home) { agent in await envBox.provider?(agent) ?? [:] }
        self.externalHarness = external
        self.deps = RoomDeps(home: home, hub: hub, llm: QueuedRuntime(inner: router, queue: queue),
                             harness: HarnessRouter(builtIn: harness, external: external), library: lib)
        envBox.provider = { [weak self] agent in await self?.harnessEnvironment(agent) ?? [:] }
        self.mcp = MCPManager(hub: hub, home: home)
        deps.agentLookup = { [weak self] id in self?.agent(id) }
        self.commands = CommandCenter(engine: self)
        self.endpoint = ParlorMCPEndpoint(engine: self)
        self.runtimeAPI = RuntimeAPIEndpoint(engine: self)
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
        noteChange("Changed settings")
        Task { await self.applySettings() }
    }

    public func resetSettings() { updateSettings { $0 = ParlorSettings() } }

    private func applySettings() async {
        deps.settings = settings
        await mlx?.setKeepAlive(minutes: settings.llm.keepLoadedMinutes)
        await mlx?.configure(budgetFraction: settings.memoryBudgetFraction, slots: settings.cacheSlotsPerModel)
        await image?.configure(settings.image)
        router.configure(settings.providers, home: home)
        await syncServer()
        await syncRuntimeAPI()
    }

    /// Start, stop or move the runtime API to match settings. A launched harness keeps it up.
    private func syncRuntimeAPI() async {
        let want = settings.runtimeAPI
        if !want.enabled && !runtimeForHarness { runtimeAPI.stop(); runtimeError = nil; return }
        if runtimeAPI.isRunning, want.port == 0 || runtimeAPI.port == want.port { return }
        do { try await runtimeAPI.start(port: want.port); runtimeError = nil }
        catch { runtimeAPI.stop(); runtimeError = error.localizedDescription }
    }

    /// Bring the runtime API up for a harness, on the configured port or any free one.
    func ensureRuntimeAPI() async {
        if runtimeAPI.isRunning { return }
        runtimeForHarness = true
        if (try? await runtimeAPI.start(port: settings.runtimeAPI.port)) == nil {
            do { try await runtimeAPI.start(port: 0); runtimeError = nil } catch { runtimeError = error.localizedDescription }
        } else { runtimeError = nil }
    }

    /// What a launched harness gets in its environment (SPEC §2.5). One secret serves both servers, at the agent's tier.
    func harnessEnvironment(_ agent: Agent) async -> [String: String] {
        await ensureRuntimeAPI()
        let key = agent.id + ":" + agent.tier.rawValue
        if harnessTokens[key] == nil { harnessTokens[key] = endpoint.issueEphemeral(name: "harness:" + agent.id, tier: agent.tier) }
        let token = harnessTokens[key]!
        let parlorPort = endpoint.isRunning ? endpoint.port : settings.parlorServer.port
        return [
            "OPENAI_BASE_URL": "http://127.0.0.1:\(runtimeAPI.port)/v1", "OPENAI_API_KEY": token,
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

    // MARK: llama.cpp (GGUF) models

    /// Provider id and model name an installed GGUF model is served as: `llama-<slug>/<slug>`.
    public static func ggufNames(_ m: InstalledModel) -> (provider: String, model: String) {
        let slug = Agent.slug(m.shortName)
        return ("llama-" + slug, slug)
    }

    /// Give a GGUF model a provider that runs `llama-server` on it when first used. Keeps an entry the user has edited.
    func registerGGUF(_ m: InstalledModel) {
        guard m.format == .gguf, let file = m.file else { return }
        let (id, name) = Self.ggufNames(m)
        guard !settings.providers.contains(where: { $0.id == id }) else { return }
        var used = Set(settings.providers.compactMap { URL(string: $0.baseURL)?.port })
        used.insert(11500)
        var port = 8100
        while used.contains(port) { port += 1 }
        let path = m.directory.appendingPathComponent(file).path.replacingOccurrences(of: "'", with: "'\\''")
        let cmd = "llama-server -m '\(path)' --alias \(name) --port \(port) --jinja -c 32768 -ngl 99"
        let entry = ProviderConfig(id: id, baseURL: "http://127.0.0.1:\(port)/v1", models: [name],
                                   service: .init(command: cmd, idleMinutes: 30, startupSeconds: 600))
        updateSettings { $0.providers.append(entry) }
    }

    func unregisterGGUF(_ m: InstalledModel) {
        let id = Self.ggufNames(m).provider
        updateSettings { $0.providers.removeAll { $0.id == id } }
    }

    /// Make sure every downloaded GGUF model has its provider (covers files placed by hand and settings edits).
    func syncGGUFProviders() { for m in library.installed where m.format == .gguf { registerGGUF(m) } }

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
        return library.installedDirectory(for: repo) != nil
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

    /// The harness guide and the sample harness, rewritten when this app version has newer ones.
    private func seedHarnessFiles() {
        let fm = FileManager.default
        try? fm.createDirectory(at: home.harnessesDir, withIntermediateDirectories: true)
        for (name, text) in [("CLAUDE.md", Seeds.harnessGuide), ("openai_loop.py", Seeds.harnessSample), ("character.py", Seeds.harnessCharacter)] {
            let url = home.harnessesDir.appendingPathComponent(name)
            let current = try? String(contentsOf: url, encoding: .utf8)
            // Missing, or one of ours (marked) that differs: rewrite. A file without the marker is the user's and stays.
            if current == nil || (current!.contains(Seeds.harnessMarker) && current != text) { try? text.write(to: url, atomically: true, encoding: .utf8) }
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
        seedIfEmpty()
        history.setup()
        reloadAgents()
        if selection == nil { selection = agents.first?.id }
        let img = ImageProvider(home: home, library: library)
        image = img
        deps.image = img
        settings = ParlorSettings.load(home.settingsFile)
        if !FileManager.default.fileExists(atPath: home.settingsFile.path) { settings.save(home.settingsFile) }
        library.onGGUFInstalled = { [weak self] m in Task { @MainActor in self?.registerGGUF(m) } }
        library.onGGUFRemoved = { [weak self] m in Task { @MainActor in self?.unregisterGGUF(m) } }
        syncGGUFProviders()
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
        seedHarnessFiles()
        reloadModules()
        await mcp.setModuleProvider { [weak self] in await self?.moduleConfigs() ?? [:] }
        // Servers start in the background: a slow or broken one must not hold up the window.
        Task { await self.mcp.reload(); await self.syncFromHub() }
        await syncFromHub()
        watcher = FolderWatcher(path: home.root) { [weak self] paths in
            Task { @MainActor in self?.filesChanged(paths) }
        }
        triggers.start()
    }

    public func shutdown() async {
        watcher?.stop()
        triggers.stop()
        history.flush()
        endpoint.stop()
        runtimeAPI.stop()
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
        let existing = (try? fm.contentsOfDirectory(atPath: home.agentsDir.path))?.filter { $0.hasSuffix(".md") } ?? []
        if existing.isEmpty {
            for var a in [Seeds.mabel(), Seeds.pip()] { a.model = availableModel(a.model); write(a) }
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
        noteChange("Created room \(id)")
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
            history.willChange(actor: actor)
            try? text.write(to: url, atomically: true, encoding: .utf8)
            noteChange("Edited agent \(a.id)")
        }
    }

    /// Ask history to record the change, credited to whoever is acting.
    public func noteChange(_ what: String) { history.record(actor: actor, what) }

    public func save(_ a: Agent) {
        write(a)
        reloadAgents()
    }

    @discardableResult
    public func create(kind: Agent.Kind, name: String? = nil) -> Agent {
        create(template: AgentTemplates.template(kind == .assistant ? "assistant" : "character")!, name: name)
    }

    /// A new agent from a template, with a free id. Starter models fall back to one that is installed.
    @discardableResult
    public func create(template t: AgentTemplate, name: String? = nil) -> Agent {
        let base = name ?? "New " + t.title.lowercased()
        var id = Agent.slug(base)
        var n = 2
        while agents.contains(where: { $0.id == id }) { id = Agent.slug(base) + "-\(n)"; n += 1 }
        var a = Agent(
            id: id, name: base, summary: "", persona: t.persona(base), kind: t.kind, model: availableModel(t.languageModel),
            imageModel: t.withImage ? Defaults.imageModel : nil, tools: t.tools, contextTokens: t.contextTokens,
            temperature: t.temperature, thinking: t.thinking, color: agents.count % 6)
        if let g = t.group { a.group = g }
        if let file = t.harness ?? (t.scripted ? "openai_loop.py" : nil) { a.harness = "python3 '\(home.harnessesDir.appendingPathComponent(file).path)'" }
        for (k, v) in t.extra.sorted(by: { $0.key < $1.key }) { a.extra.append(ExtraField(key: k, value: .scalar(v))) }
        save(a)
        selection = a.id
        return a
    }

    /// `preferred` if it is installed (or nothing can be checked), else the first installed language model, else `preferred`.
    func availableModel(_ preferred: String) -> String {
        guard mlx != nil, library.installedDirectory(for: preferred) == nil else { return preferred }
        return library.installed(kind: .language).map(\.repo).sorted().first ?? preferred
    }

    /// Why an agent's model cannot answer right now, if so.
    public func modelProblem(for a: Agent) -> String? {
        if let (p, _) = router.route(a.model) { return p.config.enabled ? nil : "Provider \(p.config.id) is off." }
        guard mlx != nil else { return nil }
        let repo = a.model.hasPrefix("mlx/") ? String(a.model.dropFirst(4)) : a.model
        return library.installedDirectory(for: repo) == nil ? "\(repo.split(separator: "/").last.map(String.init) ?? repo) is not downloaded." : nil
    }

    public func delete(_ id: String) {
        try? FileManager.default.removeItem(at: home.agentsDir.appendingPathComponent("\(id).md"))
        try? FileManager.default.removeItem(at: home.chatDir(for: id))
        rooms[id] = nil
        noteChange("Deleted agent \(id)")
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
        noteChange("Changed mounted servers")
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
        let depth = await queue.depth
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
        let ps = settings.parlorServer
        let psState: ServiceState = !ps.enabled ? .stopped : (serverError.map { .failed($0) } ?? (endpoint.isRunning ? .running : .starting))
        let server = ServiceEntry(
            info: ServiceInfo(
                id: "parlorServer", kind: "server", title: "Parlor server",
                detail: ps.enabled ? "http://127.0.0.1:\(ps.port)/mcp · \(settings.tokens.count) token\(settings.tokens.count == 1 ? "" : "s")" : "off",
                state: psState, memoryBytes: nil, lastUsed: nil, queueDepth: nil, lifecycle: "always", restarts: 0,
                canStart: !ps.enabled, canStop: ps.enabled, canRestart: ps.enabled, logURL: nil),
            start: { [weak self] in guard let self else { return }; self.updateSettings { $0.parlorServer.enabled = true }; await self.syncServer() },
            stop: { [weak self] in guard let self else { return }; self.updateSettings { $0.parlorServer.enabled = false }; await self.syncServer() },
            restart: { [weak self] in guard let self else { return }; self.endpoint.stop(); await self.syncServer() })
        let ra = settings.runtimeAPI
        let raState: ServiceState = runtimeAPI.isRunning ? .running : (runtimeError.map { .failed($0) } ?? .stopped)
        let api = ServiceEntry(
            info: ServiceInfo(
                id: "runtimeAPI", kind: "server", title: "Runtime API",
                detail: runtimeAPI.isRunning ? "http://127.0.0.1:\(runtimeAPI.port)/v1 · OpenAI-compatible" : (ra.enabled ? "starting" : "off · starts with an external harness"),
                state: raState, memoryBytes: nil, lastUsed: nil, queueDepth: nil, lifecycle: "always", restarts: 0,
                canStart: !runtimeAPI.isRunning, canStop: runtimeAPI.isRunning, canRestart: runtimeAPI.isRunning, logURL: nil),
            start: { [weak self] in
                guard let self else { return }
                self.updateSettings { $0.runtimeAPI.enabled = true }; self.runtimeAPI.stop(); await self.syncRuntimeAPI()
            },
            stop: { [weak self] in
                guard let self else { return }
                self.updateSettings { $0.runtimeAPI.enabled = false }; self.runtimeForHarness = false; self.runtimeAPI.stop()
            },
            restart: { [weak self] in
                guard let self else { return }
                self.runtimeAPI.stop(); await self.syncRuntimeAPI(); if !self.runtimeAPI.isRunning { await self.ensureRuntimeAPI() }
            })
        return [server, api]
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
        triggers.filesChanged(paths)
        // FSEvents reports resolved paths (/private/var/…), so compare against resolved prefixes too.
        let prefixes = [home.agentsDir.path, Self.realPath(home.agentsDir.path)]
        let relevant = paths.contains { p in prefixes.contains { p.hasPrefix($0) } || p.hasSuffix("/mcp.json") }
        _ = relevant
        let chats = paths.filter { $0.contains("/chats/") }
        let tracked = paths.contains { p in
            !p.contains("/.git/") && (p.contains("/agents/") || p.contains("/memory/") || p.hasSuffix("/settings.json") || p.hasSuffix("/mcp.json")
                                      || p.hasSuffix("/module.json") || p.hasSuffix("/conversation.json"))
        }
        if tracked { history.record(actor: "outside edit", "Edited outside the app") }
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

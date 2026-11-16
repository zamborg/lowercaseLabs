import Foundation

extension ParlorEngine {
    // MARK: services

    /// The dashboard and the `services.*` verbs are projections of what each owner registers with the Supervisor.
    /// Nothing here switches on an id prefix.
    func registerServicesIfNeeded() async {
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

    func llmEntries() async -> [ServiceEntry] {
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

    func imageEntries() async -> [ServiceEntry] {
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

    func mcpEntries() async -> [ServiceEntry] {
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

    func serverEntries() -> [ServiceEntry] {
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
    func providerEntries() -> [ServiceEntry] {
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
    func harnessEntries() -> [ServiceEntry] {
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

    func syncFromHub() async {
        servers = await hub.allStatuses()
        tools = await hub.allTools()
    }
}

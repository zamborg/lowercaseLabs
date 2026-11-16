import Foundation

extension ParlorEngine {
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
    func derivedProviders(_ cat: Catalog) -> [ProviderConfig] {
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
    func syncServer() async {
        let want = settings.gateway
        if !want.enabled && !gatewayForHarness { endpoint.stop(); serverError = nil; return }
        if endpoint.isRunning, want.port == 0 || endpoint.port == want.port { return }
        do { try await endpoint.start(port: want.port); serverError = nil }
        catch { endpoint.stop(); serverError = error.localizedDescription }
    }
}

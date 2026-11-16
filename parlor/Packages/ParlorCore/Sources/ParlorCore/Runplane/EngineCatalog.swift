import Foundation

extension ParlorEngine {
    // MARK: engines, hosts and the catalog

    /// Everything that decides availability, as one value. Rebuilt from what is loaded; nothing else answers "is it there?".
    public var catalog: Catalog {
        var states: [String: ServiceState] = [:]
        for h in hostList {
            if let e = engineList.first(where: { $0.id == h.spec.engine }), let s = adapters[e.adapter]?.state(of: h) { states[h.id] = s }
        }
        var dirs: [String: URL] = [:]
        for e in engineList { dirs[e.id] = e.directory }
        return Catalog(models: library.installed, engines: engineList, hosts: hostList, states: states, engineDirs: dirs)
    }

    /// Reload engines and hosts from disk, recompute the auto hosts, and tell every adapter, so a new host is live at once.
    public func reloadRunplane() {
        var engines = engineStore.list()
        for i in engines.indices { engines[i].directory = home.enginesDir.appendingPathComponent(engines[i].id) }
        engineList = engines
        var harnesses = harnessStore.list()
        for i in harnesses.indices { harnesses[i].directory = home.harnessesDir.appendingPathComponent(harnesses[i].id) }
        harnessList = harnesses
        hostList = HostRegistry.build(explicit: hostStore.list(), installed: library.installed, engines: engines)
        let cat = catalog
        for a in adapters.all { a.reconfigure(catalog: cat) }
    }

    /// The state of the host an agent chats on (nil when it names none that exists).
    public func chatHostStatus(for a: Agent) -> HostStatus? {
        let cat = catalog
        return cat.resolve(a.chatHost).map { cat.status(host: $0.id) }
    }

    /// Agents that use a host serving this model, for "used by" in the Library.
    public func agents(usingModel repo: String) -> [Agent] {
        let cat = catalog
        let hostIDs = Set(cat.hosts(usingModel: repo).map(\.id))
        return agents.filter { a in a.hosts.values.contains { ref in cat.resolve(ref).map { hostIDs.contains($0.id) } ?? false } }
    }

    /// The image host an agent draws with: its `image_model` through the catalog, else the first usable image host (Z-Image first).
    func defaultImageHost(for agentID: String?) -> String? {
        let cat = catalog
        if let id = agentID, let a = agent(id), let m = a.imageHost, !m.isEmpty,
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

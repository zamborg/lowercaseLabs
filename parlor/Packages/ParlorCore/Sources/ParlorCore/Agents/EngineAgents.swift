import Foundation

extension ParlorEngine {
    // MARK: agents

    public func reloadAgents() {
        let fm = FileManager.default
        var out: [Agent] = []
        for id in agentStore.ids() {
            guard let r = agentStore.read(id) else { continue }
            let persona = (try? String(contentsOf: home.agentsDir.appendingPathComponent(id).appendingPathComponent("persona.md"), encoding: .utf8)) ?? ""
            out.append(r.agent(persona: persona.trimmingCharacters(in: .whitespacesAndNewlines)))
        }
        out.sort { ($0.group == "Assistants" ? 0 : 1, $0.name.lowercased()) < ($1.group == "Assistants" ? 0 : 1, $1.name.lowercased()) }
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

    func write(_ a: Agent) {
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

    /// Write an agent from its JSON (the `agent.json` keys plus `persona`), keeping unknown keys, then reload.
    func writeAgent(_ draft: JSON) throws {
        var o = draft.objectValue ?? [:]
        let persona = o["persona"]?.stringValue ?? (try? String(contentsOf: home.agentsDir.appendingPathComponent(o["id"]?.stringValue ?? "").appendingPathComponent("persona.md"), encoding: .utf8)) ?? ""
        o["persona"] = nil; o["memory"] = nil
        guard let id = o["id"]?.stringValue else { throw CommandError("An agent needs an id.") }
        try? home.ensure()
        try agentStore.write(id, o)
        try (persona.trimmingCharacters(in: .whitespacesAndNewlines) + "\n").write(to: home.agentsDir.appendingPathComponent(id).appendingPathComponent("persona.md"), atomically: true, encoding: .utf8)
        reloadAgents()
    }

    public func save(_ a: Agent) {
        write(a)
        reloadAgents()
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
        var hosts = ["chat": availableModel(character ? Defaults.characterModel : Defaults.languageModel)]
        if t?.image ?? false { hosts["image"] = availableImageHost(Defaults.imageModel) }
        var a = Agent(id: id, name: base, persona: (t?.persona ?? "You are {name}.").replacingOccurrences(of: "{name}", with: base),
                      hosts: hosts, tools: t?.tools ?? [], color: agents.count % 6, harness: hid == "chat" ? nil : hid, config: t?.config ?? [:])
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
        guard let h = cat.resolve(a.chatHost) else {
            return a.chatHost.isEmpty ? "\(a.name) has no chat host." : "There is no host \"\(a.chatHost)\". Hosts: " + cat.hosts(capability: "chat").map(\.id).joined(separator: ", ") + "."
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
        for a in adapters.all { await a.refresh() }
        loadedModel = await mlxAdapter.runtime.loadedModel()
        queueDepth = await queues.depth
        if let image { imageWorkerState = await image.stateDescription() }
    }
}

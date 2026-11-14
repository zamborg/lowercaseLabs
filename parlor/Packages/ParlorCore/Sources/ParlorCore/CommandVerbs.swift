import Foundation

// The verbs. Each one is a thin, validated wrapper over an engine method the UI already uses.
extension CommandCenter {

    // MARK: agents

    private static let agentKeys: Set<String> = [
        "name", "description", "kind", "model", "image_model", "tools", "tier", "context", "compact_at", "temperature",
        "thinking", "color", "persona", "top_p", "repetition_penalty", "max_reply_tokens", "max_steps",
        "tool_result_chars", "tool_timeout", "keep_recent", "summary_tokens", "group", "harness", "template",
    ]

    private func apply(_ a: inout Agent, _ args: Args, caller: Caller) throws {
        try args.check(allowed: Self.agentKeys.union(["id"]))
        func num(_ k: String, _ r: ClosedRange<Double>) throws -> Double? {
            guard let v = args.raw[k] else { return nil }
            guard let d = v.doubleValue, r.contains(d) else { throw CommandError("\"\(k)\" must be a number from \(r.lowerBound) to \(r.upperBound).") }
            return d
        }
        func whole(_ k: String, _ r: ClosedRange<Int>) throws -> Int? {
            guard let v = args.raw[k] else { return nil }
            guard let i = v.intValue, r.contains(i) else { throw CommandError("\"\(k)\" must be a whole number from \(r.lowerBound) to \(r.upperBound).") }
            return i
        }
        if let v = args.optString("name") { guard !v.trimmingCharacters(in: .whitespaces).isEmpty else { throw CommandError("\"name\" is empty.") }; a.name = v }
        if let v = args.optString("description") { a.summary = v }
        if let v = args.optString("persona") { a.persona = v }
        if let v = args.optString("kind") {
            guard let k = Agent.Kind(rawValue: v) else { throw CommandError("\"kind\" is assistant or character.") }
            a.kind = k
        }
        if let v = args.optString("group") { a.group = v }
        if let v = args.optString("harness") { a.harness = v.trimmingCharacters(in: .whitespaces).isEmpty ? nil : v }
        if let v = args.optString("model") { a.model = v }
        if let v = args.optString("image_model") { a.imageModel = v.isEmpty ? nil : v }
        if let v = args.strings("tools") { a.tools = v }
        if let v = args.optString("tier") {
            guard let t = Tier(rawValue: v) else { throw CommandError("\"tier\" is one of none, read, act, configure.") }
            if caller.kind == .agent || caller.kind == .token, t > caller.tier { throw CommandError("A caller cannot grant a tier above its own (\(caller.tier.rawValue)).") }
            a.tier = t
        }
        if let v = try whole("context", 512...262_144) { a.contextTokens = v }
        if let v = try num("compact_at", 0.1...1) { a.compactAt = v }
        if let v = try num("temperature", 0...2) { a.temperature = v }
        if let v = args.raw["thinking"] { guard let b = v.boolValue else { throw CommandError("\"thinking\" is true or false.") }; a.thinking = b }
        if let v = try whole("color", 0...5) { a.color = v }
        if let v = try num("top_p", 0...1) { a.overrides.topP = v }
        if let v = try num("repetition_penalty", 0.5...2) { a.overrides.repetitionPenalty = v }
        if let v = try whole("max_reply_tokens", 0...131_072) { a.overrides.maxReplyTokens = v }
        if let v = try whole("max_steps", 1...64) { a.overrides.maxSteps = v }
        if let v = try whole("tool_result_chars", 100...200_000) { a.overrides.toolResultChars = v }
        if let v = try whole("tool_timeout", 1...86_400) { a.overrides.toolTimeoutSeconds = v }
        if let v = try whole("keep_recent", 0...200) { a.overrides.keepRecent = v }
        if let v = try whole("summary_tokens", 64...8192) { a.overrides.summaryTokens = v }
    }

    /// A harness is a command the app will run. An agent or token setting one needs the user's yes.
    private func confirmHarness(_ args: Args, _ caller: Caller) async throws {
        guard let h = args.optString("harness"), !h.trimmingCharacters(in: .whitespaces).isEmpty, caller.mustConfirmDestructive else { return }
        guard let ask = engine.confirm else { throw CommandError("Setting a harness runs a program on this Mac, so it needs the user. Nobody is available to confirm it.") }
        guard await ask("\(caller.name) wants an agent to run this program for its turns:\n\n\(h)") else { throw CommandError("The user declined the harness.") }
    }

    private func agent(_ id: String) throws -> Agent {
        guard let a = engine.agent(id) else { throw CommandError("No agent \"\(id)\".") }
        return a
    }

    func registerAgents() {
        let fields: [(String, String, String)] = [
            ("name", "string", "Display name"), ("description", "string", "One line shown in the sidebar"),
            ("persona", "string", "The system prompt"), ("kind", "string", "assistant or character"),
            ("model", "string", "Language model repo id"), ("image_model", "string", "Image model repo id"),
            ("tools", "strings", "Tool patterns: *, server.*, server.tool"), ("tier", "string", "none, read, act or configure"),
            ("context", "integer", "Context tokens"), ("compact_at", "number", "Compaction threshold, 0.1 to 1"),
            ("temperature", "number", "0 to 2"), ("thinking", "boolean", "Let the model think"), ("color", "integer", "Avatar color 0-5"),
            ("top_p", "number", "Override of the machine default"), ("repetition_penalty", "number", "Override"),
            ("max_reply_tokens", "integer", "Override; 0 for no cap"), ("max_steps", "integer", "Override"),
            ("tool_result_chars", "integer", "Override"), ("tool_timeout", "integer", "Override, seconds"),
            ("keep_recent", "integer", "Override"), ("summary_tokens", "integer", "Override"),
            ("group", "string", "Sidebar section"), ("harness", "string", "Command that runs this agent's turns instead of the built-in loop (asks the user first when set by an agent or token)"),
        ]
        add("agents.list", .read, "List agents (id, name, kind, model, tools, tier).") { _, _ in
            .array(self.engine.agents.map { self.agentJSON($0) })
        }
        add("agents.read", .read, "One agent in full, including its persona.", props: [("id", "string", "Agent id")], required: ["id"]) { a, _ in
            self.agentJSON(try self.agent(try a.string("id")), full: true)
        }
        add("agents.templates", .read, "Starting points for agents.create: id, title, what it sets up.") { _, _ in
            .array(AgentTemplates.all.map { .object(["id": .string($0.id), "title": .string($0.title), "description": .string($0.blurb), "tools": .strings($0.tools)]) })
        }
        add("agents.create", .configure, "Create an agent from a template (see agents.templates), then set any of the fields.", props: [("template", "string", "Template id; default assistant")] + fields) { a, c in
            try await self.confirmHarness(a, c)
            let name = a.optString("name") ?? "New assistant"
            let kind = Agent.Kind(rawValue: a.optString("kind") ?? "assistant") ?? .assistant
            var made: Agent
            if let tid = a.optString("template") {
                guard let t = AgentTemplates.template(tid) else { throw CommandError("No template \"\(tid)\". Use agents.templates.") }
                made = self.engine.create(template: t, name: name)
            } else { made = self.engine.create(kind: kind, name: name) }
            try self.apply(&made, a, caller: c)
            self.engine.save(made)
            let id = self.engine.renameToMatchName(made.id)
            return self.agentJSON(try self.agent(id), full: true)
        }
        add("agents.update", .configure, "Change fields of an agent. Unknown or out-of-range values are refused.",
            props: [("id", "string", "Agent id")] + fields, required: ["id"]) { a, c in
            try await self.confirmHarness(a, c)
            var cur = try self.agent(try a.string("id"))
            try self.apply(&cur, a, caller: c)
            self.engine.save(cur)
            let id = self.engine.renameToMatchName(cur.id)
            return self.agentJSON(try self.agent(id), full: true)
        }
        add("agents.duplicate", .configure, "Copy an agent (without its chat).", props: [("id", "string", "Agent id")], required: ["id"]) { a, _ in
            let id = try a.string("id"); _ = try self.agent(id)
            self.engine.duplicate(id)
            return .object(["id": .string(self.engine.selection ?? "")])
        }
        add("agents.delete", .configure, destructive: true, "Delete an agent and its chat.", props: [("id", "string", "Agent id")], required: ["id"]) { a, _ in
            let id = try a.string("id"); _ = try self.agent(id)
            self.engine.delete(id)
            return .object(["deleted": .string(id)])
        }
    }

    // MARK: conversations

    private func conversation(_ id: String) throws -> (room: Room, agent: Agent?) {
        if let a = engine.agent(id) { return (engine.room(for: id), a) }
        if let c = engine.conversations.first(where: { $0.id == id }) {
            return (engine.room(for: id), c.config.agentIDs.first.flatMap { engine.agent($0) })
        }
        throw CommandError("No conversation \"\(id)\". Use conversations.list.")
    }

    func registerConversations() {
        let idProp: [(String, String, String)] = [("id", "string", "Conversation id: an agent id for a one-to-one chat, or a room id")]
        add("conversations.list", .read, "Rooms and one-to-one chats.") { _, _ in
            var out: [JSON] = []
            for a in self.engine.agents { out.append(.object(["id": .string(a.id), "title": .string(a.name), "participants": .strings([a.id]), "scheduler": .string("userDriven"), "room": .bool(false)])) }
            for c in self.engine.conversations {
                out.append(.object(["id": .string(c.id), "title": .string(c.title), "participants": .strings(c.config.agentIDs), "scheduler": .string(c.config.scheduler.kind), "room": .bool(true)]))
            }
            return .array(out)
        }
        add("conversations.read", .read, "The last messages of a conversation.", props: idProp + [("limit", "integer", "How many, default 40")], required: ["id"]) { a, _ in
            let (room, _) = try self.conversation(try a.string("id"))
            let n = max(1, min(500, a.optInt("limit") ?? 40))
            let ms = room.messages.suffix(n).map { self.messageJSON($0) }
            return .object(["messages": .array(Array(ms)), "busy": .bool(room.isBusy), "summary": room.summaryText.map { .string($0) } ?? .null])
        }
        add("conversations.create", .act, "Create a room from existing agents.",
            props: [("title", "string", "Room title"), ("agents", "strings", "Agent ids"),
                    ("scheduler", "string", "userDriven, roundRobin, mention or autonomous"), ("max_turns", "integer", "Turn budget per run")],
            required: ["agents"]) { a, _ in
            let ids = a.strings("agents") ?? []
            guard ids.count >= 1 else { throw CommandError("\"agents\" needs at least one agent id.") }
            for i in ids { _ = try self.agent(i) }
            let kind = a.optString("scheduler") ?? "roundRobin"
            guard ["userDriven", "roundRobin", "mention", "autonomous"].contains(kind) else { throw CommandError("Unknown scheduler \"\(kind)\".") }
            let id = self.engine.createRoom(title: a.optString("title") ?? "Room", agents: ids, scheduler: SchedulerSpec(kind: kind, maxTurns: a.optInt("max_turns")))
            return .object(["id": .string(id)])
        }
        add("conversations.post", .act, "Say something in a conversation; the agents answer. With wait=true the call returns after they finish.",
            props: idProp + [("text", "string", "Message"), ("wait", "boolean", "Wait for the replies")], required: ["id", "text"]) { a, c in
            let id = try a.string("id"), text = try a.string("text")
            let (room, agent) = try self.conversation(id)
            guard !room.isBusy else { throw CommandError("That conversation is busy. Try again when it finishes, or conversations.stop.") }
            let before = room.messages.count
            if a.bool("wait") { await room.sendAndWait(text, agent: agent) } else { room.send(text, agent: agent) }
            let new = room.messages.dropFirst(before).map { self.messageJSON($0) }
            return .object(["accepted": .bool(true), "messages": .array(Array(new))])
        }
        add("conversations.run", .act, "Let the room's scheduler run without a new message.", props: idProp + [("wait", "boolean", "Wait for it to finish")], required: ["id"]) { a, _ in
            let (room, _) = try self.conversation(try a.string("id"))
            guard !room.isBusy else { throw CommandError("That conversation is busy.") }
            let before = room.messages.count
            if a.bool("wait") { await room.runAndWait() } else { room.run() }
            return .object(["messages": .array(Array(room.messages.dropFirst(before).map { self.messageJSON($0) }))])
        }
        add("conversations.stop", .act, "Stop the reply in progress.", props: idProp, required: ["id"]) { a, _ in
            try self.conversation(try a.string("id")).room.stop(); return .object(["stopped": .bool(true)])
        }
        add("conversations.compact", .act, "Fold older messages into a summary now.", props: idProp, required: ["id"]) { a, _ in
            let (room, agent) = try self.conversation(try a.string("id"))
            guard let agent else { throw CommandError("No agent to compact with.") }
            let did = try await room.compact(agent: agent)
            return .object(["compacted": .bool(did)])
        }
        add("conversations.prune", .act, destructive: true, "Delete messages by id, or cut everything after a message.",
            props: idProp + [("ids", "strings", "Message ids to delete"), ("after", "string", "Keep up to and including this message id")], required: ["id"]) { a, _ in
            let (room, _) = try self.conversation(try a.string("id"))
            guard !room.isBusy else { throw CommandError("That conversation is busy.") }
            if let after = a.optString("after") {
                guard let u = UUID(uuidString: after), room.messages.contains(where: { $0.id == u }) else { throw CommandError("No message \(after).") }
                room.cut(after: u)
            } else if let ids = a.strings("ids"), !ids.isEmpty {
                room.delete(ids: Set(ids.compactMap(UUID.init)))
            } else { throw CommandError("Pass \"ids\" or \"after\".") }
            return .object(["messages": .int(room.messages.count)])
        }
        add("conversations.reset", .act, destructive: true, "Clear a conversation completely.", props: idProp, required: ["id"]) { a, _ in
            let (room, _) = try self.conversation(try a.string("id"))
            guard !room.isBusy else { throw CommandError("That conversation is busy.") }
            room.reset(); return .object(["cleared": .bool(true)])
        }
    }

    // MARK: models and providers

    func registerModels() {
        add("models.list", .read, "Installed models and downloads in progress.") { _, _ in
            let lib = self.engine.library
            let installed = lib.installed.map { JSON.object(["repo": .string($0.repo), "kind": .string($0.kind.rawValue), "bytes": .int(Int($0.bytes))]) }
            let dl = lib.downloads.values.map { d -> JSON in
                var status = "running"
                switch d.status { case .paused: status = "paused"; case .done: status = "done"; case .failed(let e): status = "failed: " + e; case .running: break }
                return .object(["repo": .string(d.repo), "fraction": .double(d.fraction), "status": .string(status)])
            }
            return .object(["installed": .array(installed), "downloads": .array(dl), "loaded": .strings(await self.engine.loadedModels())])
        }
        add("models.search", .read, "Search Hugging Face for MLX models.", props: [("query", "string", "Search text"), ("kind", "string", "language or image")], required: ["query"]) { a, _ in
            let kind = ModelKind(rawValue: a.optString("kind") ?? "language") ?? .language
            let rs = try await self.engine.library.search(try a.string("query"), kind: kind)
            return .array(rs.map { .object(["repo": .string($0.repo), "downloads": .int($0.downloads), "bytes": $0.bytes.map { .int(Int($0)) } ?? .null, "unsupported": $0.unsupported.map { .string($0) } ?? .null]) })
        }
        add("models.download", .act, "Start (or resume) a download.", props: [("repo", "string", "Repo id"), ("kind", "string", "language or image")], required: ["repo"]) { a, _ in
            let kind = ModelKind(rawValue: a.optString("kind") ?? "language") ?? .language
            self.engine.library.download(try a.string("repo"), kind: kind)
            return .object(["started": .bool(true)])
        }
        add("models.pause", .act, "Pause a download.", props: [("repo", "string", "Repo id")], required: ["repo"]) { a, _ in
            self.engine.library.pause(try a.string("repo")); return .object(["paused": .bool(true)])
        }
        add("models.delete", .act, destructive: true, "Delete an installed model from disk.", props: [("repo", "string", "Repo id")], required: ["repo"]) { a, _ in
            let repo = try a.string("repo")
            guard self.engine.library.installedDirectory(for: repo) != nil else { throw CommandError("\(repo) is not installed.") }
            self.engine.library.delete(repo); return .object(["deleted": .string(repo)])
        }
        add("models.load", .act, "Load a language model into memory now.", props: [("repo", "string", "Repo id")], required: ["repo"]) { a, _ in
            let repo = try a.string("repo")
            guard self.engine.library.installedDirectory(for: repo) != nil else { throw CommandError("\(repo) is not installed.") }
            try await self.engine.loadModel(repo); return .object(["loaded": .string(repo)])
        }
        add("models.unload", .act, "Unload language models from memory.") { _, _ in
            await self.engine.unloadModels(); return .object(["unloaded": .bool(true)])
        }
        add("providers.list", .read, "Model providers: the built-in MLX one and the ones in settings.json.") { _, _ in
            var rows: [JSON] = [.object(["id": .string("mlx"), "kind": .string("mlx"), "locality": .string("local"), "enabled": .bool(true),
                                         "models": .int(self.engine.library.installed(kind: .language).count)])]
            for c in self.engine.settings.providers {
                var o: [String: JSON] = ["id": .string(c.id), "kind": .string(c.kind), "baseURL": .string(c.baseURL), "locality": .string(c.locality.rawValue),
                                         "enabled": .bool(c.enabled), "has_service": .bool(c.service != nil)]
                if let p = c.problem { o["problem"] = .string(p) }
                if let s = self.engine.router.provider(c.id)?.process?.snapshot() { o["state"] = .string(s.state.label) }
                if let m = c.models { o["models"] = .strings(m) }
                rows.append(.object(o))
            }
            return .array(rows)
        }
        add("providers.test", .read, "Check a provider answers and list its models. Starts its server if it has one.", props: [("id", "string", "Provider id")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            if id == "mlx" { return .object(["ok": .bool(true), "models": .strings(self.engine.library.installed(kind: .language).map(\.repo))]) }
            guard let p = self.engine.router.provider(id) else {
                if let c = self.engine.settings.providers.first(where: { $0.id == id }), let why = c.problem { throw CommandError("Provider \"\(id)\" is misconfigured: \(why).") }
                throw CommandError("No provider \"\(id)\". Use providers.list.")
            }
            guard p.config.enabled else { throw CommandError("Provider \"\(id)\" is off. Turn it on in System first.") }
            do {
                let started = Date()
                let ms = try await p.check()
                return .object(["ok": .bool(true), "models": .strings(ms.map(\.qualified)), "seconds": .double(Date().timeIntervalSince(started))])
            } catch { return .object(["ok": .bool(false), "error": .string(error.localizedDescription)]) }
        }
    }

    // MARK: services

    func registerServices() {
        add("services.list", .read, "Supervised services: state, memory, last used.") { _, _ in
            await self.engine.refreshServices()
            return .array(self.engine.services.map { s in
                .object(["id": .string(s.id), "kind": .string(s.kind), "state": .string(s.state.label),
                         "error": { if case .failed(let e) = s.state { return JSON.string(e) } else { return .null } }(),
                         "memory_bytes": s.memoryBytes.map { .int(Int($0)) } ?? .null, "lifecycle": s.lifecycle.map { .string($0) } ?? .null,
                         "restarts": .int(s.restarts), "detail": .string(s.detail)])
            })
        }
        add("services.logs", .read, "The end of a service's log.", props: [("id", "string", "Service id, e.g. mcp:fs or image"), ("lines", "integer", "Default 40")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            let lines = max(1, min(500, a.optInt("lines") ?? 40))
            return .string(ServiceLog.tail(home: self.engine.home, service: id, lines: lines, banners: true))
        }
        add("services.start", .act, "Start a service.", props: [("id", "string", "Service id")], required: ["id"]) { a, _ in
            try await self.engine.controlService(try a.string("id"), .start); return .object(["ok": .bool(true)])
        }
        add("services.stop", .act, "Stop a service.", props: [("id", "string", "Service id")], required: ["id"]) { a, _ in
            try await self.engine.controlService(try a.string("id"), .stop); return .object(["ok": .bool(true)])
        }
        add("services.restart", .act, "Restart a service.", props: [("id", "string", "Service id")], required: ["id"]) { a, _ in
            try await self.engine.controlService(try a.string("id"), .restart); return .object(["ok": .bool(true)])
        }
        add("home.paths", .read, "Where everything lives on disk.") { _, _ in
            let h = self.engine.home
            return .object(["root": .string(h.root.path), "agents": .string(h.agentsDir.path), "chats": .string(h.chatsDir.path),
                            "models": .string(h.modelsDir.path), "media": .string(h.mediaDir.path), "logs": .string(h.logsDir.path),
                            "modules": .string(h.modulesDir.path), "mcp_json": .string(h.mcpConfig.path), "settings_json": .string(h.settingsFile.path)])
        }
        add("home.reveal", .read, "Show a folder in Finder. Does nothing when called over the wire.", props: [("what", "string", "root, agents, chats, models, media, logs or modules")]) { a, c in
            .object(["revealed": .bool(false), "note": .string("Finder is only opened from the app's own buttons.")])
        }
    }

    // MARK: settings

    func registerSettings() {
        add("settings.get", .read, "Machine settings (settings.json), or one value by dotted path such as llm.topP.", props: [("path", "string", "Optional dotted path")]) { a, _ in
            let all = self.engine.settings.publicJSON
            guard let p = a.optString("path") else { return all }
            guard let spec = SettingSpec.table[p] else { throw CommandError("Unknown setting \"\(p)\". Known: \(SettingSpec.table.keys.sorted().joined(separator: ", ")).") }
            return spec.get(self.engine.settings)
        }
        add("settings.set", .configure, "Change one setting by dotted path. Out-of-range values are refused.",
            props: [("path", "string", "e.g. memoryBudgetFraction or llm.keepLoadedMinutes"), ("value", "any", "New value")], required: ["path", "value"]) { a, _ in
            let p = try a.string("path")
            guard let spec = SettingSpec.table[p] else { throw CommandError("Unknown setting \"\(p)\". Known: \(SettingSpec.table.keys.sorted().joined(separator: ", ")).") }
            guard let v = a.raw["value"] else { throw CommandError("Missing \"value\".") }
            var s = self.engine.settings
            try spec.set(&s, v)
            self.engine.updateSettings { $0 = s }
            return spec.get(self.engine.settings)
        }
        add("settings.reset", .configure, "Reset one setting, or all of them, to defaults.", props: [("path", "string", "Optional dotted path")]) { a, _ in
            if let p = a.optString("path") {
                guard let spec = SettingSpec.table[p] else { throw CommandError("Unknown setting \"\(p)\".") }
                var s = self.engine.settings
                try spec.set(&s, spec.get(ParlorSettings()))
                self.engine.updateSettings { $0 = s }
            } else {
                let keep = self.engine.settings
                self.engine.updateSettings { $0 = ParlorSettings(); $0.tokens = keep.tokens; $0.moduleApprovals = keep.moduleApprovals; $0.parlorServer = keep.parlorServer }
            }
            return self.engine.settings.publicJSON
        }
    }

    // MARK: tools

    func registerTools() {
        add("tools.list", .read, "Mounted tool servers and the tools they offer.") { _, _ in
            let servers = await self.engine.hub.allStatuses()
            let tools = await self.engine.hub.allTools()
            return .object([
                "servers": .array(servers.map { s in
                    var state = "ready"
                    switch s.state { case .connecting: state = "starting"; case .failed(let e): state = "failed: " + e; case .stopped: state = "stopped"; case .ready: break }
                    return .object(["id": .string(s.id), "kind": .string(s.kind), "state": .string(state), "tools": .int(s.toolCount)])
                }),
                "tools": .array(tools.map { .object(["name": .string($0.qualified), "description": .string($0.description)]) }),
            ])
        }
        add("modules.list", .read, "Modules in modules/: what each needs and whether the user has approved it. Approval is only given in the app.") { _, _ in
            .array(self.engine.modules.map { m in
                .object(["name": .string(m.name), "approved": .bool(m.approved), "error": m.error.map { .string($0) } ?? .null,
                         "description": .string(m.manifest?.description ?? ""), "tier": .string(m.manifest?.tier.rawValue ?? "none"),
                         "command": .string(([m.manifest?.command ?? ""] + (m.manifest?.args ?? [])).joined(separator: " "))])
            })
        }
        add("tools.mount", .configure, "Add an MCP server to mcp.json (stdio command or http url).",
            props: [("id", "string", "Name"), ("command", "string", "Executable"), ("args", "strings", "Arguments"), ("url", "string", "HTTP endpoint instead of a command"),
                    ("cwd", "string", "Working directory"), ("lifecycle", "string", "always or onDemand"), ("idle_minutes", "integer", "Idle stop for onDemand")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            guard id.range(of: "^[A-Za-z0-9_-]+$", options: .regularExpression) != nil else { throw CommandError("\"id\" may contain letters, digits, - and _ only.") }
            guard a.optString("command") != nil || a.optString("url") != nil else { throw CommandError("Give a \"command\" or a \"url\".") }
            if let l = a.optString("lifecycle"), !["always", "onDemand"].contains(l) { throw CommandError("\"lifecycle\" is always or onDemand.") }
            var f = MCPConfigFile.load(self.engine.home.mcpConfig)
            f.mcpServers[id] = MCPServerConfig(command: a.optString("command"), args: a.strings("args"), cwd: a.optString("cwd"), url: a.optString("url"),
                                               lifecycle: a.optString("lifecycle"), idleMinutes: a.optInt("idle_minutes"))
            try self.engine.saveMCPConfig(f)
            return .object(["mounted": .string(id)])
        }
        add("tools.unmount", .configure, "Remove a server from mcp.json.", props: [("id", "string", "Name")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            guard self.engine.configuredServers[id] != nil else { throw CommandError("No mounted server \"\(id)\".") }
            self.engine.deleteServer(id); return .object(["unmounted": .string(id)])
        }
        add("tools.enable", .configure, "Turn a mounted server on or off without losing its settings.", props: [("id", "string", "Name"), ("enabled", "boolean", "On or off")], required: ["id", "enabled"]) { a, _ in
            let id = try a.string("id")
            guard self.engine.configuredServers[id] != nil else { throw CommandError("No mounted server \"\(id)\".") }
            self.engine.setServer(id, enabled: a.bool("enabled", default: true)); return .object(["id": .string(id), "enabled": .bool(a.bool("enabled", default: true))])
        }
    }
}

// MARK: settings table

/// Every setting that can be changed by name, with its type and range. This is what `settings.set` checks against.
struct SettingSpec {
    var get: (ParlorSettings) -> JSON
    var set: (inout ParlorSettings, JSON) throws -> Void

    static func int(_ r: ClosedRange<Int>, _ kp: WritableKeyPath<ParlorSettings, Int>) -> SettingSpec {
        SettingSpec(get: { .int($0[keyPath: kp]) }, set: { s, v in
            guard let i = v.intValue, r.contains(i) else { throw CommandError("Expected a whole number from \(r.lowerBound) to \(r.upperBound).") }
            s[keyPath: kp] = i })
    }
    static func optInt(_ r: ClosedRange<Int>, _ kp: WritableKeyPath<ParlorSettings, Int?>, zeroIsOff: Bool = true) -> SettingSpec {
        SettingSpec(get: { $0[keyPath: kp].map { .int($0) } ?? .int(0) }, set: { s, v in
            guard let i = v.intValue, i == 0 || r.contains(i) else { throw CommandError("Expected 0 (off) or a whole number from \(r.lowerBound) to \(r.upperBound).") }
            s[keyPath: kp] = i == 0 ? nil : i })
    }
    static func dbl(_ r: ClosedRange<Double>, _ kp: WritableKeyPath<ParlorSettings, Double>) -> SettingSpec {
        SettingSpec(get: { .double($0[keyPath: kp]) }, set: { s, v in
            guard let d = v.doubleValue, r.contains(d) else { throw CommandError("Expected a number from \(r.lowerBound) to \(r.upperBound).") }
            s[keyPath: kp] = d })
    }
    static func bool(_ kp: WritableKeyPath<ParlorSettings, Bool>) -> SettingSpec {
        SettingSpec(get: { .bool($0[keyPath: kp]) }, set: { s, v in
            guard let b = v.boolValue else { throw CommandError("Expected true or false.") }
            s[keyPath: kp] = b })
    }

    static let table: [String: SettingSpec] = [
        "memoryBudgetFraction": .dbl(0.1...0.95, \.memoryBudgetFraction),
        "cacheSlotsPerModel": .int(1...16, \.cacheSlotsPerModel),
        "llm.promptCache": .bool(\.llm.promptCache),
        "llm.kvBits": SettingSpec(get: { .int($0.llm.kvBits ?? 0) }, set: { s, v in
            guard let i = v.intValue, [0, 4, 8].contains(i) else { throw CommandError("Expected 0 (full), 4 or 8.") }
            s.llm.kvBits = i == 0 ? nil : i }),
        "llm.kvGroupSize": .int(16...256, \.llm.kvGroupSize),
        "llm.quantizedKVStart": .int(0...1_000_000, \.llm.quantizedKVStart),
        "llm.maxKVSize": .optInt(256...1_000_000, \.llm.maxKVSize),
        "llm.prefillStepSize": .int(64...8192, \.llm.prefillStepSize),
        "llm.topP": .dbl(0.01...1, \.llm.topP),
        "llm.repetitionPenalty": SettingSpec(get: { .double($0.llm.repetitionPenalty ?? 0) }, set: { s, v in
            guard let d = v.doubleValue, d == 0 || (0.5...2).contains(d) else { throw CommandError("Expected 0 (off) or a number from 0.5 to 2.") }
            s.llm.repetitionPenalty = d == 0 ? nil : d }),
        "llm.maxReplyTokens": SettingSpec(get: { .int($0.llm.maxReplyTokens ?? 0) }, set: { s, v in
            guard let i = v.intValue, i == 0 || (16...131_072).contains(i) else { throw CommandError("Expected 0 (no cap) or a whole number from 16 to 131072.") }
            s.llm.maxReplyTokens = i == 0 ? nil : i }),
        "llm.keepLoadedMinutes": .int(0...10_080, \.llm.keepLoadedMinutes),
        "image.precision": SettingSpec(get: { .string($0.image.precision) }, set: { s, v in
            guard let p = v.stringValue, ["bf16", "int8", "int4"].contains(p) else { throw CommandError("Expected bf16, int8 or int4.") }
            s.image.precision = p }),
        "image.fastSteps": .optInt(1...100, \.image.fastSteps),
        "image.detailedSteps": .optInt(1...100, \.image.detailedSteps),
        "image.fastSide": .int(256...2048, \.image.fastSide),
        "image.detailedSide": .int(256...2048, \.image.detailedSide),
        "image.idleSeconds": .int(0...86_400, \.image.idleSeconds),
        "loop.maxSteps": .int(1...64, \.loop.maxSteps),
        "loop.toolResultChars": .int(100...200_000, \.loop.toolResultChars),
        "loop.toolTimeoutSeconds": .int(1...86_400, \.loop.toolTimeoutSeconds),
        "loop.keepRecent": .int(0...200, \.loop.keepRecent),
        "loop.summaryTokens": .int(64...8192, \.loop.summaryTokens),
        "parlorServer.enabled": .bool(\.parlorServer.enabled),
        "parlorServer.port": .int(1024...65535, \.parlorServer.port),
    ]
}

extension ParlorSettings {
    /// The settings as JSON without secrets (token hashes) or approvals.
    var publicJSON: JSON {
        var s = self
        s.tokens = []; s.moduleApprovals = [:]
        guard let d = try? JSONEncoder().encode(s), var j = try? JSONDecoder().decode(JSON.self, from: d), case .object(var o) = j else { return .null }
        o["tokens"] = nil; o["moduleApprovals"] = nil
        j = .object(o)
        return j
    }
}

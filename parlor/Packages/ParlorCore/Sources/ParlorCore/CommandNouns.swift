import Foundation

// The uniform noun verbs: schema.get, and list/read/validate/create/update/delete for engines, hosts and harnesses,
// plus the draft verbs for agents (validate, try). One JSON shape: the file is the verb input and output.

// MARK: - schemas

/// Hand-written JSON Schemas, one per noun, beside the structs they describe. A test validates every seeded file against them.
public enum NounSchemas {
    public static var nouns: [String] { NounRegistry.ids }

    static func parse(_ s: String) -> JSON { (try? JSONDecoder().decode(JSON.self, from: Data(s.utf8))) ?? .null }

    public static func schema(for noun: String) -> JSON? {
        switch noun {
        case "agent": return parse(agent)
        case "host": return parse(host)
        case "harness": return parse(harness)
        case "engine": return parse(engine)
        default: return nil
        }
    }

    static let agent = """
    { "type": "object", "description": "agents/<id>/agent.json. The persona is a separate file; in drafts pass it as \\"persona\\".",
      "required": ["id", "name"],
      "properties": {
        "version": { "type": "integer" },
        "id": { "type": "string", "description": "letters, digits, - and _; also the folder name" },
        "name": { "type": "string" },
        "description": { "type": "string", "description": "one line shown in the sidebar" },
        "persona": { "type": "string", "format": "multiline", "description": "the system prompt (drafts only; saved as persona.md)" },
        "memory": { "type": "string", "format": "multiline", "description": "read-only: agents.read returns it; create and update ignore it (use memory.append)" },
        "harness": { "type": "string", "description": "a harness id from harnesses.list; blank means chat" },
        "hosts": { "type": "object", "description": "{ \\"chat\\": host id, \\"image\\": host id }; see hosts.list" },
        "tools": { "type": "array", "items": { "type": "string" }, "description": "patterns: *, server.*, server.tool" },
        "tier": { "type": "string", "enum": ["none", "read", "act", "configure"] },
        "group": { "type": "string" },
        "color": { "type": "integer", "minimum": 0, "maximum": 5 },
        "config": { "type": "object", "description": "the harness's settings; harnesses.read shows its schema" } } }
    """

    static let host = """
    { "type": "object", "description": "hosts/<id>.json: an engine serving a model at the gateway.",
      "required": ["id", "engine"],
      "properties": {
        "version": { "type": "integer" },
        "id": { "type": "string" },
        "engine": { "type": "string", "description": "an engine id from engines.list" },
        "model": { "type": "string", "description": "the model repo, for engines that serve local weights" },
        "settings": { "type": "object", "description": "how the model runs; engines.read shows the schema" },
        "defaults": { "type": "object", "description": "request defaults used when an agent is silent: temperature, top_p, min_p, repetition_penalty, max_tokens" },
        "enabled": { "type": "boolean" },
        "url": { "type": "string", "description": "remote engines: the server's /v1 URL" },
        "keyRef": { "type": "string", "description": "remote engines: the Keychain item holding the API key" },
        "models": { "type": "array", "items": { "type": "string" }, "description": "remote engines: the model names to offer" } } }
    """

    static let harness = """
    { "type": "object", "description": "harnesses/<id>/harness.json: a program that runs an agent's turns.",
      "required": ["id", "command"],
      "properties": {
        "managed": { "type": "string" },
        "version": { "type": "integer" },
        "id": { "type": "string" },
        "title": { "type": "string" },
        "description": { "type": "string" },
        "command": { "type": "string" },
        "args": { "type": "array", "items": { "type": "string" } },
        "needs": { "type": "object", "description": "{ \\"chat\\": \\"required\\", \\"image\\": \\"optional\\" }" },
        "config": { "type": "object", "description": "a JSON Schema subset for the agent's config" },
        "actions": { "type": "array", "description": "buttons: id, title, icon, placement composer|message|settings, needs, hint, args" },
        "templates": { "type": "array", "description": "starting points: id, title, description, persona, tools, config, group, image" } } }
    """

    static let engine = """
    { "type": "object", "description": "engines/<id>/engine.json: a backend that serves models.",
      "required": ["id", "adapter"],
      "properties": {
        "managed": { "type": "string" },
        "version": { "type": "integer" },
        "id": { "type": "string" },
        "title": { "type": "string" },
        "capabilities": { "type": "array", "items": { "type": "string" } },
        "formats": { "type": "array", "items": { "type": "string" } },
        "adapter": { "type": "string", "enum": ["inprocess-mlx", "http-openai", "command", "mcp-stdio"] },
        "launch": { "description": "command and args, with {settings.x} placeholders" },
        "health": { "description": "how to tell the server is up" },
        "tool": { "description": "mcp-stdio: the tool to call" },
        "params": { "description": "which sampling parameters the engine honours" },
        "requires": { "type": "object" },
        "settings": { "type": "object", "description": "a JSON Schema subset that Tuning renders as a form" },
        "lifecycle": { "type": "object" },
        "remote": { "type": "boolean" } } }
    """

    /// Required keys the schema lists but the draft lacks, plus type and range problems, all at once.
    public static func check(_ draft: JSON, noun: String) -> [Issue] {
        guard let s = schema(for: noun), let o = draft.objectValue else { return [] }
        var out = SchemaCheck.validate(draft, schema: s)
        for k in s.objectValue?["required"]?.arrayValue?.compactMap(\.stringValue) ?? [] where o[k] == nil {
            out.append(Issue(code: "missing", path: k, message: "required"))
        }
        return out
    }
}

// MARK: - verbs

extension CommandCenter {
    func registerNouns() {
        registerSchema(); registerEngineVerbs(); registerHostVerbs(); registerHarnessVerbs(); registerDuplicateVerbs(); registerDraftVerbs()
    }

    private func ids(_ s: [String]) -> String { s.isEmpty ? "none yet" : s.joined(separator: ", ") }

    private func issuesJSON(_ issues: [Issue]) -> JSON {
        .object(["ok": .bool(issues.errors.isEmpty), "issues": issues.json])
    }

    private func schemaIssues(_ draft: JSON, noun: String) -> [Issue] {
        NounSchemas.check(draft, noun: noun).filter { $0.code != "unknown" || noun == "agent" || noun == "host" }
    }

    /// How a listed file reads: `ok`, its issues, and whether parlor ships it (`seeded`) or a person wrote it (`user`).
    private func entryFields<T>(_ e: NounEntry<T>) -> [String: JSON] {
        ["ok": .bool(e.ok), "issues": e.issues.json, "source": .string(e.source.rawValue)]
    }

    /// Files that do not load, so a typo shows up in a list instead of the file vanishing.
    private func brokenEntries<T>(_ scan: [NounEntry<T>]) -> [JSON] {
        scan.filter { $0.value == nil }.map { .object(["id": .string($0.id), "ok": .bool(false), "issues": $0.issues.json, "source": .string($0.source.rawValue)]) }
    }

    /// Shipped files are rewritten by parlor, so a verb that changed one would be undone at the next launch.
    private func refuseSeeded(_ noun: String, _ id: String, _ raw: [String: JSON]?) throws {
        guard raw?["managed"] != nil else { return }
        throw CommandError("\(noun) \"\(id)\" is built in: parlor rewrites it. Duplicate it first with parlor.\(noun == "engine" ? "engines" : "harnesses").duplicate {\"id\":\"\(id)\",\"new_id\":\"my-\(id)\"}, then change the copy.")
    }

    /// Code that will run needs the user's yes, whoever asks.
    private func askUser(_ what: String, _ caller: Caller) async throws {
        guard let ask = engine.confirm else { throw CommandError("This runs code on this Mac, so it needs the user. Nobody is available to confirm it.") }
        guard await ask("\(caller.name) wants to \(what)") else { throw CommandError("The user declined.") }
    }

    // MARK: schema

    func registerSchema() {
        add("schema.get", .read, "The JSON Schema of a noun (agent, host, harness, engine). The file, the verb input and the verb output all have this shape.",
            props: [("noun", "string", "agent, host, harness or engine")], required: ["noun"]) { a, _ in
            let n = try a.string("noun")
            guard let s = NounSchemas.schema(for: n) else { throw CommandError("No noun \"\(n)\". Nouns: \(NounSchemas.nouns.joined(separator: ", ")).") }
            return s
        }
    }

    // MARK: engines

    func registerEngineVerbs() {
        add("engines.list", .read, "Engines: id, title, capabilities, formats, adapter, and whether their binary is present. A file that does not load is listed with ok: false and its issues.") { _, _ in
            let cat = self.engine.catalog
            let scan = self.engine.engineStore.scan()
            let good: [JSON] = cat.engines.map { e in
                var o: [String: JSON] = ["id": .string(e.id), "title": .string(e.title), "capabilities": .strings(e.capabilities), "formats": .strings(e.formats),
                         "adapter": .string(e.adapter), "remote": .bool(e.remote), "binary": .bool(e.remote || e.requires?.binary == nil || cat.binary(for: e) != nil)]
                if let en = scan.first(where: { $0.id == e.id }) { o.merge(self.entryFields(en)) { _, n in n } }
                return .object(o)
            }
            return .array(good + self.brokenEntries(scan))
        }
        add("engines.read", .read, "One engine's manifest, including its settings schema.", props: [("id", "string", "Engine id")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            guard let raw = self.engine.engineStore.raw(id) else { throw CommandError("No engine \"\(id)\". Engines: \(self.ids(self.engine.engineStore.ids())).") }
            return .object(raw)
        }
        add("engines.validate", .read, "Check an engine manifest draft and return every problem at once. Writes nothing.", props: [("draft", "object", "engine.json contents")], required: ["draft"]) { a, _ in
            let d = a.raw["draft"] ?? .null
            return self.issuesJSON(self.engine.engineStore.validate(d) + self.schemaIssues(d, noun: "engine"))
        }
        add("engines.create", .configure, "Add an engine (always asks the user: its launch command runs on this Mac).", props: [("draft", "object", "engine.json contents")], required: ["draft"]) { a, c in
            let d = a.raw["draft"] ?? .null
            let issues = self.engine.engineStore.validate(d) + self.schemaIssues(d, noun: "engine")
            if !issues.errors.isEmpty { throw ValidationError(issues: issues) }
            try await self.askUser("add this engine, which can run programs:\n\n" + d.pretty, c)
            try self.engine.engineStore.create(d)
            self.engine.reloadRunplane()
            return .object(["created": d.objectValue?["id"] ?? .null])
        }
        add("engines.update", .configure, "Change an engine with a JSON merge patch (always asks the user).", props: [("id", "string", "Engine id"), ("patch", "object", "Merge patch")], required: ["id", "patch"]) { a, c in
            let id = try a.string("id")
            try self.refuseSeeded("engine", id, self.engine.engineStore.raw(id))
            try await self.askUser("change engine \(id), which can run programs:\n\n" + (a.raw["patch"] ?? .null).pretty, c)
            try self.engine.engineStore.update(id, patch: a.raw["patch"] ?? .null)
            self.engine.reloadRunplane()
            return .object(self.engine.engineStore.raw(id) ?? [:])
        }
        add("engines.delete", .configure, destructive: true, "Remove an engine manifest. Hosts on it stop working.", props: [("id", "string", "Engine id")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            guard let cur = self.engine.engineStore.raw(id) else { throw CommandError("No engine \"\(id)\". Engines: \(self.ids(self.engine.engineStore.ids())).") }
            try self.refuseSeeded("engine", id, cur)
            try self.engine.engineStore.delete(id)
            self.engine.reloadRunplane()
            return .object(["deleted": .string(id)])
        }
    }

    func registerDuplicateVerbs() {
        for (noun, plural, tool) in [("engine", "engines", "which can run programs"), ("harness", "harnesses", "a program that runs agents' turns")] {
            add("\(plural).duplicate", .configure, "Copy a \(noun) under a new id. The copy is yours to edit; built-in ones are read-only. Asks the user: it can run code.",
                props: [("id", "string", "\(noun.capitalized) id"), ("new_id", "string", "Id for the copy; default my-<id>")], required: ["id"]) { a, c in
                let id = try a.string("id")
                var newID = a.optString("new_id") ?? ""
                if newID.isEmpty { newID = "my-" + id }
                try await self.askUser("copy \(noun) \(id) as \(newID), \(tool)", c)
                if noun == "engine" { try self.engine.engineStore.duplicate(id, as: newID) } else { try self.engine.harnessStore.duplicate(id, as: newID) }
                self.engine.reloadRunplane()
                return .object(["duplicated": .string(id), "id": .string(newID)])
            }
        }
    }

    // MARK: hosts

    /// Everything wrong with a host draft: shape, then references and settings against the engine.
    func hostIssues(_ d: JSON) -> [Issue] {
        var out = engine.hostStore.validate(d) + schemaIssues(d, noun: "host")
        guard let o = d.objectValue else { return out }
        let cat = engine.catalog
        if let eid = o["engine"]?.stringValue {
            if let e = cat.engine(eid) {
                if let s = e.settings, let hs = o["settings"], hs.objectValue?.isEmpty == false {
                    out += SchemaCheck.validate(hs, schema: s, path: "settings").filter { $0.code != "unknown" }
                    for i in SchemaCheck.validate(hs, schema: s, path: "settings") where i.code == "unknown" { out.append(i) }
                }
                if e.remote, (o["url"]?.stringValue ?? "").isEmpty {
                    out.append(Issue(code: "missing", path: "url", message: "required for engine \(eid)", hint: "for example http://127.0.0.1:1234/v1"))
                }
                if !e.remote, let m = o["model"]?.stringValue, !m.isEmpty, !cat.isInstalled(m), !e.formats.isEmpty {
                    out.append(Issue(code: "missing_model", path: "model", message: "\(m) is not installed", hint: "installed: " + self.ids(cat.models.map(\.repo)), severity: .warning))
                }
            } else {
                out.append(Issue(code: "unknown_engine", path: "engine", message: "no engine \"\(eid)\"", hint: "engines: " + ids(cat.engines.map(\.id))))
            }
        }
        return out
    }

    private func hostJSON(_ h: HostInfo) -> JSON {
        let cat = engine.catalog
        var o = (try? JSONDecoder().decode(JSON.self, from: JSONEncoder().encode(h.spec)))?.objectValue ?? [:]
        o["auto"] = .bool(h.isAuto)
        o["capabilities"] = .strings(cat.capabilities(of: h))
        o["status"] = .string(cat.status(host: h.id).label)
        return .object(o)
    }

    func registerHostVerbs() {
        add("hosts.list", .read, "Hosts (an engine serving a model): id, engine, model, capabilities, status. Auto hosts are computed from installed models.") { _, _ in
            let scan = self.engine.hostStore.scan()
            let good: [JSON] = self.engine.catalog.hosts.map { h in
                var o = self.hostJSON(h).objectValue ?? [:]
                if let en = scan.first(where: { $0.id == h.id }) { o.merge(self.entryFields(en)) { _, n in n } } else { o["ok"] = .bool(true); o["issues"] = .array([]); o["source"] = .string("auto") }
                return .object(o)
            }
            return .array(good + self.brokenEntries(scan))
        }
        add("hosts.read", .read, "One host in full, with its status and last benchmark.", props: [("id", "string", "Host id")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            guard let h = self.engine.catalog.host(id) else { throw CommandError("No host \"\(id)\". Hosts: \(self.ids(self.engine.catalog.hosts.map(\.id))).") }
            var o = self.hostJSON(h).objectValue ?? [:]
            if let b = self.benchRuns(id).last { o["lastBench"] = b }
            return .object(o)
        }
        add("hosts.validate", .read, "Check a host draft (engine exists, settings fit the engine's schema, model installed) and return every problem at once.",
            props: [("draft", "object", "hosts/<id>.json contents")], required: ["draft"]) { a, _ in
            self.issuesJSON(self.hostIssues(a.raw["draft"] ?? .null))
        }
        add("hosts.create", .configure, "Create a host from a draft. Same checks as hosts.validate.", props: [("draft", "object", "hosts/<id>.json contents")], required: ["draft"]) { a, _ in
            let d = a.raw["draft"] ?? .null
            let issues = self.hostIssues(d)
            if !issues.errors.isEmpty { throw ValidationError(issues: issues) }
            try self.engine.hostStore.create(d)
            self.engine.reloadRunplane()
            return self.hostJSON(self.engine.catalog.host(d.objectValue?["id"]?.stringValue ?? "")!)
        }
        add("hosts.update", .configure, "Change a host with a JSON merge patch (null removes a key). Explicit hosts only; duplicate an auto host first.",
            props: [("id", "string", "Host id"), ("patch", "object", "Merge patch")], required: ["id", "patch"]) { a, _ in
            let id = try a.string("id")
            guard let cur = self.engine.hostStore.raw(id) else {
                if self.engine.catalog.host(id)?.isAuto == true { throw CommandError("\"\(id)\" is an auto host (computed from an installed model). Use hosts.duplicate to make an editable one.") }
                throw CommandError("No host \"\(id)\". Hosts: \(self.ids(self.engine.hostStore.ids())).")
            }
            var merged = NounStore<HostSpec>.merge(.object(cur), a.raw["patch"] ?? .null).objectValue ?? cur
            merged["id"] = .string(id)
            let issues = self.hostIssues(.object(merged))
            if !issues.errors.isEmpty { throw ValidationError(issues: issues) }
            try self.engine.hostStore.update(id, patch: a.raw["patch"] ?? .null)
            self.engine.reloadRunplane()
            return self.hostJSON(self.engine.catalog.host(id)!)
        }
        add("hosts.delete", .configure, destructive: true, "Delete a host file. Agents that use it will show it as missing.", props: [("id", "string", "Host id")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            guard self.engine.hostStore.raw(id) != nil else { throw CommandError("No explicit host \"\(id)\". Hosts: \(self.ids(self.engine.hostStore.ids())).") }
            try self.engine.hostStore.delete(id)
            try? FileManager.default.removeItem(at: self.benchURL(id))
            self.engine.reloadRunplane()
            return .object(["deleted": .string(id)])
        }
        add("hosts.duplicate", .configure, "Copy a host (auto hosts too) under a new id, to A/B different settings on the same model.",
            props: [("id", "string", "Host id"), ("new_id", "string", "Id for the copy; default <id>-2"), ("settings", "object", "Merge patch for the copy's settings")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            guard let src = self.engine.catalog.host(id) else { throw CommandError("No host \"\(id)\". Hosts: \(self.ids(self.engine.catalog.hosts.map(\.id))).") }
            let taken = Set(self.engine.catalog.hosts.map(\.id))
            var newID = a.optString("new_id") ?? ""
            if newID.isEmpty { var n = 2; while taken.contains("\(id)-\(n)") { n += 1 }; newID = "\(id)-\(n)" }
            var spec = src.spec; spec.id = newID
            if let s = a.raw["settings"] { spec.settings = NounStore<HostSpec>.merge(.object(spec.settings), s).objectValue ?? spec.settings }
            let d = try JSONDecoder().decode(JSON.self, from: JSONEncoder().encode(spec))
            let issues = self.hostIssues(d)
            if !issues.errors.isEmpty { throw ValidationError(issues: issues) }
            try self.engine.hostStore.create(d)
            self.engine.reloadRunplane()
            return self.hostJSON(self.engine.catalog.host(newID)!)
        }
        add("hosts.enable", .configure, "Turn a host on or off.", props: [("id", "string", "Host id"), ("enabled", "boolean", "On or off")], required: ["id", "enabled"]) { a, _ in
            let id = try a.string("id")
            guard self.engine.hostStore.raw(id) != nil else { throw CommandError("No explicit host \"\(id)\". Hosts: \(self.ids(self.engine.hostStore.ids())).") }
            try self.engine.hostStore.update(id, patch: .object(["enabled": .bool(a.bool("enabled", default: true))]))
            self.engine.reloadRunplane()
            return self.hostJSON(self.engine.catalog.host(id)!)
        }
        add("hosts.start", .act, "Warm a host: serve one token so its model is loaded.", props: [("id", "string", "Host id")], required: ["id"]) { a, c in
            let id = try a.string("id")
            let r = try await self.hostRequest(id, body: self.chatBody(id, "Hi", maxTokens: 1), caller: c)
            return .object(["ok": .bool(r.status == 200), "status": .string(self.engine.catalog.status(host: id).label)])
        }
        add("hosts.stop", .act, "Unload a host's model from memory.", props: [("id", "string", "Host id")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            guard let h = self.engine.catalog.host(id) else { throw CommandError("No host \"\(id)\".") }
            if h.spec.engine == "mlx" { await self.engine.unloadModels() }
            else if h.spec.engine == "z-image" { try? await self.engine.controlService("image", .stop) }
            else { try? await self.engine.controlService("provider:" + id, .stop) }
            return .object(["status": .string(self.engine.catalog.status(host: id).label)])
        }
        add("hosts.test", .read, "Check a host can answer: its status, and for a remote host whether the server lists its models.", props: [("id", "string", "Host id")], required: ["id"]) { a, c in
            let id = try a.string("id")
            guard let h = self.engine.catalog.host(id) else { throw CommandError("No host \"\(id)\". Hosts: \(self.ids(self.engine.catalog.hosts.map(\.id))).") }
            let st = self.engine.catalog.status(host: id)
            var o: [String: JSON] = ["host": .string(id), "status": .string(st.label), "usable": .bool(st.usable)]
            if let url = h.spec.url, let u = URL(string: url.hasSuffix("/") ? url + "models" : url + "/models") {
                var req = URLRequest(url: u); req.timeoutInterval = 5
                do {
                    let (data, resp) = try await URLSession.shared.data(for: req)
                    o["reachable"] = .bool((resp as? HTTPURLResponse)?.statusCode == 200)
                    if let j = try? JSONDecoder().decode(JSON.self, from: data), let items = j.objectValue?["data"]?.arrayValue {
                        o["models"] = .strings(items.compactMap { $0.objectValue?["id"]?.stringValue })
                    }
                } catch { o["reachable"] = .bool(false); o["error"] = .string(error.localizedDescription) }
            }
            return .object(o)
        }
        add("hosts.stats", .read, "Live numbers for a host: status, memory, and the last benchmark runs.", props: [("id", "string", "Host id")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            guard self.engine.catalog.host(id) != nil else { throw CommandError("No host \"\(id)\". Hosts: \(self.ids(self.engine.catalog.hosts.map(\.id))).") }
            await self.engine.refreshServices()
            let mem = self.engine.services.first { $0.id == "provider:" + id || $0.id == "llm" || $0.id == "image" }?.memoryBytes
            return .object(["host": .string(id), "status": .string(self.engine.catalog.status(host: id).label),
                            "memory_bytes": mem.map { .int(Int($0)) } ?? .null, "bench": .array(self.benchRuns(id))])
        }
        add("hosts.bench", .act, "Benchmark a host: a fixed three-turn chat (text) or a fixed prompt at fast and detailed size (image). Saves the run, with the settings that produced it, to hosts/<id>.bench.json.",
            props: [("id", "string", "Host id")], required: ["id"]) { a, c in
            try await self.bench(try a.string("id"), caller: c)
        }
    }

    // MARK: bench

    func benchURL(_ id: String) -> URL { engine.home.hostsDir.appendingPathComponent(id + ".bench.json") }

    func benchRuns(_ id: String) -> [JSON] {
        guard let d = try? Data(contentsOf: benchURL(id)), let j = try? JSONDecoder().decode(JSON.self, from: d) else { return [] }
        return j.arrayValue ?? []
    }

    private func chatBody(_ host: String, _ text: String, maxTokens: Int = 64, stream: Bool = false) -> JSON {
        var o: [String: JSON] = ["model": .string(host), "messages": .array([.object(["role": .string("user"), "content": .string(text)])]),
                                 "max_tokens": .int(maxTokens), "stream": .bool(stream)]
        if stream { o["stream_options"] = .object(["include_usage": .bool(true)]) }
        return .object(o)
    }

    /// POST to our own gateway with a one-shot token scoped to this host, so a benchmark takes the same path as a harness.
    private func hostRequest(_ host: String, path: String = "/v1/chat/completions", body: JSON, caller: Caller) async throws -> (status: Int, data: Data) {
        await engine.ensureGateway()
        guard engine.endpoint.isRunning else { throw CommandError("The gateway is off. Turn it on in System.") }
        let token = engine.endpoint.issueEphemeral(name: "bench", tier: .act, hosts: [host])
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(engine.endpoint.port)\(path)")!)
        req.httpMethod = "POST"; req.timeoutInterval = 900
        req.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("background", forHTTPHeaderField: "X-Parlor-Priority")
        req.httpBody = try JSONEncoder().encode(body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        return ((resp as? HTTPURLResponse)?.statusCode ?? 0, data)
    }

    private func bench(_ id: String, caller: Caller) async throws -> JSON {
        guard let h = engine.catalog.host(id) else { throw CommandError("No host \"\(id)\". Hosts: \(ids(engine.catalog.hosts.map(\.id))).") }
        let caps = engine.catalog.capabilities(of: h)
        var result: [String: JSON] = ["at": .string(ISO8601DateFormatter().string(from: Date())), "host": .string(id),
                                      "settings": .object(h.spec.settings), "defaults": .object(h.spec.defaults)]
        if caps.contains("chat") {
            let prompts = ["Name three rivers in Europe.", "Which of those is longest?", "Write two sentences about it."]
            var firsts: [Double] = [], speeds: [Double] = []
            for p in prompts {
                let t0 = Date()
                let r = try await hostRequest(id, body: chatBody(id, p, maxTokens: 96), caller: caller)
                guard r.status == 200 else { throw CommandError("Host \(id) answered \(r.status): " + String(decoding: r.data.prefix(300), as: UTF8.self)) }
                let secs = Date().timeIntervalSince(t0)
                let j = try? JSONDecoder().decode(JSON.self, from: r.data)
                let toks = j?.objectValue?["usage"]?.objectValue?["completion_tokens"]?.intValue ?? 0
                firsts.append(secs)
                if toks > 0, secs > 0 { speeds.append(Double(toks) / secs) }
            }
            func r2(_ d: Double) -> JSON { .double((d * 100).rounded() / 100) }
            result["kind"] = .string("text")
            result["seconds_per_turn"] = .array(firsts.map(r2))
            result["tokens_per_second"] = .array(speeds.map(r2))
            if let s = speeds.last { result["tokens_per_second_last"] = r2(s) }
        } else if caps.contains("image") {
            var timings: [String: JSON] = [:]
            for (name, size) in [("fast", "512x512"), ("detailed", "768x768")] {
                let t0 = Date()
                let body: JSON = .object(["model": .string(id), "prompt": .string("a lighthouse at dusk, painting"), "size": .string(size), "n": .int(1)])
                let r = try await hostRequest(id, path: "/v1/images/generations", body: body, caller: caller)
                guard r.status == 200 else { throw CommandError("Host \(id) answered \(r.status): " + String(decoding: r.data.prefix(300), as: UTF8.self)) }
                timings[name] = .double(((Date().timeIntervalSince(t0)) * 100).rounded() / 100)
            }
            result["kind"] = .string("image"); result["seconds_per_image"] = .object(timings)
        } else { throw CommandError("Host \(id) has no chat or image capability to benchmark.") }
        var runs = benchRuns(id)
        runs.append(.object(result))
        if runs.count > 10 { runs = Array(runs.suffix(10)) }
        try FileManager.default.createDirectory(at: engine.home.hostsDir, withIntermediateDirectories: true)
        try JSONEncoder().encode(JSON.array(runs)).write(to: benchURL(id), options: .atomic)
        return .object(["run": .object(result), "previous": runs.count > 1 ? runs[runs.count - 2] : .null])
    }

    // MARK: harnesses

    func registerHarnessVerbs() {
        add("harnesses.list", .read, "Harnesses: id, title, what they need, their templates and actions. A folder that does not load is listed with ok: false and its issues.") { _, _ in
            let scan = self.engine.harnessStore.scan()
            let good: [JSON] = self.engine.harnessList.map { h in
                var o: [String: JSON] = ["id": .string(h.id), "title": .string(h.title), "description": .string(h.description),
                         "needs": .object(h.needs.mapValues { .string($0) }), "templates": .strings(h.templates.map(\.id)), "actions": .strings(h.actions.map(\.id))]
                if let en = scan.first(where: { $0.id == h.id }) { o.merge(self.entryFields(en)) { _, n in n } }
                return .object(o)
            }
            return .array(good + self.brokenEntries(scan))
        }
        add("harnesses.read", .read, "One harness manifest in full: command, needs, the config schema agents' config must fit, actions and templates.",
            props: [("id", "string", "Harness id")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            guard let raw = self.engine.harnessStore.raw(id) else { throw CommandError("No harness \"\(id)\". Harnesses: \(self.ids(self.engine.harnessStore.ids())).") }
            return .object(raw)
        }
        add("harnesses.validate", .read, "Check a harness manifest draft and return every problem at once. Writes nothing.", props: [("draft", "object", "harness.json contents")], required: ["draft"]) { a, _ in
            let d = a.raw["draft"] ?? .null
            return self.issuesJSON(self.engine.harnessStore.validate(d) + self.schemaIssues(d, noun: "harness"))
        }
        add("harnesses.create", .configure, "Add a harness: a manifest plus script files (always asks the user, showing both).",
            props: [("draft", "object", "harness.json contents"), ("files", "object", "{ filename: text } script files written beside the manifest")], required: ["draft"]) { a, c in
            let d = a.raw["draft"] ?? .null
            let issues = self.engine.harnessStore.validate(d) + self.schemaIssues(d, noun: "harness")
            if !issues.errors.isEmpty { throw ValidationError(issues: issues) }
            let files = a.raw["files"]?.objectValue ?? [:]
            for name in files.keys where name.contains("/") || name.hasPrefix(".") {
                throw ValidationError(issues: [Issue(path: "files." + name, message: "file names are plain, with no folders")])
            }
            let shown = files.sorted { $0.key < $1.key }.map { "--- \($0.key) ---\n" + ($0.value.stringValue ?? "") }.joined(separator: "\n")
            try await self.askUser("add this harness, a program that will run agents' turns:\n\n" + d.pretty + "\n\n" + shown, c)
            try self.engine.harnessStore.create(d)
            let dir = self.engine.home.harnessesDir.appendingPathComponent(d.objectValue!["id"]!.stringValue!)
            for (name, text) in files { try (text.stringValue ?? "").write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8) }
            self.engine.reloadRunplane()
            return .object(["created": d.objectValue?["id"] ?? .null])
        }
        add("harnesses.update", .configure, "Change a harness manifest with a merge patch (always asks the user).", props: [("id", "string", "Harness id"), ("patch", "object", "Merge patch")], required: ["id", "patch"]) { a, c in
            let id = try a.string("id")
            try self.refuseSeeded("harness", id, self.engine.harnessStore.raw(id))
            try await self.askUser("change harness \(id), a program that runs agents' turns:\n\n" + (a.raw["patch"] ?? .null).pretty, c)
            try self.engine.harnessStore.update(id, patch: a.raw["patch"] ?? .null)
            self.engine.reloadRunplane()
            return .object(self.engine.harnessStore.raw(id) ?? [:])
        }
        add("harnesses.delete", .configure, destructive: true, "Delete a harness folder. Agents that use it will not run.", props: [("id", "string", "Harness id")], required: ["id"]) { a, _ in
            let id = try a.string("id")
            guard let cur = self.engine.harnessStore.raw(id) else { throw CommandError("No harness \"\(id)\". Harnesses: \(self.ids(self.engine.harnessStore.ids())).") }
            try self.refuseSeeded("harness", id, cur)
            try self.engine.harnessStore.delete(id)
            self.engine.reloadRunplane()
            return .object(["deleted": .string(id)])
        }
    }

    // MARK: agent drafts

    /// Every problem with an agent draft (the agent.json shape, plus `persona`), all at once.
    func agentIssues(_ d: JSON, caller: Caller? = nil) -> [Issue] {
        var o = d.objectValue ?? [:]
        let persona = o["persona"]
        o["persona"] = nil
        var out = engine.agentStore.validate(.object(o)) + schemaIssues(d, noun: "agent")
        if let persona, persona.stringValue == nil { out.append(Issue(path: "persona", message: "must be a string")) }
        let cat = engine.catalog
        let hid = o["harness"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 } ?? "chat"
        let hosts = (o["hosts"]?.objectValue ?? [:]).compactMapValues(\.stringValue)
        let inline = hid.contains(" ") || hid.contains("/")
        if !inline {
            if let m = engine.harnessList.first(where: { $0.id == hid }) {
                for (need, level) in m.needs.sorted(by: { $0.key < $1.key }) where level == "required" && (hosts[need] ?? "").isEmpty {
                    let options = cat.hosts(capability: need).filter { cat.status(host: $0.id).usable }.map(\.id)
                    out.append(Issue(code: "missing", path: "hosts.\(need)", message: "harness \(hid) needs a \(need) host",
                                     hint: options.count == 1 ? "use \"\(options[0])\"" : "hosts with \(need): " + ids(options)))
                }
                if let schema = m.config, let cfg = o["config"] {
                    out += SchemaCheck.validate(cfg, schema: schema, path: "config").filter { $0.code != "unknown" }
                    out += SchemaCheck.validate(cfg, schema: schema, path: "config").filter { $0.code == "unknown" }
                }
            } else if let broken = engine.harnessStore.scan().first(where: { $0.id == hid }) {
                out.append(Issue(code: "harness_invalid", path: "harness", message: "harness \(hid) has errors and cannot run",
                                 hint: "fix harnesses/\(hid): " + broken.issues.errors.summary))
            } else {
                out.append(Issue(code: "unknown_harness", path: "harness", message: "no harness \"\(hid)\"", hint: "harnesses: " + ids(engine.harnessList.map(\.id))))
            }
        }
        for (need, ref) in hosts.sorted(by: { $0.key < $1.key }) where !ref.isEmpty {
            guard let h = cat.resolve(ref) else {
                out.append(Issue(code: "unknown_host", path: "hosts.\(need)", message: "no host \"\(ref)\"", hint: "hosts with \(need): " + ids(cat.hosts(capability: need).map(\.id))))
                continue
            }
            if !cat.capabilities(of: h).contains(need) {
                out.append(Issue(path: "hosts.\(need)", message: "host \(h.id) cannot do \(need)", hint: "hosts with \(need): " + ids(cat.hosts(capability: need).map(\.id))))
            } else if let why = cat.problem(host: h.id) {
                out.append(Issue(code: "unavailable", path: "hosts.\(need)", message: why, severity: .warning))
            }
        }
        if let tools = o["tools"]?.arrayValue?.compactMap(\.stringValue) {
            let known = Set(engine.tools.map(\.qualified))
            for (i, p) in tools.enumerated() where p != "*" {
                let prefix = p.hasSuffix(".*") ? String(p.dropLast()) : nil
                let hit = prefix.map { pre in known.contains { $0.hasPrefix(pre) } } ?? known.contains(p)
                if !hit { out.append(Issue(code: "no_match", path: "tools[\(i)]", message: "\"\(p)\" matches no tool right now", hint: "see tools.list", severity: .warning)) }
            }
        }
        if let t = o["tier"]?.stringValue, let tier = Tier(rawValue: t), let c = caller, c.kind == .agent || c.kind == .token, tier > c.tier {
            out.append(Issue(path: "tier", message: "a caller cannot grant a tier above its own (\(c.tier.rawValue))"))
        }
        return out
    }

    func registerDraftVerbs() {
        add("agents.validate", .read, "Check an agent draft (agent.json shape plus persona) and return every problem at once, with paths and hints. Writes nothing.",
            props: [("draft", "object", "The agent: id, name, harness, hosts, tools, tier, config, persona")], required: ["draft"]) { a, c in
            self.issuesJSON(self.agentIssues(a.raw["draft"] ?? .null, caller: c))
        }
        add("agents.try", .act, "Run one turn of a draft agent in a throwaway conversation and return its reply. Nothing is saved. Use it to check a character before agents.create.",
            props: [("draft", "object", "The agent draft, as for agents.validate"), ("message", "string", "What to say to it")], required: ["draft", "message"]) { a, c in
            let d = a.raw["draft"] ?? .null
            let issues = self.agentIssues(d, caller: c)
            if !issues.errors.isEmpty { throw ValidationError(issues: issues) }
            let agent = try self.engine.draftAgent(d)
            let reply = try await self.engine.tryTurn(agent, message: try a.string("message"))
            self.tried.insert(d.pretty)
            return .object(["reply": .string(reply), "warnings": issues.json])
        }
    }
}

// MARK: - engine support

extension ParlorEngine {
    /// An agent from a draft (agent.json shape + `persona`), without saving it.
    func draftAgent(_ d: JSON) throws -> Agent {
        var o = d.objectValue ?? [:]
        let persona = o["persona"]?.stringValue ?? ""
        o["persona"] = nil
        let data = try JSONEncoder().encode(JSON.object(o))
        let rec = try JSONDecoder().decode(AgentRecord.self, from: data)
        return rec.agent(persona: persona)
    }

    /// One turn of an agent that is not saved: a throwaway conversation, one user line, the reply returned.
    func tryTurn(_ agent: Agent, message: String) async throws -> String {
        var a = agent
        a.id = "try-" + agent.id
        let tools = await hub.tools(for: a)
        var input = TurnInput(
            agent: a, conversationID: a.id, participants: [.agent(a.id), .user], transcript: [ChatMessage(role: .user, text: message, speaker: "user")],
            summary: nil, tools: tools, settings: ResolvedSettings.resolve(agent: a, model: nil, machine: settings),
            lastPromptTokens: 0, priority: .user, names: [a.id: a.name], memory: "")
        let launch = self.launch(for: a)
        input.command = launch?.command ?? ""
        input.config = launch?.config ?? a.config
        input.gatewayURL = deps.gatewayURL?() ?? ""
        let box = TryBox()
        defer { externalHarness.stop(agent: a.id) }
        try await externalHarness.run(input, context: TurnContext(llm: deps.llm, tools: HubToolExecutor(hub: hub))) { ev in
            if case .message(let m) = ev, m.role == .assistant, !m.text.isEmpty { box.set(m.text) }
        }
        let text = box.text
        if text.isEmpty { throw CommandError("The draft agent produced no reply.") }
        return text
    }
}

final class TryBox: @unchecked Sendable {
    private let lock = NSLock(); private var t = ""
    func set(_ s: String) { lock.lock(); t = s; lock.unlock() }
    var text: String { lock.lock(); defer { lock.unlock() }; return t }
}

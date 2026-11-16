import Foundation

/// How the `http-openai` adapter reaches one host: a URL, and optionally the command that serves it. Derived from hosts and engines, never written.
public struct OpenAIEndpoint: Codable, Equatable, Sendable, Identifiable {
    public enum Locality: String, Codable, Sendable { case local, remote }

    /// A command the app runs on demand to serve the provider (a llama.cpp server, say).
    public struct Service: Codable, Equatable, Sendable {
        public var command: String
        /// Stop after this many idle minutes. 0 keeps it running.
        public var idleMinutes: Int = 30
        /// How long to wait for the server to answer `GET <baseURL>/models` after launch.
        public var startupSeconds: Int = 180
        public var env: [String: String]?
        public init(command: String, idleMinutes: Int = 30, startupSeconds: Int = 180, env: [String: String]? = nil) {
            self.command = command; self.idleMinutes = idleMinutes; self.startupSeconds = startupSeconds; self.env = env
        }
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            command = try c.decode(String.self, forKey: .command)
            idleMinutes = try c.decodeIfPresent(Int.self, forKey: .idleMinutes) ?? 30
            startupSeconds = try c.decodeIfPresent(Int.self, forKey: .startupSeconds) ?? 180
            env = try c.decodeIfPresent([String: String].self, forKey: .env)
        }
    }

    public var id: String
    /// `openai-compatible` (the built-in `mlx` provider is not listed in settings).
    public var kind: String = "openai-compatible"
    public var baseURL: String
    /// Name of the Keychain item holding the API key. Never the key itself.
    public var keyRef: String?
    /// Model names to offer; nil asks the server (`GET /models`).
    public var models: [String]?
    public var locality: Locality = .local
    /// Remote providers start off: the user turns them on knowing prompts leave the machine.
    public var enabled: Bool
    public var service: Service?

    public init(id: String, baseURL: String, kind: String = "openai-compatible", keyRef: String? = nil, models: [String]? = nil,
                locality: Locality = .local, enabled: Bool? = nil, service: Service? = nil) {
        self.id = id; self.baseURL = baseURL; self.kind = kind; self.keyRef = keyRef; self.models = models
        self.locality = locality; self.enabled = enabled ?? (locality == .local); self.service = service
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        kind = try c.decodeIfPresent(String.self, forKey: .kind) ?? "openai-compatible"
        baseURL = try c.decodeIfPresent(String.self, forKey: .baseURL) ?? ""
        keyRef = try c.decodeIfPresent(String.self, forKey: .keyRef)
        models = try c.decodeIfPresent([String].self, forKey: .models)
        locality = try c.decodeIfPresent(Locality.self, forKey: .locality) ?? .local
        enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? (locality == .local)
        service = try c.decodeIfPresent(Service.self, forKey: .service)
    }

    /// Why this entry cannot work, if so.
    public var problem: String? {
        if id.isEmpty || id == "mlx" || id.contains("/") || id.contains(":") { return "id must be a short name without / or : (and not \"mlx\")" }
        if kind != "openai-compatible" { return "kind must be openai-compatible" }
        if URL(string: baseURL)?.host == nil { return "baseURL is not a URL" }
        return nil
    }
}


// MARK: - the provider

/// A server that speaks OpenAI chat completions: llama.cpp's `llama-server`, LM Studio, Ollama, a hosted API.
public final class OpenAIClient: LanguageModelRuntime, @unchecked Sendable {
    public let config: OpenAIEndpoint
    public let process: HostProcess?
    private let session: URLSession

    public init(config: OpenAIEndpoint, home: ParlorHome) {
        self.config = config
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 900     // long prompts take a while before the first byte
        cfg.timeoutIntervalForResource = 24 * 3600
        self.session = URLSession(configuration: cfg)
        self.process = config.service.map { HostProcess(config: config, service: $0, home: home) }
    }

    private var base: String { config.baseURL.hasSuffix("/") ? String(config.baseURL.dropLast()) : config.baseURL }

    private func request(_ path: String, method: String = "GET", body: Data? = nil) -> URLRequest {
        var r = URLRequest(url: URL(string: base + path)!)
        r.httpMethod = method
        if let body { r.httpBody = body; r.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        if let ref = config.keyRef, let key = KeyStore.get(ref) { r.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
        return r
    }

    public func loadedModel() async -> String? { nil }
    public func unload() async { process?.stop() }

    /// The models this provider serves: the configured list, else `GET /models`.
    public func models() async throws -> [EndpointModel] {
        if let m = config.models, !m.isEmpty { return m.map { EndpointModel(provider: config.id, id: $0) } }
        try await process?.ensureRunning()
        process?.touch()
        let (data, resp) = try await session.data(for: request("/models"))
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else { throw ParlorError.message("\(config.id): GET /models answered \((resp as? HTTPURLResponse)?.statusCode ?? 0)") }
        guard let list = try? JSONDecoder().decode(JSON.self, from: data).objectValue?["data"],
              case .array(let items) = list else { throw ParlorError.message("\(config.id): /models did not return a list") }
        return items.compactMap { $0.objectValue?["id"]?.stringValue }.map { EndpointModel(provider: config.id, id: $0) }
    }

    /// Starts the server if it has one and proves it answers `GET /models`. Returns the models it lists.
    public func check() async throws -> [EndpointModel] {
        try await process?.ensureRunning()
        process?.touch()
        let (data, resp) = try await session.data(for: request("/models"))
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw ParlorError.message("\(config.id): GET /models answered \(code)") }
        let listed = (try? JSONDecoder().decode(JSON.self, from: data))?.objectValue?["data"]
        var ids: [String] = []
        if case .array(let items)? = listed { ids = items.compactMap { $0.objectValue?["id"]?.stringValue } }
        if let m = config.models, !m.isEmpty { ids = m }
        return ids.map { EndpointModel(provider: config.id, id: $0) }
    }

    public func generate(model: String, request req: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
        AsyncThrowingStream { cont in
            let task = Task {
                process?.begin()
                defer { process?.end() }
                do {
                    try await process?.ensureRunning()
                    var body: [String: JSON] = [
                        "model": .string(model), "stream": .bool(true),
                        "stream_options": .object(["include_usage": .bool(true)]),
                        "messages": .array(OpenAIWire.messages(req.messages)),
                        "temperature": .double(req.temperature), "top_p": .double(req.tuning.settings.topP)]
                    if let m = req.maxTokens { body["max_tokens"] = .int(m) }
                    for (k, v) in req.sampling { body[k] = v }
                    if let rp = req.sampling["repetition_penalty"] { body["repeat_penalty"] = rp }   // llama-server's name for it
                    if !req.tools.isEmpty { body["tools"] = .array(OpenAIWire.toolSpecs(req.tools)) }
                    if config.locality == .local { body["chat_template_kwargs"] = .object(["enable_thinking": .bool(req.thinking)]) }
                    let started = Date()
                    let (bytes, resp) = try await session.bytes(for: request("/chat/completions", method: "POST", body: OpenAIWire.encode(.object(body))))
                    guard let http = resp as? HTTPURLResponse else { throw ParlorError.message("\(config.id): no response") }
                    guard http.statusCode == 200 else {
                        var text = ""
                        for try await line in bytes.lines { text += line; if text.count > 600 { break } }
                        let msg = JSON.parse(text)?.objectValue?["error"]?.objectValue?["message"]?.stringValue ?? text
                        throw ParlorError.message("\(config.id) answered \(http.statusCode): \(msg.isEmpty ? "(no body)" : msg)")
                    }
                    var calls: [Int: (id: String, name: String, args: String)] = [:]
                    var completion = 0, prompt = 0, cached = 0
                    var tps: Double?
                    var deltas = 0
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data:") else { continue }
                        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                        if payload == "[DONE]" { break }
                        guard let o = JSON.parse(payload)?.objectValue else { continue }
                        if let err = o["error"]?.objectValue { throw ParlorError.message("\(config.id): \(err["message"]?.stringValue ?? "error")") }
                        if let choice = o["choices"].flatMap({ v -> JSON? in if case .array(let a) = v { return a.first } else { return nil } })?.objectValue,
                           let delta = choice["delta"]?.objectValue {
                            if let t = delta["content"]?.stringValue, !t.isEmpty { deltas += 1; cont.yield(.text(t)) }
                            if case .array(let tcs)? = delta["tool_calls"] {
                                for tc in tcs {
                                    guard let t = tc.objectValue else { continue }
                                    let i = t["index"]?.intValue ?? 0
                                    var cur = calls[i] ?? ("", "", "")
                                    if let id = t["id"]?.stringValue, !id.isEmpty { cur.id = id }
                                    if let f = t["function"]?.objectValue {
                                        if let n = f["name"]?.stringValue { cur.name += n }
                                        if let a = f["arguments"]?.stringValue { cur.args += a }
                                    }
                                    calls[i] = cur
                                }
                            }
                        }
                        if let u = o["usage"]?.objectValue {
                            prompt = u["prompt_tokens"]?.intValue ?? prompt
                            completion = u["completion_tokens"]?.intValue ?? completion
                            cached = u["prompt_tokens_details"]?.objectValue?["cached_tokens"]?.intValue ?? cached
                        }
                        if let t = o["timings"]?.objectValue {
                            if case .double(let d)? = t["predicted_per_second"] { tps = d }
                            if let c = t["cache_n"]?.intValue { cached = c }
                            if prompt == 0, let n = t["prompt_n"]?.intValue { prompt = n + cached }
                            if completion == 0, let n = t["predicted_n"]?.intValue { completion = n }
                        }
                    }
                    for i in calls.keys.sorted() {
                        let c = calls[i]!
                        guard !c.name.isEmpty else { continue }
                        cont.yield(.toolCall(ToolCallRecord(id: c.id.isEmpty ? UUID().uuidString : c.id, name: c.name,
                                                            arguments: JSON.parse(c.args) ?? .object([:]))))
                    }
                    let secs = max(0.001, Date().timeIntervalSince(started))
                    if completion == 0 { completion = deltas }
                    cont.yield(.stats(LLMStats(promptTokens: prompt, completionTokens: completion,
                                               tokensPerSecond: tps ?? Double(completion) / secs, cachedTokens: cached)))
                    cont.finish()
                } catch is CancellationError {
                    cont.finish(throwing: CancellationError())
                } catch let e as URLError where e.code == .cancelled {
                    cont.finish(throwing: CancellationError())
                } catch {
                    cont.finish(throwing: error)
                }
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }
}

// MARK: - a provider's server process

/// Runs a provider's `service` command on demand, waits until it answers, stops it when idle.
public final class HostProcess: @unchecked Sendable {
    public let config: OpenAIEndpoint
    let service: OpenAIEndpoint.Service
    let home: ParlorHome
    private let lock = NSLock()
    private var proc: ManagedProcess?
    private var starting: Task<Void, Error>?
    private var _state: ServiceState = .stopped
    private var _lastUsed: Date?
    private var active = 0
    private var idleTask: Task<Void, Never>?
    private var external = false
    private var _restarts = 0
    /// Tests shorten the idle window below one minute.
    var idleSecondsOverride: TimeInterval?

    public var serviceID: String { "host:" + config.id }

    init(config: OpenAIEndpoint, service: OpenAIEndpoint.Service, home: ParlorHome) {
        self.config = config; self.service = service; self.home = home
    }

    public struct Snapshot: Sendable { public var state: ServiceState; public var pid: Int32?; public var lastUsed: Date?; public var external: Bool; public var restarts: Int; public var active: Int }
    public func snapshot() -> Snapshot {
        lock.lock(); defer { lock.unlock() }
        return Snapshot(state: _state, pid: proc?.pid, lastUsed: _lastUsed, external: external, restarts: _restarts, active: active)
    }

    func touch() { lock.lock(); _lastUsed = Date(); lock.unlock() }
    func begin() { lock.lock(); active += 1; _lastUsed = Date(); lock.unlock() }
    func end() { lock.lock(); active -= 1; _lastUsed = Date(); lock.unlock(); scheduleIdle() }

    private var healthURL: URL { URL(string: (config.baseURL.hasSuffix("/") ? String(config.baseURL.dropLast()) : config.baseURL) + "/models")! }

    private func answers() async -> Bool {
        var r = URLRequest(url: healthURL); r.timeoutInterval = 1.5
        if let ref = config.keyRef, let key = KeyStore.get(ref) { r.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
        guard let (_, resp) = try? await URLSession.shared.data(for: r) else { return false }
        return (resp as? HTTPURLResponse)?.statusCode == 200
    }

    public func ensureRunning() async throws {
        lock.lock()
        if _state == .running || _state == .idle, proc?.isRunning == true || external { _lastUsed = Date(); lock.unlock(); return }
        if let s = starting { lock.unlock(); try await s.value; return }
        let t = Task { try await self.launch() }
        starting = t
        if case .failed = _state { _restarts += 1 }
        _state = .starting
        lock.unlock()
        do { try await t.value } catch { lock.lock(); starting = nil; lock.unlock(); throw error }
        lock.lock(); starting = nil; lock.unlock()
    }

    private func launch() async throws {
        // Something already answers at the base URL (the user started it themselves): use it, don't own it.
        if await answers() {
            lock.lock(); external = true; _state = .running; _lastUsed = Date(); lock.unlock()
            return
        }
        let log = ServiceLog.open(home: home, service: serviceID, banner: "starting: \(service.command)")
        let launch = Launch.shell(service.command, environment: service.env ?? [:], cwd: home.root)
        let p: ManagedProcess
        do {
            p = try Supervisor.spawn(launch, streams: Streams(stdout: log, stderr: log)) { [weak self] proc, status in
                guard let self else { return }
                self.lock.lock()
                let wasStopping = self._state == .stopped
                if self.proc === proc {
                    self.proc = nil
                    if !wasStopping { self._state = .failed("exited with status \(status)") }
                }
                self.lock.unlock()
                try? log?.write(contentsOf: Data("--- exited \(status)\n".utf8))
            }
        } catch {
            lock.lock(); _state = .failed(error.localizedDescription); lock.unlock()
            throw ParlorError.message("Could not start \(config.id): \(error.localizedDescription)")
        }
        lock.lock(); proc = p; external = false; lock.unlock()
        let deadline = Date().addingTimeInterval(TimeInterval(max(5, service.startupSeconds)))
        while Date() < deadline {
            if !p.isRunning {
                let tail = ServiceLog.tail(home: home, service: serviceID, lines: 6)
                lock.lock(); _state = .failed("exited with status \(p.terminationStatus)"); lock.unlock()
                throw ParlorError.message("\(config.id) exited while starting (status \(p.terminationStatus)).\n\(tail)")
            }
            if await answers() {
                lock.lock(); _state = .running; _lastUsed = Date(); lock.unlock()
                scheduleIdle()
                return
            }
            try await Task.sleep(nanoseconds: 400_000_000)
        }
        stop()
        throw ParlorError.message("\(config.id) did not answer within \(service.startupSeconds)s. See logs/host-\(config.id).log")
    }

    private func scheduleIdle() {
        guard service.idleMinutes > 0 || idleSecondsOverride != nil else { return }
        lock.lock(); idleTask?.cancel()
        let limit = idleSecondsOverride ?? TimeInterval(service.idleMinutes * 60)
        idleTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(min(limit, 30) * 1_000_000_000))
                guard let self, !Task.isCancelled else { return }
                self.lock.lock()
                let idle = self.active == 0 && Date().timeIntervalSince(self._lastUsed ?? .distantPast) >= limit
                self.lock.unlock()
                if idle { self.stop(); return }
            }
        }
        lock.unlock()
    }

    /// Stop the server, if the app started it.
    public func stop() {
        lock.lock()
        idleTask?.cancel(); idleTask = nil
        let p = proc
        let wasExternal = external
        _state = .stopped; external = false
        lock.unlock()
        guard !wasExternal, let p, p.isRunning else { return }
        p.terminate(grace: 3)
    }
}


// MARK: - the adapter

/// `http-openai`: a server that speaks OpenAI chat completions. Local hosts get a port and the command from their
/// engine manifest; remote hosts use `url`, `keyRef` and `models`. One `OpenAIClient` per host, rebuilt from the
/// catalog on every reload; a client whose endpoint did not change keeps its process.
public final class OpenAIAdapter: Adapter, @unchecked Sendable {
    public let id = "http-openai"
    let home: ParlorHome
    private let lock = NSLock()
    private var clients: [String: OpenAIClient] = [:]

    init(home: ParlorHome) { self.home = home }

    public var all: [OpenAIClient] { lock.lock(); defer { lock.unlock() }; return clients.values.sorted { $0.config.id < $1.config.id } }
    public func client(_ host: String) -> OpenAIClient? { lock.lock(); defer { lock.unlock() }; return clients[host] }

    public func reconfigure(catalog: Catalog) {
        let configs = Self.endpoints(catalog)
        lock.lock(); defer { lock.unlock() }
        var next: [String: OpenAIClient] = [:]
        for c in configs where c.problem == nil {
            if let old = clients[c.id], old.config == c { next[c.id] = old } else { next[c.id] = OpenAIClient(config: c, home: home) }
        }
        for (id, old) in clients where next[id] !== old { old.process?.stop() }
        clients = next
    }

    /// Endpoints for hosts on an http-openai engine, computed on every load and never written.
    static func endpoints(_ cat: Catalog) -> [OpenAIEndpoint] {
        var out: [OpenAIEndpoint] = []
        var used: Set<Int> = [11500]
        for h in cat.hosts.sorted(by: { $0.id < $1.id }) {
            guard let e = cat.engine(h.spec.engine), e.adapter == "http-openai" else { continue }
            if e.remote {
                guard let url = h.spec.url, !url.isEmpty else { continue }
                out.append(OpenAIEndpoint(id: h.id, baseURL: url, keyRef: h.spec.keyRef, models: h.spec.models, locality: .remote, enabled: h.spec.enabled))
                continue
            }
            guard h.spec.enabled, let launch = e.launch, let repo = h.spec.model, let m = cat.model(repo),
                  let bin = cat.binary(for: e) ?? EngineBinary.find(launch.command, engineDir: e.directory) else { continue }
            var port = 8100 + Int(stableHash(h.id) % 800)
            while used.contains(port) { port += 1 }
            used.insert(port)
            var values: [String: JSON] = ["host": .string(h.id), "port": .int(port), "model.dir": .string(m.directory.path)]
            if let f = m.file { values["model.file"] = .string(m.directory.appendingPathComponent(f).path) }
            for (k, v) in SchemaCheck.defaults(e.settings) { values["settings." + k] = v }
            for (k, v) in h.spec.settings { values["settings." + k] = v }
            let (args, missing) = ArgTemplate.expand(launch.args, values: values)
            guard missing.isEmpty else { continue }
            let cmd = ([bin.path] + args).map { "'" + $0.replacingOccurrences(of: "'", with: "'\\''") + "'" }.joined(separator: " ")
            out.append(OpenAIEndpoint(id: h.id, baseURL: "http://127.0.0.1:\(port)/v1", models: [h.id],
                                      service: .init(command: cmd, idleMinutes: e.lifecycle.idleMinutes, startupSeconds: e.lifecycle.startupSeconds)))
        }
        return out
    }

    static func stableHash(_ s: String) -> UInt64 {
        var h: UInt64 = 14695981039346656037
        for b in s.utf8 { h = (h ^ UInt64(b)) &* 1099511628211 }
        return h
    }

    public func chat(_ req: LLMRequest, host: HostInfo, engine: EngineManifest, model: String?) -> AsyncThrowingStream<LLMEvent, Error> {
        guard let c = client(host.id) else {
            return AsyncThrowingStream { $0.finish(throwing: ParlorError.message("Host \(host.id) has no running setup yet (check its engine, model and settings in Library).")) }
        }
        guard c.config.enabled else {
            return AsyncThrowingStream { $0.finish(throwing: ParlorError.message("Host \(host.id) is off. Turn it on in Library first.")) }
        }
        return c.generate(model: model ?? host.id, request: req)
    }

    public func state(of host: HostInfo) -> ServiceState? { client(host.id)?.process?.snapshot().state }

    public func usage(of host: HostInfo) async -> HostUsage {
        guard let s = client(host.id)?.process?.snapshot() else { return HostUsage(alive: false) }
        let bytes = s.pid.flatMap { ProcessMemory.footprint(pid: $0) }.map { Int64($0) } ?? 0
        return HostUsage(alive: s.state.isAlive, active: s.active, lastUsed: s.lastUsed, bytes: bytes)
    }

    public func residentBytes() async -> Int64 {
        var t: Int64 = 0
        for c in all { if let s = c.process?.snapshot(), s.state.isAlive, let pid = s.pid, let b = ProcessMemory.footprint(pid: pid) { t += Int64(b) } }
        return t
    }

    public func start(_ host: HostInfo) async throws {
        guard let c = client(host.id) else { throw ParlorError.message("Host \(host.id) is not set up to start.") }
        try await c.process?.ensureRunning()
    }

    public func stop(_ host: HostInfo) async { client(host.id)?.process?.stop() }
    public func shutdown() async { for c in all { c.process?.stop() } }
}

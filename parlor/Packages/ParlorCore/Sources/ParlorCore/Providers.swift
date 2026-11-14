import Foundation
import Security

/// One entry of `settings.json → providers` (SPEC §5).
public struct ProviderConfig: Codable, Equatable, Sendable, Identifiable {
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

public struct ProviderModel: Sendable, Equatable {
    public var provider: String
    public var id: String
    /// What an agent file's `model:` holds.
    public var qualified: String { "\(provider)/\(id)" }
}

/// API keys live in the Keychain; `PARLOR_KEY_<NAME>` in the environment wins, for scripts and tests.
public enum KeyStore {
    static func envName(_ ref: String) -> String {
        "PARLOR_KEY_" + String(ref.uppercased().map { $0.isLetter || $0.isNumber ? $0 : "_" })
    }

    public static func get(_ ref: String) -> String? {
        if let v = ProcessInfo.processInfo.environment[envName(ref)], !v.isEmpty { return v }
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "parlor",
                                kSecAttrAccount as String: ref, kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        var out: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data else { return nil }
        return String(data: d, encoding: .utf8)
    }

    public static func has(_ ref: String) -> Bool { get(ref) != nil }

    @discardableResult
    public static func set(_ ref: String, _ value: String) -> Bool {
        let base: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "parlor", kSecAttrAccount as String: ref]
        SecItemDelete(base as CFDictionary)
        var add = base
        add[kSecValueData as String] = Data(value.utf8)
        return SecItemAdd(add as CFDictionary, nil) == errSecSuccess
    }

    public static func remove(_ ref: String) {
        SecItemDelete([kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "parlor", kSecAttrAccount as String: ref] as CFDictionary)
    }
}

// MARK: - OpenAI wire format

/// Translation between parlor's request types and OpenAI chat-completions JSON, in both directions:
/// the provider client writes requests and reads chunks, the runtime API reads requests and writes chunks.
enum OpenAIWire {
    /// Tool names reach OpenAI-style callers exactly as they wrote them; inside parlor they carry this server prefix.
    static let externalServer = "ext"

    static func toolSpecs(_ tools: [ToolDef]) -> [JSON] {
        tools.map { t in
            .object(["type": .string("function"), "function": .object([
                "name": .string(t.wireName), "description": .string(t.description), "parameters": t.schema])])
        }
    }

    static func messages(_ msgs: [LLMMessage]) -> [JSON] {
        msgs.map { m in
            switch m.role {
            case .system: return .object(["role": .string("system"), "content": .string(m.text)])
            case .user: return .object(["role": .string("user"), "content": .string(m.text)])
            case .assistant:
                var o: [String: JSON] = ["role": .string("assistant"), "content": m.text.isEmpty && !m.toolCalls.isEmpty ? .null : .string(m.text)]
                if !m.toolCalls.isEmpty {
                    o["tool_calls"] = .array(m.toolCalls.map { c in
                        .object(["id": .string(c.id), "type": .string("function"),
                                 "function": .object(["name": .string(c.name), "arguments": .string(c.arguments.text)])])
                    })
                }
                return .object(o)
            case .tool:
                return .object(["role": .string("tool"), "tool_call_id": .string(m.toolCallID ?? ""), "content": .string(m.text)])
            }
        }
    }

    // Server side: an OpenAI request into parlor's types.

    static func parseTools(_ v: JSON?) -> [ToolDef] {
        guard case .array(let a)? = v else { return [] }
        return a.compactMap { t in
            guard let f = t.objectValue?["function"]?.objectValue, let name = f["name"]?.stringValue else { return nil }
            return ToolDef(server: externalServer, name: name, description: f["description"]?.stringValue ?? "",
                           schema: f["parameters"] ?? .object(["type": .string("object"), "properties": .object([:])]))
        }
    }

    static func text(of content: JSON?) -> String {
        switch content {
        case .string(let s)?: return s
        case .array(let parts)?:
            return parts.compactMap { p -> String? in
                guard let o = p.objectValue, o["type"]?.stringValue == "text" || o["type"] == nil else { return nil }
                return o["text"]?.stringValue
            }.joined()
        default: return ""
        }
    }

    static func parseMessages(_ v: JSON?) -> [LLMMessage]? {
        guard case .array(let a)? = v else { return nil }
        var names: [String: String] = [:]
        var out: [LLMMessage] = []
        for item in a {
            guard let o = item.objectValue, let role = o["role"]?.stringValue else { return nil }
            switch role {
            case "system", "developer": out.append(LLMMessage(role: .system, text: text(of: o["content"])))
            case "user": out.append(LLMMessage(role: .user, text: text(of: o["content"])))
            case "assistant":
                var calls: [ToolCallRecord] = []
                if case .array(let tcs)? = o["tool_calls"] {
                    for tc in tcs {
                        guard let t = tc.objectValue, let f = t["function"]?.objectValue, let n = f["name"]?.stringValue else { continue }
                        let args: JSON
                        switch f["arguments"] {
                        case .string(let s)?: args = JSON.parse(s) ?? .object([:])
                        case let j?: args = j
                        default: args = .object([:])
                        }
                        let id = t["id"]?.stringValue ?? UUID().uuidString
                        names[id] = n
                        calls.append(ToolCallRecord(id: id, name: externalServer + "__" + n, arguments: args))
                    }
                }
                out.append(LLMMessage(role: .assistant, text: text(of: o["content"]), toolCalls: calls))
            case "tool":
                let id = o["tool_call_id"]?.stringValue
                out.append(LLMMessage(role: .tool, text: text(of: o["content"]), toolCallID: id,
                                      toolName: (o["name"]?.stringValue ?? id.flatMap { names[$0] }).map { externalServer + "__" + $0 }))
            default: return nil
            }
        }
        return out
    }

    /// The name an OpenAI-style caller gave the tool.
    static func externalName(_ wire: String) -> String {
        let p = externalServer + "__"
        return wire.hasPrefix(p) ? String(wire.dropFirst(p.count)) : wire
    }

    static func encode(_ j: JSON) -> Data { (try? JSONEncoder().encode(j)) ?? Data() }
}

// MARK: - the provider

/// A server that speaks OpenAI chat completions: llama.cpp's `llama-server`, LM Studio, Ollama, a hosted API.
public final class OpenAIProvider: LanguageModelRuntime, @unchecked Sendable {
    public let config: ProviderConfig
    public let process: ProviderProcess?
    private let session: URLSession

    public init(config: ProviderConfig, home: ParlorHome) {
        self.config = config
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 900     // long prompts take a while before the first byte
        cfg.timeoutIntervalForResource = 24 * 3600
        self.session = URLSession(configuration: cfg)
        self.process = config.service.map { ProviderProcess(config: config, service: $0, home: home) }
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
    public func models() async throws -> [ProviderModel] {
        if let m = config.models, !m.isEmpty { return m.map { ProviderModel(provider: config.id, id: $0) } }
        try await process?.ensureRunning()
        process?.touch()
        let (data, resp) = try await session.data(for: request("/models"))
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else { throw ParlorError.message("\(config.id): GET /models answered \((resp as? HTTPURLResponse)?.statusCode ?? 0)") }
        guard let list = try? JSONDecoder().decode(JSON.self, from: data).objectValue?["data"],
              case .array(let items) = list else { throw ParlorError.message("\(config.id): /models did not return a list") }
        return items.compactMap { $0.objectValue?["id"]?.stringValue }.map { ProviderModel(provider: config.id, id: $0) }
    }

    /// Starts the server if it has one and proves it answers `GET /models`. Returns the models it lists.
    public func check() async throws -> [ProviderModel] {
        try await process?.ensureRunning()
        process?.touch()
        let (data, resp) = try await session.data(for: request("/models"))
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw ParlorError.message("\(config.id): GET /models answered \(code)") }
        let listed = (try? JSONDecoder().decode(JSON.self, from: data))?.objectValue?["data"]
        var ids: [String] = []
        if case .array(let items)? = listed { ids = items.compactMap { $0.objectValue?["id"]?.stringValue } }
        if let m = config.models, !m.isEmpty { ids = m }
        return ids.map { ProviderModel(provider: config.id, id: $0) }
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
public final class ProviderProcess: @unchecked Sendable {
    public let config: ProviderConfig
    let service: ProviderConfig.Service
    let home: ParlorHome
    private let lock = NSLock()
    private var proc: Process?
    private var starting: Task<Void, Error>?
    private var _state: ServiceState = .stopped
    private var _lastUsed: Date?
    private var active = 0
    private var idleTask: Task<Void, Never>?
    private var external = false
    /// Tests shorten the idle window below one minute.
    var idleSecondsOverride: TimeInterval?

    public var serviceID: String { "provider:" + config.id }

    init(config: ProviderConfig, service: ProviderConfig.Service, home: ParlorHome) {
        self.config = config; self.service = service; self.home = home
    }

    public struct Snapshot: Sendable { public var state: ServiceState; public var pid: Int32?; public var lastUsed: Date?; public var external: Bool }
    public func snapshot() -> Snapshot {
        lock.lock(); defer { lock.unlock() }
        return Snapshot(state: _state, pid: proc?.isRunning == true ? proc?.processIdentifier : nil, lastUsed: _lastUsed, external: external)
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
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/zsh")
        p.arguments = ["-l", "-c", "exec " + service.command]
        var env = ProcessInfo.processInfo.environment
        for (k, v) in service.env ?? [:] { env[k] = v }
        p.environment = env
        p.currentDirectoryURL = home.root
        p.standardOutput = log; p.standardError = log
        p.standardInput = FileHandle.nullDevice
        p.terminationHandler = { [weak self] proc in
            guard let self else { return }
            self.lock.lock()
            let wasStopping = self._state == .stopped
            if self.proc === proc {
                self.proc = nil
                if !wasStopping { self._state = .failed("exited with status \(proc.terminationStatus)") }
            }
            self.lock.unlock()
            try? log?.write(contentsOf: Data("--- exited \(proc.terminationStatus)\n".utf8))
        }
        do { try p.run() } catch {
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
        throw ParlorError.message("\(config.id) did not answer within \(service.startupSeconds)s. See logs/provider-\(config.id).log")
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
        p.terminate()
        DispatchQueue.global().asyncAfter(deadline: .now() + 3) { if p.isRunning { kill(p.processIdentifier, SIGKILL) } }
    }
}

// MARK: - routing

/// Sends a model to the right runtime. `provider/model` goes to that provider; anything else is MLX
/// (`mlx/<repo>` also works, so `/v1/models` can list every model as `provider/repo`).
public final class ProviderRouter: LanguageModelRuntime, @unchecked Sendable {
    public let inner: LanguageModelRuntime
    private let lock = NSLock()
    private var providers: [String: OpenAIProvider] = [:]

    public init(inner: LanguageModelRuntime) { self.inner = inner }

    /// Apply the registry. A provider whose entry did not change keeps its process.
    public func configure(_ configs: [ProviderConfig], home: ParlorHome) {
        lock.lock(); defer { lock.unlock() }
        var next: [String: OpenAIProvider] = [:]
        for c in configs where c.problem == nil {
            if let old = providers[c.id], old.config == c { next[c.id] = old } else { next[c.id] = OpenAIProvider(config: c, home: home) }
        }
        for (id, old) in providers where next[id] !== old { old.process?.stop() }
        providers = next
    }

    public var all: [OpenAIProvider] { lock.lock(); defer { lock.unlock() }; return providers.values.sorted { $0.config.id < $1.config.id } }
    public func provider(_ id: String) -> OpenAIProvider? { lock.lock(); defer { lock.unlock() }; return providers[id] }

    /// The provider a qualified model name belongs to, and the model name it understands.
    public func route(_ model: String) -> (provider: OpenAIProvider, model: String)? {
        guard let i = model.firstIndex(of: "/") else { return nil }
        guard let p = provider(String(model[..<i])) else { return nil }
        return (p, String(model[model.index(after: i)...]))
    }

    public func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
        if let (p, name) = route(model) {
            guard p.config.enabled else {
                return AsyncThrowingStream { $0.finish(throwing: ParlorError.message("Provider \(p.config.id) is off. Turn it on in System first.")) }
            }
            return p.generate(model: name, request: request)
        }
        if model.hasPrefix("mlx/") { return inner.generate(model: String(model.dropFirst(4)), request: request) }
        return inner.generate(model: model, request: request)
    }

    public func loadedModel() async -> String? { await inner.loadedModel() }
    public func unload() async { await inner.unload() }
    public func stopAll() { for p in all { p.process?.stop() } }
}

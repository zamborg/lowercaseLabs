import Foundation

/// What to draw.
public struct ImageRequest: Sendable {
    public var prompt: String
    public var negative: String?
    /// `square`, `portrait` or `landscape`; ignored when `width` and `height` are given.
    public var shape = "square"
    public var detailed = false
    public var seed: Int?
    public var width: Int?
    public var height: Int?
    public var steps: Int?
    public init(prompt: String, negative: String? = nil, shape: String = "square", detailed: Bool = false, seed: Int? = nil,
                width: Int? = nil, height: Int? = nil, steps: Int? = nil) {
        self.prompt = prompt; self.negative = negative; self.shape = shape; self.detailed = detailed; self.seed = seed
        self.width = width; self.height = height; self.steps = steps
    }
}

public struct ImageResult: Sendable {
    /// Home-relative path under `media/YYYY-MM/`.
    public var path: String
    public var url: URL
    public var width: Int
    public var height: Int
    public var seed: Int
    /// The prompt as drawn (tag prefixes included).
    public var prompt: String
    public var host: String
    public var seconds: Double
}

/// The image side of the gateway, and the `image.generate` tool built on it. A host picks an engine, and the engine's
/// adapter decides how the picture is made: `mcp-stdio` keeps a worker process (Z-Image), `command` runs one process
/// per image (stable-diffusion.cpp). Everything about availability comes from the `Catalog`.
public actor ImageProvider: ToolProvider {
    public nonisolated let id = "image"
    private let home: ParlorHome
    private let library: ModelLibrary
    private let catalog: @Sendable () async -> Catalog
    /// The image host an agent draws with (nil agent: the default). Set by the engine.
    private let hostFor: @Sendable (String?) async -> String?
    private var cfg = ParlorSettings.Image()

    private struct Worker {
        var conn: MCPServerConnection
        var key: String
        var model: String
        var idleTask: Task<Void, Never>?
        var active = 0
    }
    private var workers: [String: Worker] = [:]
    private var starting: Set<String> = []
    private var commandsRunning = 0
    private var runningModel: String?
    public private(set) var lastUsed: Date?
    public private(set) var lastError: String?

    public init(home: ParlorHome, library: ModelLibrary,
                catalog: @escaping @Sendable () async -> Catalog,
                hostFor: @escaping @Sendable (String?) async -> String?) {
        self.home = home; self.library = library; self.catalog = catalog; self.hostFor = hostFor
    }

    /// Apply the image knobs. They are the base for a Z-Image host's settings until Tuning owns them.
    public func configure(_ new: ParlorSettings.Image) {
        let precisionChanged = new.precision != cfg.precision
        cfg = new
        if precisionChanged { for id in Array(workers.keys) { stopWorker(id) } }
    }

    // MARK: status

    public func stateDescription() -> String {
        if let m = workers.values.first?.model ?? runningModel { return "loaded · \(m.split(separator: "/").last.map(String.init) ?? m)" }
        return "idle"
    }

    public func serviceSnapshot() async -> (state: ServiceState, pid: Int32?, model: String?, lastUsed: Date?) {
        let w = workers.values.first
        let pid = await w?.conn.pid
        let state: ServiceState
        if !starting.isEmpty { state = .starting }
        else if w != nil || commandsRunning > 0 { state = (commandsRunning > 0 || (w?.active ?? 0) > 0) ? .running : .idle }
        else if let e = lastError { state = .failed(e) }
        else { state = .stopped }
        return (state, pid, w?.model ?? runningModel, lastUsed)
    }

    /// Start the default host's worker without a request, so the model is loaded before the first image.
    public func start() async throws {
        let cat = await catalog()
        guard let id = await hostFor(nil), let host = cat.host(id), let e = cat.engine(host.spec.engine) else {
            throw ParlorError.message("No image model is downloaded. Download one in Models.")
        }
        if e.adapter == "mcp-stdio" { _ = try await ensureWorker(host: host, engine: e, cat: cat) }
    }

    public func stop() { for id in Array(workers.keys) { stopWorker(id) } }
    public func shutdown() { stop() }

    /// Stop one host's worker (the gateway does this to make room).
    public func stop(host id: String) { stopWorker(id) }
    public func hasWorker(_ id: String) -> Bool { workers[id] != nil }
    /// A host's worker as the budget sees it: nil when none runs.
    public func workerUsage(host id: String) async -> (active: Int, lastUsed: Date?, bytes: Int64)? {
        guard let w = workers[id] else { return nil }
        let pid = await w.conn.pid
        return (w.active, lastUsed, Int64(pid.flatMap { ProcessMemory.footprint(pid: $0) } ?? 0))
    }
    public func idleWorkers() -> [String] { workers.filter { $0.value.active == 0 }.map(\.key) }
    public func workerPIDs() async -> [Int32] {
        var out: [Int32] = []
        for w in workers.values { if let p = await w.conn.pid { out.append(p) } }
        return out
    }

    // MARK: generating

    /// Draw one image on a host. Throws a readable problem when the host cannot serve.
    public func generate(host id: String, _ req: ImageRequest) async throws -> ImageResult {
        let cat = await catalog()
        guard let host = cat.host(id) else {
            throw ParlorError.message("No image host \"\(id)\". Hosts: " + cat.hosts(capability: "image").map(\.id).joined(separator: ", "))
        }
        guard let engine = cat.engine(host.spec.engine), engine.capabilities.contains("image") else {
            throw ParlorError.message("Host \(id) does not draw images.")
        }
        if let why = cat.problem(host: id) { throw ParlorError.message(why) }
        let started = Date()
        let seed = req.seed ?? Int.random(in: 0..<100_000)
        defer { lastUsed = Date() }
        do {
            switch engine.adapter {
            case "command": return try await runCommand(host: host, engine: engine, cat: cat, req: req, seed: seed, started: started)
            case "mcp-stdio": return try await runWorker(host: host, engine: engine, cat: cat, req: req, seed: seed, started: started)
            default: throw ParlorError.message("Engine \(engine.id) has adapter \(engine.adapter), which cannot draw.")
            }
        } catch { lastError = error.localizedDescription; throw error }
    }

    // `command`: one process per image.
    private func runCommand(host: HostInfo, engine: EngineManifest, cat: Catalog, req: ImageRequest, seed: Int, started: Date) async throws -> ImageResult {
        guard let repo = host.spec.model, let m = cat.model(repo), let file = m.file else { throw ParlorError.message("Host \(host.id) has no model file.") }
        guard let launch = engine.launch, let exe = cat.binary(for: engine) ?? EngineBinary.find(launch.command, engineDir: engine.directory) else {
            throw ParlorError.message("Engine \(engine.id) needs \(engine.requires?.hint ?? "its program").")
        }
        var c = SDCppImage.Config.load(dir: m.directory, file: file)
        Self.overlay(&c, host.spec.settings)
        let (w, h) = (req.width.flatMap { a in req.height.map { (a, $0) } }) ?? c.size(shape: req.shape)
        let steps = req.steps ?? (req.detailed ? c.steps : min(c.fastSteps, c.steps))
        let final = SDCppImage.finalPrompt(req.prompt, config: c)
        let out = Media.reserve(ext: "png", home: home)
        var values: [String: JSON] = [
            "model.file": .string(m.directory.appendingPathComponent(file).path), "prompt": .string(final),
            "negative": .string(req.negative ?? c.negative), "width": .int(w), "height": .int(h), "steps": .int(steps),
            "seed": .int(seed), "out": .string(out.url.path),
            "settings.cfg": .double(c.cfg), "settings.sampler": .string(c.sampler), "settings.scheduler": .string(c.scheduler),
            "settings.clip_skip": .int(c.clipSkip), "settings.args": .array(c.extraArgs.map { .string($0) })]
        if let p = c.prediction, !p.isEmpty { values["settings.prediction"] = .string(p) }
        if let v = c.vae, !v.isEmpty { values["settings.vae"] = .string(v) }
        let (args, missing) = ArgTemplate.expand(launch.args, values: values)
        guard missing.isEmpty else { throw ParlorError.message("Engine \(engine.id) needs values for " + missing.joined(separator: ", ")) }
        let log = ServiceLog.open(home: home, service: "image-" + host.id, banner: launch.command + " " + args.filter { !$0.hasPrefix("/") }.prefix(6).joined(separator: " "))
        commandsRunning += 1; runningModel = repo
        defer { commandsRunning -= 1; if commandsRunning == 0 { runningModel = nil }; try? log?.close() }
        let status = try await Supervisor.run(Launch(executable: exe, arguments: args, cwd: exe.deletingLastPathComponent()), output: log, timeout: 1500)
        guard status == 0 else {
            throw ParlorError.message("\(launch.command) exited with status \(status). " + ServiceLog.tail(home: home, service: "image-" + host.id, lines: 4))
        }
        guard FileManager.default.fileExists(atPath: out.url.path) else { throw ParlorError.message("\(launch.command) wrote no image.") }
        lastError = nil
        return ImageResult(path: out.path, url: out.url, width: w, height: h, seed: seed, prompt: final, host: host.id, seconds: Date().timeIntervalSince(started))
    }

    private static func overlay(_ c: inout SDCppImage.Config, _ s: [String: JSON]) {
        func int(_ k: String) -> Int? { s[k]?.intValue }
        func dbl(_ k: String) -> Double? { if case .double(let d)? = s[k] { return d }; return s[k]?.intValue.map(Double.init) }
        func ints(_ k: String) -> [Int]? { if case .array(let a)? = s[k] { return a.compactMap { $0.intValue } }; return nil }
        if let v = int("steps") { c.steps = v }
        if let v = int("fast_steps") { c.fastSteps = v }
        if let v = dbl("cfg") { c.cfg = v }
        if let v = int("clip_skip") { c.clipSkip = v }
        if let v = s["sampler"]?.stringValue { c.sampler = v }
        if let v = s["scheduler"]?.stringValue { c.scheduler = v }
        if let v = s["prediction"]?.stringValue { c.prediction = v }
        if let v = s["vae"]?.stringValue { c.vae = v }
        if let v = ints("portrait"), v.count == 2 { c.portrait = v }
        if let v = ints("square"), v.count == 2 { c.square = v }
        if let v = ints("landscape"), v.count == 2 { c.landscape = v }
        if let v = s["prompt_prefix"]?.stringValue { c.promptPrefix = v }
        if let v = s["negative"]?.stringValue { c.negative = v }
        if case .array(let a)? = s["args"] { c.extraArgs = a.compactMap { $0.stringValue } }
    }

    // `mcp-stdio`: a worker process that stays up.
    private func effective(_ host: HostInfo) -> ParlorSettings.Image {
        var i = cfg
        let s = host.spec.settings
        if let v = s["precision"]?.stringValue { i.precision = v }
        if let v = s["fast_steps"]?.intValue { i.fastSteps = v }
        if let v = s["detailed_steps"]?.intValue { i.detailedSteps = v }
        if let v = s["fast_side"]?.intValue { i.fastSide = v }
        if let v = s["detailed_side"]?.intValue { i.detailedSide = v }
        return i
    }

    private func runWorker(host: HostInfo, engine: EngineManifest, cat: Catalog, req: ImageRequest, seed: Int, started: Date) async throws -> ImageResult {
        let i = effective(host)
        let (w, h) = (req.width.flatMap { a in req.height.map { (a, $0) } }) ?? i.size(shape: req.shape, detailed: req.detailed)
        let conn = try await ensureWorker(host: host, engine: engine, cat: cat)
        workers[host.id]?.idleTask?.cancel()
        workers[host.id]?.active += 1
        defer { workers[host.id]?.active -= 1; scheduleIdle(host.id, engine: engine) }
        var a: [String: JSON] = ["prompt": .string(req.prompt), "width": .int(w), "height": .int(h), "seed": .int(seed)]
        if let steps = req.steps ?? (req.detailed ? i.detailedSteps : i.fastSteps) { a["steps"] = .int(steps) }
        let out: ToolOutput
        do { out = try await conn.call(tool: engine.tool ?? "generate", arguments: .object(a)) }
        catch {
            if await conn.pid == nil { stopWorker(host.id) }
            throw ParlorError.message("Image worker unavailable: \(error.localizedDescription)")
        }
        guard !out.isError, let rel = out.images.first else { throw ParlorError.message("Image generation failed: \(out.text)") }
        lastError = nil
        return ImageResult(path: rel, url: home.root.appendingPathComponent(rel), width: w, height: h, seed: seed, prompt: req.prompt, host: host.id, seconds: Date().timeIntervalSince(started))
    }

    static func workerURL() -> URL? {
        let fm = FileManager.default
        if let p = ProcessInfo.processInfo.environment["PARLOR_IMAGEGEN"], fm.isExecutableFile(atPath: p) { return URL(fileURLWithPath: p) }
        return nil
    }

    private func ensureWorker(host: HostInfo, engine: EngineManifest, cat: Catalog) async throws -> MCPServerConnection {
        let i = effective(host)
        guard let repo = host.spec.model, let m = cat.model(repo) else { throw ParlorError.message("Host \(host.id) has no model.") }
        let key = "\(repo)|\(i.precision)"
        if let w = workers[host.id], w.key == key { return w.conn }
        stopWorker(host.id)
        guard let launch = engine.launch,
              let exe = Self.workerURL() ?? cat.binary(for: engine) ?? EngineBinary.find(launch.command, engineDir: engine.directory) else {
            throw ParlorError.message("parlor-imagegen worker not found next to the app")
        }
        var values: [String: JSON] = ["model.dir": .string(m.directory.path), "settings.precision": .string(i.precision), "host": .string(host.id)]
        for (k, v) in host.spec.settings { values["settings." + k] = v }
        let (args, missing) = ArgTemplate.expand(launch.args, values: values)
        guard missing.isEmpty else { throw ParlorError.message("Engine \(engine.id) needs values for " + missing.joined(separator: ", ")) }
        starting.insert(host.id)
        defer { starting.remove(host.id) }
        let hostID = host.id
        let c = MCPServerConnection(
            id: "image", config: MCPServerConfig(command: exe.path, args: args),
            home: home, hub: nil, logName: "image",
            onExit: { [weak self] dead, code in Task { await self?.workerExited(hostID, dead, code) } })
        do { try await c.start(timeout: 90) } catch {
            lastError = error.localizedDescription + ServiceLog.tail(home: home, service: "image", lines: 3).prependingNewline
            throw error
        }
        lastError = nil
        workers[host.id] = Worker(conn: c, key: key, model: repo)
        return c
    }

    private func workerExited(_ host: String, _ dead: MCPServerConnection, _ code: Int32) {
        guard workers[host]?.conn === dead else { return }
        lastError = "Image worker exited with status \(code). \(ServiceLog.tail(home: home, service: "image", lines: 3))"
        workers[host]?.idleTask?.cancel()
        workers[host] = nil
    }

    private func scheduleIdle(_ host: String, engine: EngineManifest) {
        workers[host]?.idleTask?.cancel()
        let secs = cfg.idleSeconds > 0 || engine.lifecycle.idleMinutes == 0 ? Double(cfg.idleSeconds) : Double(engine.lifecycle.idleMinutes * 60)
        guard secs > 0 else { return }
        workers[host]?.idleTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(secs * 1_000_000_000))
            if !Task.isCancelled { await self?.stopIfIdle(host) }
        }
    }

    private func stopIfIdle(_ host: String) { if (workers[host]?.active ?? 1) == 0 { stopWorker(host) } }

    private func stopWorker(_ host: String) {
        guard let w = workers[host] else { return }
        w.idleTask?.cancel()
        Task { await w.conn.stop() }
        workers[host] = nil
    }

    // MARK: the tool

    public func listTools() async throws -> [ToolDef] {
        let schema: JSON = .object([
            "type": .string("object"),
            "properties": .object([
                "prompt": .object(["type": .string("string"), "description": .string("What to draw. Be concrete: subject, setting, style, lighting.")]),
                "shape": .object(["type": .string("string"), "enum": .array([.string("square"), .string("portrait"), .string("landscape")]), "description": .string("Image shape. Default square.")]),
                "detail": .object(["type": .string("string"), "enum": .array([.string("fast"), .string("detailed")]), "description": .string("fast takes about 30 seconds, detailed about 70. Default fast.")]),
                "seed": .object(["type": .string("integer"), "description": .string("Optional seed for repeatable results.")]),
                "negative": .object(["type": .string("string"), "description": .string("Optional: what to keep out of the picture (Stable Diffusion models only).")]),
            ]),
            "required": .array([.string("prompt")]),
        ])
        return [ToolDef(server: "image", name: "generate", description: "Generate an image from a text prompt with the local image model and show it in the chat.", schema: schema)]
    }

    public func call(tool: String, arguments: JSON) async throws -> ToolOutput { try await call(tool: tool, arguments: arguments, caller: .local) }

    public func call(tool: String, arguments: JSON, caller: Caller) async throws -> ToolOutput {
        guard tool == "generate" else { return ToolOutput(text: "Unknown image tool \(tool)", isError: true) }
        guard let prompt = arguments.objectValue?["prompt"]?.stringValue, !prompt.isEmpty else {
            return ToolOutput(text: "Missing prompt", isError: true)
        }
        // The host comes from the calling agent on every call.
        guard let host = await hostFor(caller.kind == .agent ? caller.name : nil) else {
            return ToolOutput(text: "No image model is downloaded. Ask the user to download one in Models.", isError: true)
        }
        let o = arguments.objectValue ?? [:]
        let req = ImageRequest(prompt: prompt, negative: o["negative"]?.stringValue, shape: o["shape"]?.stringValue ?? "square",
                               detailed: o["detail"]?.stringValue == "detailed", seed: o["seed"]?.intValue)
        do {
            let r = try await generate(host: host, req)
            return ToolOutput(text: "The image was generated and is already displayed to the user inline; do not write a placeholder like [image] or a link in your reply, just comment on it naturally. (\(r.width)x\(r.height), seed \(r.seed), prompt: \(r.prompt))", images: [r.path])
        } catch {
            return ToolOutput(text: "Image generation failed: \(error.localizedDescription)", isError: true)
        }
    }
}

private extension String { var prependingNewline: String { isEmpty ? "" : "\n" + self } }

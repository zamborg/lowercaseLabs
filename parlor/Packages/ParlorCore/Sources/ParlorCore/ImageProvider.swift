import Foundation

/// Built-in `image` tool server. The model runs in a separate worker process (`parlor-imagegen`), which is an
/// ordinary MCP server over stdio, so image-model memory returns to the OS when idle and a crash cannot take the
/// chat down. This type is the app's side: it picks the model and sizes from settings, supervises the worker
/// (start, idle stop, restart on a precision change) and words the result for the model.
public actor ImageProvider: ToolProvider {
    public nonisolated let id = "image"
    private let home: ParlorHome
    private let library: ModelLibrary
    private var modelRepo: String?
    private var conn: MCPServerConnection?
    private var idleTask: Task<Void, Never>?
    private var runningModel: String?
    private var cfg = ParlorSettings.Image()
    private var runningKey: String?
    private var starting = false
    private var activeCalls = 0
    public private(set) var lastUsed: Date?
    public private(set) var lastError: String?

    public init(home: ParlorHome, library: ModelLibrary) { self.home = home; self.library = library }

    public func setModel(_ repo: String?) { modelRepo = repo }

    /// Apply settings. A precision change restarts the worker on its next use; the rest takes effect immediately.
    public func configure(_ new: ParlorSettings.Image) {
        let precisionChanged = new.precision != cfg.precision
        cfg = new
        if precisionChanged { shutdownNow() } else if conn != nil { scheduleIdleShutdown() }
    }

    public func stateDescription() -> String {
        if let m = runningModel { return "loaded · \(m.split(separator: "/").last.map(String.init) ?? m)" }
        return "idle"
    }

    /// For the services dashboard.
    public func serviceSnapshot() async -> (state: ServiceState, pid: Int32?, model: String?, lastUsed: Date?) {
        let pid = await conn?.pid
        let state: ServiceState
        if starting { state = .starting }
        else if conn != nil { state = activeCalls > 0 ? .running : .idle }
        else if let e = lastError { state = .failed(e) }
        else { state = .stopped }
        return (state, pid, runningModel, lastUsed)
    }

    /// Start the worker without a request, so the model is loaded before the first image.
    public func start() async throws {
        guard let (repo, dir) = resolveModel() else { throw ParlorError.message("No image model is downloaded. Download one in Models.") }
        try await ensureWorker(model: repo, dir: dir)
        scheduleIdleShutdown()
    }

    public func stop() { shutdownNow() }

    private func resolveModel() -> (String, URL)? {
        guard let repo = modelRepo ?? library.installed(kind: .image).first?.repo,
              let dir = library.installedDirectory(for: repo) else { return nil }
        return (repo, dir)
    }

    public func listTools() async throws -> [ToolDef] {
        let schema: JSON = .object([
            "type": .string("object"),
            "properties": .object([
                "prompt": .object(["type": .string("string"), "description": .string("What to draw. Be concrete: subject, setting, style, lighting.")]),
                "shape": .object(["type": .string("string"), "enum": .array([.string("square"), .string("portrait"), .string("landscape")]), "description": .string("Image shape. Default square.")]),
                "detail": .object(["type": .string("string"), "enum": .array([.string("fast"), .string("detailed")]), "description": .string("fast takes about 30 seconds, detailed about 70. Default fast.")]),
                "seed": .object(["type": .string("integer"), "description": .string("Optional seed for repeatable results.")]),
            ]),
            "required": .array([.string("prompt")]),
        ])
        return [ToolDef(server: "image", name: "generate", description: "Generate an image from a text prompt with the local image model and show it in the chat.", schema: schema)]
    }

    public func call(tool: String, arguments: JSON) async throws -> ToolOutput {
        guard tool == "generate" else { return ToolOutput(text: "Unknown image tool \(tool)", isError: true) }
        guard let prompt = arguments.objectValue?["prompt"]?.stringValue, !prompt.isEmpty else {
            return ToolOutput(text: "Missing prompt", isError: true)
        }
        let shape = arguments.objectValue?["shape"]?.stringValue ?? "square"
        let detailed = arguments.objectValue?["detail"]?.stringValue == "detailed"
        // Render time scales with pixel count: ~30 s at 512², ~70 s at 768² on an M1 Max.
        let (w, h) = cfg.size(shape: shape, detailed: detailed)
        let seed = arguments.objectValue?["seed"]?.intValue ?? Int.random(in: 0..<100_000)
        guard let repo = modelRepo ?? library.installed(kind: .image).first?.repo else {
            return ToolOutput(text: "No image model is downloaded. Ask the user to download one in Models.", isError: true)
        }
        guard let dir = library.installedDirectory(for: repo) else {
            return ToolOutput(text: "Image model \(repo) is not downloaded.", isError: true)
        }
        idleTask?.cancel()
        activeCalls += 1
        defer { activeCalls -= 1; lastUsed = Date(); scheduleIdleShutdown() }
        let out: ToolOutput
        do {
            try await ensureWorker(model: repo, dir: dir)
            guard let conn else { return ToolOutput(text: "Image worker unavailable", isError: true) }
            var a: [String: JSON] = ["prompt": .string(prompt), "width": .int(w), "height": .int(h), "seed": .int(seed)]
            if let steps = detailed ? cfg.detailedSteps : cfg.fastSteps { a["steps"] = .int(steps) }
            out = try await conn.call(tool: "generate", arguments: .object(a))
        } catch {
            lastError = error.localizedDescription
            if conn != nil, await conn?.pid == nil { shutdownNow() }
            return ToolOutput(text: "Image worker unavailable: \(error.localizedDescription)", isError: true)
        }
        guard !out.isError, let rel = out.images.first else {
            return ToolOutput(text: "Image generation failed: \(out.text)", isError: true)
        }
        return ToolOutput(text: "The image was generated and is already displayed to the user inline; do not write a placeholder like [image] or a link in your reply, just comment on it naturally. (\(w)x\(h), seed \(seed), prompt: \(prompt))", images: [rel])
    }

    // MARK: worker process

    static func workerURL() -> URL? {
        let fm = FileManager.default
        if let p = ProcessInfo.processInfo.environment["PARLOR_IMAGEGEN"], fm.isExecutableFile(atPath: p) { return URL(fileURLWithPath: p) }
        var candidates: [URL] = []
        if let helpers = Bundle.main.bundleURL.appendingPathComponent("Contents/Resources/ImageWorker", isDirectory: true) as URL? {
            candidates.append(helpers.appendingPathComponent("parlor-imagegen"))
        }
        if let exe = Bundle.main.executableURL?.deletingLastPathComponent() {
            candidates.append(exe.appendingPathComponent("parlor-imagegen"))
        }
        return candidates.first { fm.isExecutableFile(atPath: $0.path) }
    }

    private func ensureWorker(model: String, dir: URL) async throws {
        let key = "\(model)|\(cfg.precision)"
        if conn != nil, runningKey == key { return }
        shutdownNow()
        guard let exe = Self.workerURL() else { throw ParlorError.message("parlor-imagegen worker not found next to the app") }
        starting = true
        defer { starting = false }
        let c = MCPServerConnection(
            id: "image", config: MCPServerConfig(command: exe.path, args: [dir.path, cfg.precision]),
            home: home, hub: nil, logName: "image",
            onExit: { [weak self] dead, code in Task { await self?.workerExited(dead, code) } })
        do { try await c.start(timeout: 90) } catch {
            lastError = error.localizedDescription + ServiceLog.tail(home: home, service: "image", lines: 3).prependingNewline
            throw error
        }
        lastError = nil
        conn = c; runningModel = model; runningKey = key
    }

    private func workerExited(_ dead: MCPServerConnection, _ code: Int32) {
        guard conn === dead else { return }
        lastError = "Image worker exited with status \(code). \(ServiceLog.tail(home: home, service: "image", lines: 3))"
        conn = nil; runningModel = nil; runningKey = nil
    }

    private func scheduleIdleShutdown() {
        idleTask?.cancel()
        let secs = Double(cfg.idleSeconds)
        guard secs > 0 else { return }
        idleTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(secs * 1_000_000_000))
            if !Task.isCancelled { await self?.shutdown() }
        }
    }

    private func shutdownNow() {
        idleTask?.cancel()
        if let c = conn { Task { await c.stop() } }
        conn = nil; runningModel = nil; runningKey = nil
    }

    public func shutdown() { shutdownNow() }
}

private extension String { var prependingNewline: String { isEmpty ? "" : "\n" + self } }

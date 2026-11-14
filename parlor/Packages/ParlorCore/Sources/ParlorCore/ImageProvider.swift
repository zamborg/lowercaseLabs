import Foundation

/// Built-in `image` tool server. Runs a separate worker process so image-model memory is
/// returned to the OS when idle and a crash cannot take the chat down.
///
/// Worker protocol (JSON lines): stdin `{"id","prompt","width","height","seed","steps"}`,
/// stdout `{"id","ok":true,"path"}` or `{"id","ok":false,"error"}`. Launched as
/// `parlor-imagegen <model-dir>`.
public actor ImageProvider: ToolProvider {
    public nonisolated let id = "image"
    private let home: ParlorHome
    private let library: ModelLibrary
    private var modelRepo: String?
    private var process: Process?
    private var stdin: FileHandle?
    private var reader: Task<Void, Never>?
    private var pending: [String: CheckedContinuation<WorkerReply, Never>] = [:]
    private var idleTask: Task<Void, Never>?
    private var runningModel: String?
    public var idleSeconds: Double = 120

    struct WorkerReply: Sendable { var ok: Bool; var path: String?; var error: String? }

    public init(home: ParlorHome, library: ModelLibrary) { self.home = home; self.library = library }

    public func setModel(_ repo: String?) { modelRepo = repo }

    public func stateDescription() -> String {
        if let m = runningModel { return "loaded · \(m.split(separator: "/").last.map(String.init) ?? m)" }
        return "idle"
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
        let (w, h): (Int, Int) = switch (shape, detailed) {
        case ("portrait", false): (448, 640)
        case ("landscape", false): (640, 448)
        case ("portrait", true): (640, 896)
        case ("landscape", true): (896, 640)
        case (_, true): (768, 768)
        default: (512, 512)
        }
        let seed = arguments.objectValue?["seed"]?.intValue ?? Int.random(in: 0..<100_000)
        guard let repo = modelRepo ?? library.installed(kind: .image).first?.repo else {
            return ToolOutput(text: "No image model is downloaded. Ask the user to download one in Models.", isError: true)
        }
        guard let dir = library.installedDirectory(for: repo) else {
            return ToolOutput(text: "Image model \(repo) is not downloaded.", isError: true)
        }
        idleTask?.cancel()
        do {
            try ensureWorker(model: repo, dir: dir)
        } catch {
            return ToolOutput(text: "Image worker unavailable: \(error.localizedDescription)", isError: true)
        }
        let rid = UUID().uuidString
        let req: [String: Any] = ["id": rid, "prompt": prompt, "width": w, "height": h, "seed": seed, "steps": 0]
        let reply: WorkerReply = await withCheckedContinuation { cont in
            pending[rid] = cont
            if let d = try? JSONSerialization.data(withJSONObject: req) {
                var line = d; line.append(0x0A)
                try? stdin?.write(contentsOf: line)
            } else {
                pending[rid] = nil
                cont.resume(returning: WorkerReply(ok: false, path: nil, error: "bad request"))
            }
        }
        scheduleIdleShutdown()
        guard reply.ok, let abs = reply.path, let data = FileManager.default.contents(atPath: abs) else {
            return ToolOutput(text: "Image generation failed: \(reply.error ?? "unknown error")", isError: true)
        }
        let rel = Media.save(data: data, ext: "png", home: home)
        try? FileManager.default.removeItem(atPath: abs)
        guard let rel else { return ToolOutput(text: "Could not save image", isError: true) }
        return ToolOutput(text: "Image generated (\(w)x\(h), seed \(seed)) and shown to the user. Prompt: \(prompt)", images: [rel])
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

    private func ensureWorker(model: String, dir: URL) throws {
        if let p = process, p.isRunning, runningModel == model { return }
        shutdownNow()
        guard let exe = Self.workerURL() else { throw ParlorError.message("parlor-imagegen worker not found next to the app") }
        let proc = Process()
        proc.executableURL = exe
        proc.arguments = [dir.path]
        let inPipe = Pipe(), outPipe = Pipe()
        proc.standardInput = inPipe
        proc.standardOutput = outPipe
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        process = proc
        stdin = inPipe.fileHandleForWriting
        runningModel = model
        let handle = outPipe.fileHandleForReading
        reader = Task.detached { [weak self] in
            var buffer = Data()
            while true {
                let chunk = handle.availableData
                if chunk.isEmpty { break }
                buffer.append(chunk)
                while let nl = buffer.firstIndex(of: 0x0A) {
                    let line = buffer[buffer.startIndex..<nl]
                    buffer.removeSubrange(buffer.startIndex...nl)
                    guard let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
                          let rid = obj["id"] as? String else { continue }
                    let reply = WorkerReply(ok: obj["ok"] as? Bool ?? false, path: obj["path"] as? String, error: obj["error"] as? String)
                    await self?.resolve(rid, reply)
                }
            }
            await self?.workerExited()
        }
    }

    private func resolve(_ id: String, _ r: WorkerReply) {
        pending.removeValue(forKey: id)?.resume(returning: r)
    }

    private func workerExited() {
        for (_, c) in pending { c.resume(returning: WorkerReply(ok: false, path: nil, error: "image worker exited")) }
        pending = [:]
        process = nil; stdin = nil; runningModel = nil
    }

    private func scheduleIdleShutdown() {
        idleTask?.cancel()
        let secs = idleSeconds
        idleTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(secs * 1_000_000_000))
            if !Task.isCancelled { await self?.shutdown() }
        }
    }

    private func shutdownNow() {
        idleTask?.cancel()
        if let p = process, p.isRunning { p.terminate() }
        process = nil; stdin = nil; runningModel = nil
    }

    public func shutdown() { shutdownNow() }
}

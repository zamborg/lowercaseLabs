import Foundation

/// A backend that can serve a capability. Built-ins are manifest folders like a user's.
public struct EngineManifest: Noun, Equatable {
    public static let kind = "engine"

    public struct Launch: Codable, Equatable, Sendable {
        public var command: String
        public var args: [String] = []
        public init(command: String, args: [String] = []) { self.command = command; self.args = args }
        public init(from d: Decoder) throws {
            let c = try d.container(keyedBy: CodingKeys.self)
            command = try c.decode(String.self, forKey: .command)
            args = try c.decodeIfPresent([String].self, forKey: .args) ?? []
        }
    }
    public struct Requires: Codable, Equatable, Sendable {
        public var binary: String
        public var hint: String?
    }
    public struct Lifecycle: Codable, Equatable, Sendable {
        public var idleMinutes: Int = 30
        public var startupSeconds: Int = 180
        public init() {}
        public init(from d: Decoder) throws {
            let c = try d.container(keyedBy: CodingKeys.self)
            idleMinutes = try c.decodeIfPresent(Int.self, forKey: .idleMinutes) ?? 30
            startupSeconds = try c.decodeIfPresent(Int.self, forKey: .startupSeconds) ?? 180
        }
    }

    public var version: Int = 1
    public var id: String
    public var title: String
    public var capabilities: [String]
    public var formats: [String]
    /// `inprocess-mlx`, `http-openai`, `command` or `mcp-stdio`.
    public var adapter: String
    public var launch: Launch?
    public var health: String?
    /// For `mcp-stdio`: the tool the worker exposes.
    public var tool: String?
    /// Sampling parameters this engine honours; the rest are dropped and reported.
    public var params: [String] = []
    public var requires: Requires?
    public var settings: JSON?
    public var lifecycle: Lifecycle = Lifecycle()
    /// True for engines that serve a remote URL named by the host (`openai-remote`).
    public var remote: Bool = false

    enum CodingKeys: String, CodingKey { case version, id, title, capabilities, formats, adapter, launch, health, tool, params, requires, settings, lifecycle, remote }

    public init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        version = try c.decodeIfPresent(Int.self, forKey: .version) ?? 1
        id = try c.decode(String.self, forKey: .id)
        title = try c.decodeIfPresent(String.self, forKey: .title) ?? id
        capabilities = try c.decode([String].self, forKey: .capabilities)
        formats = try c.decodeIfPresent([String].self, forKey: .formats) ?? []
        adapter = try c.decode(String.self, forKey: .adapter)
        launch = try c.decodeIfPresent(Launch.self, forKey: .launch)
        health = try c.decodeIfPresent(String.self, forKey: .health)
        tool = try c.decodeIfPresent(String.self, forKey: .tool)
        params = try c.decodeIfPresent([String].self, forKey: .params) ?? []
        requires = try c.decodeIfPresent(Requires.self, forKey: .requires)
        settings = try c.decodeIfPresent(JSON.self, forKey: .settings)
        lifecycle = try c.decodeIfPresent(Lifecycle.self, forKey: .lifecycle) ?? Lifecycle()
        remote = try c.decodeIfPresent(Bool.self, forKey: .remote) ?? false
    }

    public static let adapters = ["inprocess-mlx", "http-openai", "command", "mcp-stdio"]

    public static func check(_ d: [String: JSON]) -> [Issue] {
        var out: [Issue] = []
        if let a = d["adapter"]?.stringValue, !adapters.contains(a) {
            out.append(Issue(path: "adapter", message: "unknown adapter \"\(a)\"", hint: "one of " + adapters.joined(separator: ", ")))
        }
        if let a = d["adapter"]?.stringValue, a != "inprocess-mlx", d["launch"] == nil, d["remote"] != .bool(true) {
            out.append(Issue(code: "missing", path: "launch", message: "adapter \(a) needs a launch command (or \"remote\": true)"))
        }
        return out
    }

    /// Where the engine's own folder is, once loaded (nil for a value built in memory).
    public var directory: URL?
}

// MARK: - finding binaries

public enum EngineBinary {
    /// The manifest's folder, the app bundle's `Resources/`, `~/.local/bin`, `/opt/homebrew/bin`, `/usr/local/bin`.
    /// `PARLOR_BIN_<NAME>` overrides, for tests and odd installs.
    public static func find(_ name: String, engineDir: URL? = nil) -> URL? {
        let fm = FileManager.default
        let env = ProcessInfo.processInfo.environment
        let key = "PARLOR_BIN_" + name.uppercased().replacingOccurrences(of: "-", with: "_").replacingOccurrences(of: ".", with: "_")
        if let p = env[key], fm.isExecutableFile(atPath: p) { return URL(fileURLWithPath: p) }
        if name.contains("/") { return fm.isExecutableFile(atPath: name) ? URL(fileURLWithPath: name) : nil }
        var dirs: [URL] = []
        if let e = engineDir { dirs.append(e) }
        let res = Bundle.main.bundleURL.appendingPathComponent("Contents/Resources", isDirectory: true)
        dirs += [res, res.appendingPathComponent("SD"), res.appendingPathComponent("ImageWorker")]
        if let exe = Bundle.main.executableURL?.deletingLastPathComponent() { dirs.append(exe) }
        let home = fm.homeDirectoryForCurrentUser.path
        dirs += ["\(home)/.local/bin", "/opt/homebrew/bin", "/usr/local/bin"].map { URL(fileURLWithPath: $0) }
        return dirs.map { $0.appendingPathComponent(name) }.first { fm.isExecutableFile(atPath: $0.path) }
    }
}

// MARK: - the shipped engines

/// Built-in engines, written as plain manifest folders. A file carrying `"managed": "parlor-engines-v1"` is rewritten when it
/// differs; a file without the marker is the user's and is left alone.
public enum EngineSeeds {
    public static let marker = "parlor-engines-v1"

    static let sampling = #"["temperature", "top_p", "min_p", "top_k", "repetition_penalty", "max_tokens"]"#

    public static let manifests: [String: String] = [
        "mlx": """
        { "managed": "\(marker)", "version": 1, "id": "mlx", "title": "MLX (in the app)",
          "capabilities": ["chat"], "formats": ["mlx-lm"], "adapter": "inprocess-mlx",
          "params": ["temperature", "top_p", "repetition_penalty", "max_tokens"],
          "settings": { "type": "object", "properties": {
              "context": { "type": "integer", "minimum": 512, "description": "caps the context window; blank uses the model's own" } } } }
        """,
        "llama.cpp": """
        { "managed": "\(marker)", "version": 1, "id": "llama.cpp", "title": "llama.cpp",
          "capabilities": ["chat"], "formats": ["gguf"], "adapter": "http-openai",
          "launch": { "command": "llama-server",
                      "args": ["-m", "{model.file}", "--alias", "{host}", "--port", "{port}", "--jinja",
                               "-c", "{settings.context}", "-ngl", "{settings.gpu_layers}", "{settings.args...}"] },
          "health": "/v1/models",
          "params": \(sampling),
          "requires": { "binary": "llama-server", "hint": "brew install llama.cpp" },
          "settings": { "type": "object", "properties": {
              "context": { "type": "integer", "default": 8192, "minimum": 512, "description": "context window in tokens" },
              "gpu_layers": { "type": "integer", "default": 99, "minimum": 0, "description": "layers offloaded to the GPU" },
              "args": { "type": "array", "items": { "type": "string" }, "default": [], "description": "extra llama-server flags" } } },
          "lifecycle": { "idleMinutes": 30, "startupSeconds": 600 } }
        """,
        "stable-diffusion.cpp": """
        { "managed": "\(marker)", "version": 1, "id": "stable-diffusion.cpp", "title": "stable-diffusion.cpp",
          "capabilities": ["image"], "formats": ["sd-checkpoint"], "adapter": "command",
          "launch": { "command": "sd-cli",
                      "args": ["-m", "{model.file}", "-p", "{prompt}", "-n", "{negative}", "-W", "{width}", "-H", "{height}",
                               "--steps", "{steps}", "--cfg-scale", "{settings.cfg}", "--sampling-method", "{settings.sampler}",
                               "--scheduler", "{settings.scheduler}", "--clip-skip", "{settings.clip_skip}", "--seed", "{seed}",
                               "--prediction", "{settings.prediction}?", "--vae", "{settings.vae}?", "{settings.args...}", "-o", "{out}"] },
          "requires": { "binary": "sd-cli", "hint": "bundled with the app, or install stable-diffusion.cpp" },
          "settings": { "type": "object", "properties": {
              "sampler": { "type": "string", "default": "euler_a" },
              "scheduler": { "type": "string", "default": "discrete" },
              "steps": { "type": "integer", "default": 28, "minimum": 1, "description": "steps for detail: detailed" },
              "fast_steps": { "type": "integer", "default": 20, "minimum": 1, "description": "steps for detail: fast" },
              "cfg": { "type": "number", "default": 6, "minimum": 0 },
              "clip_skip": { "type": "integer", "default": 2, "minimum": 1 },
              "prediction": { "type": "string", "enum": ["eps", "v"], "description": "v for V-prediction checkpoints; blank detects" },
              "vae": { "type": "string", "description": "path to a standalone VAE" },
              "portrait": { "type": "array", "items": { "type": "integer" }, "default": [832, 1216] },
              "square": { "type": "array", "items": { "type": "integer" }, "default": [1024, 1024] },
              "landscape": { "type": "array", "items": { "type": "integer" }, "default": [1216, 832] },
              "prompt_prefix": { "type": "string", "default": "masterpiece, best quality, amazing quality, absurdres, ", "format": "multiline" },
              "negative": { "type": "string", "default": "worst quality, low quality, bad anatomy, bad hands, missing fingers, extra digits, jpeg artifacts, signature, watermark, text, username, blurry", "format": "multiline" },
              "args": { "type": "array", "items": { "type": "string" }, "default": [] } } },
          "lifecycle": { "idleMinutes": 0 } }
        """,
        "z-image": """
        { "managed": "\(marker)", "version": 1, "id": "z-image", "title": "Z-Image (MLX)",
          "capabilities": ["image"], "formats": ["z-image"], "adapter": "mcp-stdio", "tool": "generate",
          "launch": { "command": "parlor-imagegen", "args": ["{model.dir}", "{settings.precision}"] },
          "requires": { "binary": "parlor-imagegen", "hint": "bundled with the app" },
          "settings": { "type": "object", "properties": {
              "precision": { "type": "string", "enum": ["bf16", "int8", "int4"], "default": "int8" },
              "fast_steps": { "type": "integer", "minimum": 1, "description": "blank uses the model's own" },
              "detailed_steps": { "type": "integer", "minimum": 1 },
              "fast_side": { "type": "integer", "default": 512, "minimum": 256 },
              "detailed_side": { "type": "integer", "default": 768, "minimum": 256 } } },
          "lifecycle": { "idleMinutes": 2 } }
        """,
        "openai-remote": """
        { "managed": "\(marker)", "version": 1, "id": "openai-remote", "title": "OpenAI-compatible server",
          "capabilities": ["chat"], "formats": [], "adapter": "http-openai", "remote": true,
          "params": \(sampling),
          "settings": { "type": "object", "properties": {} } }
        """,
    ]

    /// Write the shipped manifests that are missing or managed and out of date. Returns the ids written.
    @discardableResult
    public static func seed(home: ParlorHome) -> [String] {
        let fm = FileManager.default
        var written: [String] = []
        for (id, text) in manifests.sorted(by: { $0.key < $1.key }) {
            let dir = home.enginesDir.appendingPathComponent(id)
            let file = dir.appendingPathComponent("engine.json")
            if let existing = try? String(contentsOf: file, encoding: .utf8) {
                guard existing.contains("\"managed\"") && existing.contains("parlor-engines-"), existing != text else { continue }
            }
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
            if (try? text.write(to: file, atomically: true, encoding: .utf8)) != nil { written.append(id) }
        }
        let guide = home.enginesDir.appendingPathComponent("CLAUDE.md")
        if (try? String(contentsOf: guide, encoding: .utf8))?.contains("parlor-engines-") ?? true {
            try? guideText.write(to: guide, atomically: true, encoding: .utf8)
        }
        return written
    }

    public static let guideText = "<!-- \(marker) -->\n" + """
    # Engines

    An engine is a folder here, `engines/<id>/engine.json`, that tells parlor how to serve a kind of model.
    Copy a shipped engine to make your own (remove its `"managed"` key so parlor stops rewriting it).

    Fields: `id`, `title`, `capabilities` (`chat`, `image`), `formats` (model formats it can run: `mlx-lm`, `gguf`,
    `sd-checkpoint`, `z-image`), `adapter`, `launch`, `health`, `params`, `requires`, `settings`, `lifecycle`.

    Adapters:
    - `inprocess-mlx`: loads MLX weights inside the app.
    - `http-openai`: a server that already speaks OpenAI. With `launch`, parlor starts it on a free port and
      waits for `health`. With `"remote": true`, a host names the URL and parlor just talks to it.
    - `command`: runs one process per request (images).
    - `mcp-stdio`: a long-lived MCP worker; `tool` names the tool that draws.

    Launch args are a list, never a shell line. Placeholders: `{model.file}`, `{model.dir}`, `{host}`, `{port}`,
    `{settings.x}`, `{prompt}`, `{negative}`, `{width}`, `{height}`, `{steps}`, `{seed}`, `{out}`.
    `{x...}` splices a list. A trailing `?` makes a value optional: when it is unset, the value and the flag
    before it are dropped, for example `["--vae", "{settings.vae}?"]`.

    `requires.binary` is looked up in this folder, the app's Resources, `~/.local/bin`, `/opt/homebrew/bin` and
    `/usr/local/bin`. `settings` is a small JSON Schema (type, enum, default, minimum, maximum, description,
    format: multiline) that Tuning turns into a form. A host's `settings` fill the placeholders.

    To use an LM Studio or Ollama server: a host with `"engine": "openai-remote"`, `"url": "http://127.0.0.1:1234/v1"`
    and `"enabled": true`. Nothing else is needed.
    """
}

public extension ParlorHome {
    var enginesDir: URL { root.appendingPathComponent("engines") }
    var hostsDir: URL { root.appendingPathComponent("hosts") }
}

// MARK: - argument templates

public enum ArgTemplate {
    /// Fill `{name}` placeholders. `{x...}` splices a list. `{x}?` drops itself and the flag before it when unset.
    /// A missing required value yields nil so the caller can report which one.
    public static func expand(_ args: [String], values: [String: JSON]) -> (args: [String], missing: [String]) {
        var out: [String] = []
        var missing: [String] = []
        for a in args {
            let optional = a.hasSuffix("?")
            let body = optional ? String(a.dropLast()) : a
            guard body.hasPrefix("{"), body.hasSuffix("}"), !body.dropFirst().dropLast().contains("{") else {
                out.append(substitute(body, values: values)); continue
            }
            var key = String(body.dropFirst().dropLast())
            let splice = key.hasSuffix("...")
            if splice { key = String(key.dropLast(3)) }
            let v = values[key]
            let unset: Bool = { if case .null? = v { return true }; if v == nil { return true }; if v?.stringValue == "" { return true }; return false }()
            if unset {
                if optional || splice {
                    if optional, let last = out.last, last.hasPrefix("-") { out.removeLast() }
                } else { missing.append(key) }
                continue
            }
            if splice, case .array(let items)? = v { out += items.map { scalar($0) } } else { out.append(scalar(v!)) }
        }
        return (out, missing)
    }

    private static func substitute(_ s: String, values: [String: JSON]) -> String {
        var out = s
        for (k, v) in values { out = out.replacingOccurrences(of: "{\(k)}", with: scalar(v)) }
        return out
    }

    private static func scalar(_ j: JSON) -> String {
        switch j {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return d == d.rounded() && abs(d) < 1e12 ? String(Int(d)) : String(d)
        case .bool(let b): return b ? "true" : "false"
        default: return j.text
        }
    }
}

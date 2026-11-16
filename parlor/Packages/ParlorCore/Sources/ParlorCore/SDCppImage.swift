import Foundation

/// Draws with a single-file Stable Diffusion checkpoint (SDXL, Illustrious, Pony and the like) by running
/// stable-diffusion.cpp's `sd-cli` once per image. Per-model settings live in `parlor-image.json` next to the weights.
public enum SDCppImage {
    /// `parlor-image.json`: how to sample this checkpoint. Written with defaults on install; edit it by hand.
    public struct Config: Codable, Equatable, Sendable {
        /// `v` for V-prediction checkpoints, `eps` for ordinary ones, nil lets sd-cli detect it.
        public var prediction: String?
        public var sampler = "euler_a"
        public var scheduler = "discrete"
        public var steps = 28
        /// Steps when the tool asks for `detail: fast`.
        public var fastSteps = 20
        public var cfg = 6.0
        public var clipSkip = 2
        /// Width × height per shape.
        public var portrait = [832, 1216]
        public var square = [1024, 1024]
        public var landscape = [1216, 832]
        /// Added in front of every prompt that does not already start with quality tags.
        public var promptPrefix = "masterpiece, best quality, amazing quality, absurdres, "
        public var negative = "worst quality, low quality, bad anatomy, bad hands, missing fingers, extra digits, jpeg artifacts, signature, watermark, text, username, blurry"
        /// A standalone VAE file; nil uses the checkpoint's own.
        public var vae: String?
        /// Anything else for sd-cli, for example `["--vae-tiling"]`.
        public var extraArgs: [String] = []

        public init() {}
        public init(from d: Decoder) throws {
            let c = try d.container(keyedBy: CodingKeys.self)
            let base = Config()
            prediction = try c.decodeIfPresent(String.self, forKey: .prediction)
            sampler = try c.decodeIfPresent(String.self, forKey: .sampler) ?? base.sampler
            scheduler = try c.decodeIfPresent(String.self, forKey: .scheduler) ?? base.scheduler
            steps = try c.decodeIfPresent(Int.self, forKey: .steps) ?? base.steps
            fastSteps = try c.decodeIfPresent(Int.self, forKey: .fastSteps) ?? base.fastSteps
            cfg = try c.decodeIfPresent(Double.self, forKey: .cfg) ?? base.cfg
            clipSkip = try c.decodeIfPresent(Int.self, forKey: .clipSkip) ?? base.clipSkip
            portrait = try c.decodeIfPresent([Int].self, forKey: .portrait) ?? base.portrait
            square = try c.decodeIfPresent([Int].self, forKey: .square) ?? base.square
            landscape = try c.decodeIfPresent([Int].self, forKey: .landscape) ?? base.landscape
            promptPrefix = try c.decodeIfPresent(String.self, forKey: .promptPrefix) ?? base.promptPrefix
            negative = try c.decodeIfPresent(String.self, forKey: .negative) ?? base.negative
            vae = try c.decodeIfPresent(String.self, forKey: .vae)
            extraArgs = try c.decodeIfPresent([String].self, forKey: .extraArgs) ?? []
        }

        /// Defaults for a checkpoint file name: V-prediction ones are recognised by their name.
        public static func defaults(forFile file: String) -> Config {
            var c = Config()
            let n = file.lowercased().replacingOccurrences(of: "_", with: "-")
            if ["v-pred", "vpred", "v-prediction", "vprediction"].contains(where: { n.contains($0) }) { c.prediction = "v" }
            return c
        }

        public static func url(in dir: URL) -> URL { dir.appendingPathComponent("parlor-image.json") }

        /// The model's settings: its file if there is one, else defaults from the checkpoint's name.
        public static func load(dir: URL, file: String) -> Config {
            if let d = try? Data(contentsOf: url(in: dir)), let c = try? JSONDecoder().decode(Config.self, from: d) { return c }
            return defaults(forFile: file)
        }

        /// Write the defaults beside a fresh download. Never overwrites an edited file.
        public static func writeDefault(in dir: URL, file: String) {
            let u = url(in: dir)
            guard !FileManager.default.fileExists(atPath: u.path) else { return }
            let e = JSONEncoder(); e.outputFormatting = [.prettyPrinted, .sortedKeys]
            try? e.encode(defaults(forFile: file)).write(to: u)
        }

        public func size(shape: String) -> (Int, Int) {
            let v = shape == "portrait" ? portrait : shape == "landscape" ? landscape : square
            return v.count == 2 ? (v[0], v[1]) : (1024, 1024)
        }
    }

    /// `sd-cli` and the library it loads: bundled in the app, next to the CLI, or installed by the user.
    public static func binaryURL() -> URL? {
        if let p = ProcessInfo.processInfo.environment["PARLOR_SD"], FileManager.default.isExecutableFile(atPath: p) { return URL(fileURLWithPath: p) }
        return EngineBinary.find("sd-cli")
    }

    /// The prompt as drawn: tag-style prompts get the quality tags in front unless they already carry them.
    public static func finalPrompt(_ prompt: String, config c: Config) -> String {
        let p = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if c.promptPrefix.isEmpty || p.lowercased().contains("masterpiece") || p.lowercased().contains("best quality") { return p }
        return c.promptPrefix + p
    }

    /// The `sd-cli` command line for one image.
    public static func arguments(config c: Config, model: String, prompt: String, negative: String?, width: Int, height: Int,
                                 seed: Int, steps: Int, output: String) -> [String] {
        var a = ["-m", model, "-p", finalPrompt(prompt, config: c), "-n", negative ?? c.negative,
                 "-W", String(width), "-H", String(height), "--steps", String(steps), "--cfg-scale", String(c.cfg),
                 "--sampling-method", c.sampler, "--scheduler", c.scheduler, "--clip-skip", String(c.clipSkip),
                 "--seed", String(seed), "-o", output]
        if let p = c.prediction, !p.isEmpty { a += ["--prediction", p] }
        if let v = c.vae, !v.isEmpty { a += ["--vae", v] }
        return a + c.extraArgs
    }

    /// Run one render. Cancelling the task stops `sd-cli`. Throws with the tail of its output when it fails.
    static func render(exe: URL, arguments: [String], home: ParlorHome, timeout: TimeInterval = 1500) async throws {
        let log = ServiceLog.open(home: home, service: "image-sdcpp", banner: "sd-cli " + arguments.filter { !$0.hasPrefix("/") }.prefix(6).joined(separator: " "))
        let launch = Launch(executable: exe, arguments: arguments, cwd: exe.deletingLastPathComponent())
        let status = try await Supervisor.run(launch, output: log, timeout: timeout)
        try? log?.close()
        if Task.isCancelled { throw CancellationError() }
        guard status == 0 else {
            throw ParlorError.message("sd-cli exited with status \(status). " + ServiceLog.tail(home: home, service: "image-sdcpp", lines: 4))
        }
    }
}

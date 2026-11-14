import Foundation

/// Every tunable that trades speed, memory, quality or tokens. Lives in `~/parlor/settings.json`
/// so the Models page, a text editor and an external Claude Code all change the same thing.
/// Missing keys fall back to defaults, so the file can be as small as `{"llm":{"keepLoadedMinutes":0}}`.
public struct ParlorSettings: Codable, Equatable, Sendable {
    public var llm = LLM()
    public var image = Image()
    public var loop = Loop()
    /// Share of physical memory the resident language models may use before the least recently used one is unloaded.
    public var memoryBudgetFraction = 0.6
    /// Prompt-cache slots kept per resident model; a request reuses the slot sharing the longest prefix.
    public var cacheSlotsPerModel = 4
    /// The app-hosted `parlor` MCP server that outside clients (Claude Code) and launched modules talk to.
    public var parlorServer = ParlorServer()
    /// Access tokens: names and hashes only; the secret is shown once when it is made.
    public var tokens: [APIToken] = []
    /// Module name → manifest hash the user approved.
    public var moduleApprovals: [String: String] = [:]

    public init() {}

    public struct ParlorServer: Codable, Equatable, Sendable {
        public var enabled = true
        public var port = 11501
        public init() {}
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? enabled
            port = try c.decodeIfPresent(Int.self, forKey: .port) ?? port
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        memoryBudgetFraction = try c.decodeIfPresent(Double.self, forKey: .memoryBudgetFraction) ?? memoryBudgetFraction
        cacheSlotsPerModel = try c.decodeIfPresent(Int.self, forKey: .cacheSlotsPerModel) ?? cacheSlotsPerModel
        parlorServer = try c.decodeIfPresent(ParlorServer.self, forKey: .parlorServer) ?? parlorServer
        tokens = try c.decodeIfPresent([APIToken].self, forKey: .tokens) ?? tokens
        moduleApprovals = try c.decodeIfPresent([String: String].self, forKey: .moduleApprovals) ?? moduleApprovals
        llm = try c.decodeIfPresent(LLM.self, forKey: .llm) ?? llm
        image = try c.decodeIfPresent(Image.self, forKey: .image) ?? image
        loop = try c.decodeIfPresent(Loop.self, forKey: .loop) ?? loop
    }

    /// Language model: speed (prompt cache, prefill), memory (KV quantisation, unload) and sampling.
    public struct LLM: Codable, Equatable, Sendable {
        /// Keep the attention cache between turns so only the new messages are processed.
        public var promptCache = true
        /// 4 or 8 to shrink the attention cache on long chats. Applies only with `promptCache` off.
        public var kvBits: Int? = nil
        public var kvGroupSize = 64
        public var quantizedKVStart = 1024
        /// Rolling attention window in tokens. Caps memory, forgets the oldest tokens. Disables the prompt cache.
        public var maxKVSize: Int? = nil
        /// Tokens processed per prefill step. Bigger is faster and uses more peak memory.
        public var prefillStepSize = 512
        public var topP = 1.0
        public var repetitionPenalty: Double? = nil
        /// Hard cap on one reply. nil for no cap.
        public var maxReplyTokens: Int? = 2048
        /// Unload the model after this many idle minutes. 0 keeps it loaded.
        public var keepLoadedMinutes = 30

        public init() {}
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            promptCache = try c.decodeIfPresent(Bool.self, forKey: .promptCache) ?? promptCache
            kvBits = try c.decodeIfPresent(Int.self, forKey: .kvBits)
            kvGroupSize = try c.decodeIfPresent(Int.self, forKey: .kvGroupSize) ?? kvGroupSize
            quantizedKVStart = try c.decodeIfPresent(Int.self, forKey: .quantizedKVStart) ?? quantizedKVStart
            maxKVSize = try c.decodeIfPresent(Int.self, forKey: .maxKVSize)
            prefillStepSize = try c.decodeIfPresent(Int.self, forKey: .prefillStepSize) ?? prefillStepSize
            topP = try c.decodeIfPresent(Double.self, forKey: .topP) ?? topP
            repetitionPenalty = try c.decodeIfPresent(Double.self, forKey: .repetitionPenalty)
            maxReplyTokens = c.contains(.maxReplyTokens) ? try c.decodeIfPresent(Int.self, forKey: .maxReplyTokens) : maxReplyTokens
            keepLoadedMinutes = try c.decodeIfPresent(Int.self, forKey: .keepLoadedMinutes) ?? keepLoadedMinutes
        }

        /// The prompt cache needs a plain, trimmable cache, so the two memory knobs turn it off.
        public var promptCacheActive: Bool { promptCache && kvBits == nil && maxKVSize == nil }
    }

    /// Image worker: precision, steps, size and idle behaviour.
    public struct Image: Codable, Equatable, Sendable {
        /// "bf16", "int8" or "int4". Lower is smaller and faster to load; int4 costs some detail.
        public var precision = "int8"
        /// nil uses the model's own default step count.
        public var fastSteps: Int? = nil
        public var detailedSteps: Int? = nil
        /// Square side in pixels for the two quality levels; portrait and landscape derive from it.
        public var fastSide = 512
        public var detailedSide = 768
        /// Stop the worker (freeing its memory) after this many idle seconds. 0 keeps it running.
        public var idleSeconds = 120

        public init() {}
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            precision = try c.decodeIfPresent(String.self, forKey: .precision) ?? precision
            fastSteps = try c.decodeIfPresent(Int.self, forKey: .fastSteps)
            detailedSteps = try c.decodeIfPresent(Int.self, forKey: .detailedSteps)
            fastSide = try c.decodeIfPresent(Int.self, forKey: .fastSide) ?? fastSide
            detailedSide = try c.decodeIfPresent(Int.self, forKey: .detailedSide) ?? detailedSide
            idleSeconds = try c.decodeIfPresent(Int.self, forKey: .idleSeconds) ?? idleSeconds
        }

        /// Width and height for a shape at a quality level, snapped to the 64 px grid the model wants.
        public func size(shape: String, detailed: Bool) -> (Int, Int) {
            let side = Double(max(256, detailed ? detailedSide : fastSide))
            func snap(_ v: Double) -> Int { max(256, Int(v / 64) * 64) }
            let square = snap(side)
            switch shape {
            case "portrait": return (snap(side * 0.875), snap(side * 1.25))
            case "landscape": return (snap(side * 1.25), snap(side * 0.875))
            default: return (square, square)
            }
        }
    }

    /// Agent loop, tool results and compaction: where prompt tokens are spent.
    public struct Loop: Codable, Equatable, Sendable {
        public var maxSteps = 8
        /// Longer tool results are cut to this many characters before the model sees them.
        public var toolResultChars = 6000
        public var toolTimeoutSeconds = 600
        /// Recent messages kept verbatim when older ones are summarised.
        public var keepRecent = 6
        public var summaryTokens = 500

        public init() {}
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            maxSteps = try c.decodeIfPresent(Int.self, forKey: .maxSteps) ?? maxSteps
            toolResultChars = try c.decodeIfPresent(Int.self, forKey: .toolResultChars) ?? toolResultChars
            toolTimeoutSeconds = try c.decodeIfPresent(Int.self, forKey: .toolTimeoutSeconds) ?? toolTimeoutSeconds
            keepRecent = try c.decodeIfPresent(Int.self, forKey: .keepRecent) ?? keepRecent
            summaryTokens = try c.decodeIfPresent(Int.self, forKey: .summaryTokens) ?? summaryTokens
        }
    }

    // MARK: file

    public static func load(_ url: URL) -> ParlorSettings {
        guard let d = try? Data(contentsOf: url), let s = try? JSONDecoder().decode(ParlorSettings.self, from: d) else { return ParlorSettings() }
        return s
    }

    public func save(_ url: URL) {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let d = try? enc.encode(self) else { return }
        if let old = try? Data(contentsOf: url), old == d { return }
        try? d.write(to: url, options: .atomic)
    }
}

/// What a request asks of the model besides the messages.
public struct LLMTuning: Sendable, Equatable {
    public var settings = ParlorSettings.LLM()
    /// Reuse and update the shared prompt cache. Side jobs (summaries) set this false so they never evict the chat.
    public var useCache = true
    public init(settings: ParlorSettings.LLM = .init(), useCache: Bool = true) { self.settings = settings; self.useCache = useCache }
}

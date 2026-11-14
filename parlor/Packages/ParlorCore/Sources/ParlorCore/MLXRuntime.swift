import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// In-process MLX language model runtime. Several models stay resident while they fit the memory
/// budget (least recently used leaves first), and each keeps a few prompt-cache slots.
///
/// Speed: a request takes the slot whose cached prompt shares the longest start with its own and only
/// processes the part that differs, so alternating chats or agents keep their caches. Memory: models
/// unload after an idle period, and the budget counts every resident model.
public actor MLXRuntime: LanguageModelRuntime {
    struct Resident {
        let container: ModelContainer
        let bytes: Int64
        let slots: SlotCache
        var lastUsed: Date
        var inFlight = 0
    }

    private let library: ModelLibrary
    private var residents: [String: Resident] = [:]
    private var keepAlive: TimeInterval = 0
    private var idleTask: Task<Void, Never>?
    private var budgetFraction = 0.6
    private var slotLimit = 4

    public init(library: ModelLibrary) { self.library = library }

    /// Most recently used resident model.
    public func loadedModel() async -> String? { residents.max { $0.value.lastUsed < $1.value.lastUsed }?.key }
    public func loadedModels() -> [String] { residents.keys.sorted() }
    public func residentBytes() -> Int64 { residents.values.reduce(0) { $0 + $1.bytes } }

    public func unload() async {
        idleTask?.cancel()
        residents = [:]
    }

    /// Minutes of idleness before a model is unloaded. 0 keeps it loaded.
    public func setKeepAlive(minutes: Int) {
        keepAlive = TimeInterval(max(0, minutes) * 60)
        scheduleSweep()
    }

    public func configure(budgetFraction: Double, slots: Int) {
        self.budgetFraction = min(0.95, max(0.1, budgetFraction))
        slotLimit = max(1, slots)
        for r in residents.values { r.slots.limit = slotLimit }
    }

    private func scheduleSweep() {
        idleTask?.cancel()
        guard keepAlive > 0, !residents.isEmpty else { return }
        let secs = keepAlive
        idleTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(secs * 1_000_000_000))
            if !Task.isCancelled { await self?.sweep() }
        }
    }

    private func sweep() {
        let now = Date()
        for (id, r) in residents where r.inFlight == 0 && now.timeIntervalSince(r.lastUsed) >= keepAlive { residents[id] = nil }
        scheduleSweep()
    }

    private func begin(_ id: String) { residents[id]?.inFlight += 1; residents[id]?.lastUsed = Date(); idleTask?.cancel() }
    private func end(_ id: String) { residents[id]?.inFlight -= 1; residents[id]?.lastUsed = Date(); scheduleSweep() }

    /// Unload least recently used idle models until `needed` more bytes fit the budget.
    private func makeRoom(for needed: Int64) {
        let budget = Int64(Double(ProcessInfo.processInfo.physicalMemory) * budgetFraction)
        while residentBytes() + needed > budget,
              let victim = residents.filter({ $0.value.inFlight == 0 }).min(by: { $0.value.lastUsed < $1.value.lastUsed }) {
            residents[victim.key] = nil
        }
    }

    private func ensureLoaded(_ id: String) async throws -> Resident {
        if var r = residents[id] { r.lastUsed = Date(); residents[id] = r; return r }
        guard let dir = library.installedDirectory(for: id) else { throw ParlorError.modelNotInstalled(id) }
        let bytes = library.installed(kind: .language).first { $0.repo == id }?.bytes ?? 0
        makeRoom(for: bytes)
        let c: ModelContainer
        do { c = try await loadModelContainer(from: dir, using: #huggingFaceTokenizerLoader()) }
        catch {
            throw ParlorError.message("Couldn't load \(id): \(error.localizedDescription) This usually means the weights are not an MLX-compatible conversion of a supported architecture.")
        }
        let r = Resident(container: c, bytes: bytes, slots: SlotCache(limit: slotLimit), lastUsed: Date())
        residents[id] = r
        return r
    }

    public nonisolated func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                var slots: SlotCache?
                do {
                    let r = try await self.ensureLoaded(model)
                    slots = r.slots
                    await self.begin(model)
                    try await Self.run(container: r.container, request: request, slots: r.slots, continuation: continuation)
                    await self.end(model)
                    continuation.finish()
                } catch {
                    slots?.resetAll()
                    await self.end(model)
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// One generation, serialised against the model by `perform`.
    private static func run(
        container: ModelContainer, request: LLMRequest, slots: SlotCache,
        continuation: AsyncThrowingStream<LLMEvent, Error>.Continuation
    ) async throws {
        let specs: [ToolSpec]? = request.tools.isEmpty ? nil : request.tools.map(Self.spec)
        let chat = request.messages.map(Self.chatMessage)
        var ctx: [String: any Sendable] = [:]
        ctx["enable_thinking"] = request.thinking
        let t = request.tuning.settings
        let reuse = request.tuning.useCache && t.promptCacheActive

        var params = GenerateParameters(maxTokens: request.maxTokens, temperature: Float(request.temperature))
        params.topP = Float(t.topP)
        if let rp = t.repetitionPenalty, rp > 1 { params.repetitionPenalty = Float(rp) }
        params.prefill.stepSize = max(32, t.prefillStepSize)
        if !reuse {
            if let bits = t.kvBits, [4, 8].contains(bits) {
                params.kvBits = bits; params.kvGroupSize = t.kvGroupSize; params.quantizedKVStart = t.quantizedKVStart
            }
            params.maxKVSize = t.maxKVSize
        }

        try await container.perform { context in
            let lm = try await context.processor.prepare(input: UserInput(chat: chat, tools: specs, additionalContext: ctx))
            var input = lm
            var cache: [KVCache]? = nil
            var cached = 0
            var all: [Int] = []
            var slot = -1
            if reuse, lm.image == nil, lm.video == nil, lm.audio == nil {
                all = lm.text.tokens.reshaped(-1).asArray(Int.self)
                // Try slots from longest shared prefix to shortest; always leave the final token to process.
                for (idx, shared) in slots.candidates(for: all) {
                    let common = min(shared, all.count - 1)
                    guard common >= 32, let c = slots.slots[idx].cache, canTrimPromptCache(c),
                          let off = c.first?.offset, off == slots.slots[idx].tokens.count else { continue }
                    if off > common { trimPromptCache(c, numTokens: off - common) }
                    slot = idx; cache = c; cached = common
                    input = LMInput(tokens: MLXArray(Array(all[common...])))
                    break
                }
                if slot < 0 { slot = slots.victim(); cache = try context.model.newCache(parameters: params) }
                slots.claim(slot, cache: cache)      // not valid again until this run finishes cleanly
            }

            let stream = try MLXLMCommon.generate(input: input, cache: cache, parameters: params, context: context, tools: specs)
            for await g in stream {
                if Task.isCancelled { if slot >= 0 { slots.reset(slot) }; return }
                switch g {
                case .chunk(let s): continuation.yield(.text(s))
                case .toolCall(let call):
                    let args = call.function.arguments.mapValues(Self.json)
                    continuation.yield(.toolCall(ToolCallRecord(
                        id: call.id ?? UUID().uuidString, name: call.function.name, arguments: .object(args))))
                case .info(let info):
                    continuation.yield(.stats(LLMStats(
                        promptTokens: info.promptTokenCount + cached,
                        completionTokens: info.generationTokenCount,
                        tokensPerSecond: info.tokensPerSecond, cachedTokens: cached)))
                case .rejectedToolCall: break
                }
            }
            if reuse, slot >= 0, let c = cache {
                // Drop the reply from the cache: the next prompt re-renders it through the chat template anyway.
                if let off = c.first?.offset, off > all.count, canTrimPromptCache(c) {
                    trimPromptCache(c, numTokens: off - all.count)
                }
                if c.first?.offset == all.count, canTrimPromptCache(c) { slots.finish(slot, tokens: all) } else { slots.reset(slot) }
            }
        }
    }

    static func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        var i = 0
        let n = min(a.count, b.count)
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }

    // MARK: conversions

    static func spec(_ t: ToolDef) -> ToolSpec {
        let params = (t.schema.any as? [String: Any]) ?? ["type": "object", "properties": [String: Any]()]
        let fn: [String: any Sendable] = [
            "name": t.wireName,
            "description": t.description,
            "parameters": Self.sendable(params),
        ]
        return ["type": "function", "function": fn]
    }

    static func sendable(_ v: Any) -> any Sendable {
        if let d = v as? [String: Any] { return d.mapValues { sendable($0) } as [String: any Sendable] }
        if let a = v as? [Any] { return a.map { sendable($0) } as [any Sendable] }
        if let s = v as? String { return s }
        if let b = v as? Bool { return b }
        if let i = v as? Int { return i }
        if let d = v as? Double { return d }
        return ""
    }

    static func chatMessage(_ m: LLMMessage) -> Chat.Message {
        switch m.role {
        case .system: return .system(m.text)
        case .user: return .user(m.text)
        case .assistant:
            let calls = m.toolCalls.map {
                ToolCall(function: .init(name: $0.name, arguments: $0.arguments.objectValue?.mapValues(Self.mlxValue) ?? [:]), id: $0.id)
            }
            return .assistant(m.text, toolCalls: calls.isEmpty ? nil : calls)
        case .tool: return .tool(m.text, id: m.toolCallID, name: m.toolName)
        }
    }

    static func json(_ v: JSONValue) -> JSON {
        switch v {
        case .null: return .null
        case .bool(let b): return .bool(b)
        case .int(let i): return .int(i)
        case .double(let d): return .double(d)
        case .string(let s): return .string(s)
        case .array(let a): return .array(a.map(json))
        case .object(let o): return .object(o.mapValues(json))
        }
    }

    static func mlxValue(_ v: JSON) -> JSONValue {
        switch v {
        case .null: return .null
        case .bool(let b): return .bool(b)
        case .int(let i): return .int(i)
        case .double(let d): return .double(d)
        case .string(let s): return .string(s)
        case .array(let a): return .array(a.map(mlxValue))
        case .object(let o): return .object(o.mapValues(mlxValue))
        }
    }
}

/// Prompt-cache slots for one resident model. Only touched inside `ModelContainer.perform`, which runs
/// one generation at a time, plus `resetAll` and `limit` from the runtime actor between runs.
final class SlotCache: @unchecked Sendable {
    struct Slot { var cache: [KVCache]?; var tokens: [Int] = []; var used = 0 }
    var slots: [Slot] = []
    private var tick = 0
    var limit: Int { didSet { while slots.count > max(1, limit) { evictOldest() } } }

    init(limit: Int) { self.limit = max(1, limit) }

    /// Valid slots (they hold a finished prompt), best shared prefix first.
    func candidates(for tokens: [Int]) -> [(Int, Int)] {
        slots.indices.compactMap { i in
            guard slots[i].cache != nil, !slots[i].tokens.isEmpty else { return nil }
            return (i, MLXRuntime.commonPrefix(slots[i].tokens, tokens))
        }.sorted { $0.1 > $1.1 }
    }

    /// An empty slot, a new one while under the limit, or the least recently used.
    func victim() -> Int {
        if let i = slots.firstIndex(where: { $0.cache == nil }) { return i }
        if slots.count < limit { slots.append(Slot(cache: nil)); return slots.count - 1 }
        return slots.indices.min { slots[$0].used < slots[$1].used } ?? 0
    }

    func claim(_ i: Int, cache: [KVCache]?) { tick += 1; slots[i].cache = cache; slots[i].tokens = []; slots[i].used = tick }
    func finish(_ i: Int, tokens: [Int]) { tick += 1; slots[i].tokens = tokens; slots[i].used = tick }
    func reset(_ i: Int) { guard slots.indices.contains(i) else { return }; slots[i].cache = nil; slots[i].tokens = [] }
    func resetAll() { for i in slots.indices { reset(i) } }
    private func evictOldest() {
        if let i = slots.indices.min(by: { slots[$0].used < slots[$1].used }) { slots.remove(at: i) }
    }
}

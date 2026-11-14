import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// In-process MLX language model runtime. Holds at most one model in memory;
/// switching agents to a different model unloads the old one first.
///
/// Speed: the attention cache from the last chat prompt is kept, and the next request only
/// processes the part of its prompt that differs (usually the newest messages). Memory: the
/// model unloads after an idle period.
public actor MLXRuntime: LanguageModelRuntime {
    private let library: ModelLibrary
    private var container: ModelContainer?
    private var loadedID: String?
    private let prefix = PrefixCache()
    private var keepAlive: TimeInterval = 0
    private var idleTask: Task<Void, Never>?
    private var inFlight = 0

    public init(library: ModelLibrary) { self.library = library }

    public func loadedModel() async -> String? { loadedID }

    public func unload() async {
        idleTask?.cancel()
        container = nil
        loadedID = nil
        prefix.reset()
    }

    /// Minutes of idleness before the model is unloaded. 0 keeps it loaded.
    public func setKeepAlive(minutes: Int) {
        keepAlive = TimeInterval(max(0, minutes) * 60)
        if inFlight == 0 { scheduleIdleUnload() }
    }

    private func scheduleIdleUnload() {
        idleTask?.cancel()
        guard keepAlive > 0, loadedID != nil else { return }
        let secs = keepAlive
        idleTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(secs * 1_000_000_000))
            if !Task.isCancelled { await self?.unloadIfIdle() }
        }
    }

    private func unloadIfIdle() async { if inFlight == 0 { await unload() } }

    private func begin() { inFlight += 1; idleTask?.cancel() }
    private func end() { inFlight -= 1; if inFlight == 0 { scheduleIdleUnload() } }

    private func ensureLoaded(_ id: String) async throws -> ModelContainer {
        if let container, loadedID == id { return container }
        container = nil
        loadedID = nil
        prefix.reset()
        guard let dir = library.installedDirectory(for: id) else { throw ParlorError.modelNotInstalled(id) }
        let c = try await loadModelContainer(from: dir, using: #huggingFaceTokenizerLoader())
        container = c
        loadedID = id
        return c
    }

    public nonisolated func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                await self.begin()
                do {
                    let container = try await self.ensureLoaded(model)
                    try await Self.run(container: container, request: request, prefix: self.prefix, continuation: continuation)
                    continuation.finish()
                } catch {
                    self.prefix.reset()
                    continuation.finish(throwing: error)
                }
                await self.end()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// One generation, serialised against the model by `perform`.
    private static func run(
        container: ModelContainer, request: LLMRequest, prefix: PrefixCache,
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
            if reuse, lm.image == nil, lm.video == nil, lm.audio == nil {
                all = lm.text.tokens.reshaped(-1).asArray(Int.self)
                if let c = prefix.cache, canTrimPromptCache(c), let off = c.first?.offset, off == prefix.tokens.count {
                    // Longest shared start, always leaving at least the final token to process.
                    let common = min(Self.commonPrefix(prefix.tokens, all), all.count - 1)
                    if common >= 32 {
                        if off > common { trimPromptCache(c, numTokens: off - common) }
                        cache = c
                        cached = common
                        input = LMInput(tokens: MLXArray(Array(all[common...])))
                    }
                }
                if cache == nil { cache = try context.model.newCache(parameters: params) }
                prefix.cache = cache
                prefix.tokens = []          // not valid again until this run finishes cleanly
            }

            let stream = try MLXLMCommon.generate(input: input, cache: cache, parameters: params, context: context, tools: specs)
            for await g in stream {
                if Task.isCancelled { prefix.reset(); return }
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
            if reuse, let c = cache {
                // Drop the reply from the cache: the next prompt re-renders it through the chat template anyway.
                if let off = c.first?.offset, off > all.count, canTrimPromptCache(c) {
                    trimPromptCache(c, numTokens: off - all.count)
                }
                if c.first?.offset == all.count, canTrimPromptCache(c) { prefix.tokens = all } else { prefix.reset() }
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

/// The attention cache of the last chat prompt and the tokens it covers. Only touched inside
/// `ModelContainer.perform`, which runs one generation at a time.
final class PrefixCache: @unchecked Sendable {
    var cache: [KVCache]?
    var tokens: [Int] = []
    func reset() { cache = nil; tokens = [] }
}

import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// In-process MLX language model runtime. Holds at most one model in memory;
/// switching agents to a different model unloads the old one first.
public actor MLXRuntime: LanguageModelRuntime {
    private let library: ModelLibrary
    private var container: ModelContainer?
    private var loadedID: String?

    public init(library: ModelLibrary) { self.library = library }

    public func loadedModel() async -> String? { loadedID }

    public func unload() async {
        container = nil
        loadedID = nil
    }

    private func ensureLoaded(_ id: String) async throws -> ModelContainer {
        if let container, loadedID == id { return container }
        container = nil
        loadedID = nil
        guard let dir = library.installedDirectory(for: id) else { throw ParlorError.modelNotInstalled(id) }
        let c = try await loadModelContainer(from: dir, using: #huggingFaceTokenizerLoader())
        container = c
        loadedID = id
        return c
    }

    public nonisolated func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let container = try await self.ensureLoaded(model)
                    let specs: [ToolSpec]? = request.tools.isEmpty ? nil : request.tools.map(Self.spec)
                    let chat = request.messages.map(Self.chatMessage)
                    var ctx: [String: any Sendable] = [:]
                    ctx["enable_thinking"] = request.thinking
                    let input = UserInput(chat: chat, tools: specs, additionalContext: ctx)
                    let lm = try await container.prepare(input: input)
                    var params = GenerateParameters(maxTokens: request.maxTokens, temperature: Float(request.temperature))
                    params.maxTokens = request.maxTokens
                    let stream = try await container.generate(input: lm, parameters: params, tools: specs)
                    for await g in stream {
                        if Task.isCancelled { break }
                        switch g {
                        case .chunk(let t): continuation.yield(.text(t))
                        case .toolCall(let call):
                            let args = call.function.arguments.mapValues(Self.json)
                            continuation.yield(.toolCall(ToolCallRecord(
                                id: call.id ?? UUID().uuidString, name: call.function.name,
                                arguments: .object(args))))
                        case .info(let info):
                            continuation.yield(.stats(LLMStats(
                                promptTokens: info.promptTokenCount,
                                completionTokens: info.generationTokenCount,
                                tokensPerSecond: info.tokensPerSecond)))
                        case .rejectedToolCall: break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
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

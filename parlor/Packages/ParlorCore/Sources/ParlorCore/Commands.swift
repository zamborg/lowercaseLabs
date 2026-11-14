import Foundation

public extension JSON {
    var arrayValue: [JSON]? { if case .array(let a) = self { return a }; return nil }
    var boolValue: Bool? { if case .bool(let b) = self { return b }; return nil }
    var doubleValue: Double? {
        switch self { case .double(let d): return d; case .int(let i): return Double(i); default: return nil }
    }
    var isNull: Bool { if case .null = self { return true }; return false }
    static func strings(_ s: [String]) -> JSON { .array(s.map { .string($0) }) }
}

public struct CommandError: LocalizedError {
    public var message: String
    public init(_ m: String) { message = m }
    public var errorDescription: String? { message }
}

/// One thing an agent, an outside client or the UI can do. The engine's own methods do the work; a verb is the
/// named, typed, tiered way to reach them.
public struct Verb: Sendable {
    public var name: String            // "agents.list"
    public var tier: Tier              // minimum caller tier
    public var destructive: Bool       // always confirmed when an agent or token calls it
    public var summary: String
    public var schema: JSON
    public var run: @MainActor @Sendable (Args, Caller) async throws -> JSON

    public var qualified: String { "parlor." + name }
}

/// Typed access to a verb's arguments. Unknown keys are refused so a typo never silently does nothing.
public struct Args: Sendable {
    public let raw: [String: JSON]
    public init(_ j: JSON) { raw = j.objectValue ?? [:] }

    func check(allowed: Set<String>) throws {
        let extra = Set(raw.keys).subtracting(allowed)
        if let k = extra.sorted().first { throw CommandError("Unknown argument \"\(k)\". Allowed: \(allowed.sorted().joined(separator: ", ")).") }
    }
    public func string(_ k: String) throws -> String {
        guard let v = raw[k]?.stringValue, !v.isEmpty else { throw CommandError("Missing \"\(k)\".") }
        return v
    }
    public func optString(_ k: String) -> String? { raw[k]?.stringValue }
    public func int(_ k: String, default d: Int? = nil) throws -> Int {
        if let v = raw[k]?.intValue { return v }
        if let d { return d }
        throw CommandError("Missing integer \"\(k)\".")
    }
    public func optInt(_ k: String) -> Int? { raw[k]?.intValue }
    public func bool(_ k: String, default d: Bool = false) -> Bool { raw[k]?.boolValue ?? d }
    public func strings(_ k: String) -> [String]? { raw[k]?.arrayValue?.compactMap(\.stringValue) }
}

/// The command layer: every control the app has, as named verbs behind one gate (tier check, destructive
/// confirmation, argument validation). The UI, the in-process `parlor` tool provider, the HTTP server and the CLI
/// all call `call`.
@MainActor
public final class CommandCenter {
    unowned let engine: ParlorEngine
    public private(set) var verbs: [String: Verb] = [:]

    init(engine: ParlorEngine) {
        self.engine = engine
        registerAgents(); registerConversations(); registerModels(); registerServices()
        registerSettings(); registerTools()
    }

    public var sorted: [Verb] { verbs.values.sorted { $0.name < $1.name } }

    func add(_ name: String, _ tier: Tier, destructive: Bool = false, _ summary: String,
             props: [(String, String, String)] = [], required: [String] = [],
             run: @escaping @MainActor @Sendable (Args, Caller) async throws -> JSON) {
        var p: [String: JSON] = [:]
        for (k, t, d) in props {
            if t == "strings" { p[k] = .object(["type": .string("array"), "items": .object(["type": .string("string")]), "description": .string(d)]) }
            else if t == "any" { p[k] = .object(["description": .string(d)]) }
            else if t == "object" { p[k] = .object(["type": .string("object"), "description": .string(d)]) }
            else { p[k] = .object(["type": .string(t), "description": .string(d)]) }
        }
        let schema: JSON = .object(["type": .string("object"), "properties": .object(p), "required": .strings(required), "additionalProperties": .bool(false)])
        verbs[name] = Verb(name: name, tier: tier, destructive: destructive, summary: summary, schema: schema, run: run)
    }

    /// The gate. Never throws; failures come back as error output the caller can read.
    public func call(_ name: String, _ arguments: JSON, caller: Caller) async -> ToolOutput {
        guard let v = verbs[name] else { return ToolOutput(text: "Unknown verb \(name).", isError: true) }
        guard caller.tier >= v.tier else {
            return ToolOutput(text: "\(v.qualified) needs tier \(v.tier.rawValue); this caller has \(caller.tier.rawValue).", isError: true)
        }
        if v.destructive, caller.mustConfirmDestructive {
            guard let ask = engine.confirm else {
                return ToolOutput(text: "\(v.qualified) is destructive and needs the user. Nobody is available to confirm it.", isError: true)
            }
            let ok = await ask("\(caller.name) wants to run \(v.qualified)\n\n" + arguments.text)
            guard ok else { return ToolOutput(text: "The user declined \(v.qualified).", isError: true) }
        }
        do {
            let r = try await v.run(Args(arguments), caller)
            return ToolOutput(text: r.pretty)
        } catch {
            return ToolOutput(text: error.localizedDescription, isError: true)
        }
    }

    /// Tools for the `parlor` provider and the HTTP server.
    public func toolDefs() -> [ToolDef] {
        sorted.map { ToolDef(server: "parlor", name: $0.name, description: "[\($0.tier.rawValue)\($0.destructive ? ", asks first" : "")] " + $0.summary, schema: $0.schema) }
    }
    public func tiers() -> [String: Tier] { Dictionary(uniqueKeysWithValues: verbs.values.map { ($0.qualified, $0.tier) }) }

    // MARK: JSON helpers

    func agentJSON(_ a: Agent, full: Bool = false) -> JSON {
        func r(_ d: Double) -> JSON { .double((d * 1000).rounded() / 1000) }
        var o: [String: JSON] = [
            "id": .string(a.id), "name": .string(a.name), "description": .string(a.summary), "kind": .string(a.kind.rawValue),
            "model": .string(a.model), "tools": .strings(a.tools), "tier": .string(a.tier.rawValue),
            "context": .int(a.contextTokens), "compact_at": r(a.compactAt), "temperature": r(a.temperature),
            "thinking": .bool(a.thinking), "color": .int(a.color),
        ]
        if let m = a.imageModel { o["image_model"] = .string(m) }
        let ov = a.overrides
        if let v = ov.topP { o["top_p"] = r(v) }
        if let v = ov.repetitionPenalty { o["repetition_penalty"] = r(v) }
        if let v = ov.maxReplyTokens { o["max_reply_tokens"] = .int(v) }
        if let v = ov.maxSteps { o["max_steps"] = .int(v) }
        if let v = ov.toolResultChars { o["tool_result_chars"] = .int(v) }
        if let v = ov.toolTimeoutSeconds { o["tool_timeout"] = .int(v) }
        if let v = ov.keepRecent { o["keep_recent"] = .int(v) }
        if let v = ov.summaryTokens { o["summary_tokens"] = .int(v) }
        if full { o["persona"] = .string(a.persona) }
        return .object(o)
    }

    func messageJSON(_ m: ChatMessage) -> JSON {
        var o: [String: JSON] = ["id": .string(m.id.uuidString), "role": .string(m.role.rawValue), "text": .string(m.text),
                                 "at": .string(ISO8601DateFormatter().string(from: m.createdAt))]
        if let s = m.speaker { o["speaker"] = .string(s) }
        if let t = m.toolName { o["tool"] = .string(t) }
        if let i = m.images { o["images"] = .strings(i) }
        return .object(o)
    }
}

public extension JSON {
    var pretty: String {
        guard let d = try? JSONSerialization.data(withJSONObject: any, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes, .fragmentsAllowed]),
              let s = String(data: d, encoding: .utf8) else { return text }
        return s
    }
}

/// The in-process `parlor` tool server agents see (gated by the agent's tier).
public final class CommandProvider: ToolProvider, @unchecked Sendable {
    public let id = "parlor"
    let center: CommandCenter
    init(center: CommandCenter) { self.center = center }

    public func listTools() async throws -> [ToolDef] { await center.toolDefs() }
    public func call(tool: String, arguments: JSON) async throws -> ToolOutput {
        await center.call(tool, arguments, caller: .local)
    }
    public func call(tool: String, arguments: JSON, caller: Caller) async throws -> ToolOutput {
        await center.call(tool, arguments, caller: caller)
    }
}

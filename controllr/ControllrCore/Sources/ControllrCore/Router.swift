import Foundation

public enum ExecutionOutcome: Sendable, Equatable {
    /// The module already ran inside the intent; speak this.
    case spoken(String)
    /// The router should open this URL (a background intent can't). `summary` is shown in the confirmation alert.
    case openURL(String, summary: String)
    /// A read-only result the model can use to finish the current turn.
    case toolResult(String)
}

/// Runs native modules inside the Decide intent. Implemented by the app (Contacts, EventKit, URLs).
public protocol ModuleExecutor: Sendable {
    /// Resolve or normalise arguments before a shortcut module is returned to the router
    /// (e.g. contact name -> phone number). Throw `ExecutionError.clarify` to ask the user something.
    func prepare(_ module: Module, arguments: [String: JSONValue]) async throws -> [String: JSONValue]
    func execute(_ module: Module, arguments: [String: JSONValue]) async throws -> ExecutionOutcome
}

public extension ModuleExecutor {
    func prepare(_ module: Module, arguments: [String: JSONValue]) async throws -> [String: JSONValue] { arguments }
}

public enum ExecutionError: Error, Sendable, Equatable {
    case clarify(String)
    case unavailable(String)
    case failed(String)
}

public struct RouteResult: Sendable {
    public let decision: Decision
    public let toolCall: ToolCall?
    public let latencyMs: Int
    public let provider: String
    public let model: String
    public let error: String?
    public let toolMessages: [ConversationMessage]

    public init(decision: Decision, toolCall: ToolCall?, latencyMs: Int, provider: String, model: String, error: String?,
                toolMessages: [ConversationMessage] = []) {
        self.decision = decision; self.toolCall = toolCall; self.latencyMs = latencyMs
        self.provider = provider; self.model = model; self.error = error
        self.toolMessages = toolMessages
    }
}

public enum ArgumentValidator {
    /// Returns a short problem description, or nil when `args` satisfy the module schema.
    public static func validate(_ module: Module, _ args: JSONValue) -> String? {
        guard case .object(let obj) = args else { return "arguments are not an object" }
        for p in module.params {
            guard let v = obj[p.name], v != .null else {
                if p.required { return "missing \(p.name)" }
                continue
            }
            switch p.type {
            case .string:
                guard let text = v.stringValue else { return "\(p.name) must be a string" }
                if p.required && text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "missing \(p.name)" }
            case .number: if v.numberValue == nil { return "\(p.name) must be a number" }
            case .integer: if v.numberValue.flatMap(Int.init(exactly:)) == nil { return "\(p.name) must be an integer" }
            case .bool: if v.boolValue == nil { return "\(p.name) must be a boolean" }
            case .oneOf(let allowed): if let s = v.stringValue, allowed.contains(s) {} else { return "\(p.name) must be one of \(allowed.joined(separator: "/"))" }
            }
        }
        let known = Set(module.params.map(\.name))
        if let extra = obj.keys.first(where: { !known.contains($0) }) { return "unexpected \(extra)" }
        return nil
    }
}

public struct Router: Sendable {
    public let provider: LLMProvider
    public let registry: Registry
    public let settings: RouterSettings
    public let executor: ModuleExecutor?

    public init(provider: LLMProvider, registry: Registry = .v1, settings: RouterSettings = .init(), executor: ModuleExecutor? = nil) {
        self.provider = provider; self.registry = registry; self.settings = settings; self.executor = executor
    }

    public var tools: [ToolSpec] {
        registry.modules.map { ToolSpec(name: $0.id, description: $0.description, parameters: $0.parametersSchema) }
    }

    public func route(_ transcript: String, now: Date = Date(), execute: Bool = true) async -> RouteResult {
        await route(messages: [.user(transcript)], now: now, execute: execute)
    }

    /// Run one user turn. Read-only tool results can go back to the model within the same turn.
    public func route(messages: [ConversationMessage], now: Date = Date(), execute: Bool = true) async -> RouteResult {
        let start = ContinuousClock.now
        let system = Prompt.system(now: now, settings: settings)
        let availableTools = tools
        var toolMessages: [ConversationMessage] = []
        func result(_ decision: Decision, _ call: ToolCall?, _ error: String? = nil) -> RouteResult {
            let elapsed = (ContinuousClock.now - start).components
            let ms = Int(elapsed.seconds * 1000 + elapsed.attoseconds / 1_000_000_000_000_000)
            return RouteResult(decision: decision, toolCall: call, latencyMs: ms, provider: provider.name,
                               model: provider.model, error: error, toolMessages: toolMessages)
        }

        for _ in 0..<4 {
            let call: ToolCall
            do {
                call = try await provider.route(system: system, messages: messages + toolMessages, tools: availableTools)
            } catch let e as ProviderError {
                return result(.speak(.error, Self.spokenError(e)), nil, "\(e)")
            } catch {
                return result(.speak(.error, "Something went wrong."), nil, "\(error)")
            }

            guard let module = registry.module(id: call.name) else {
                return result(.speak(.error, "I couldn't work out what to do."), call, "unknown tool \(call.name)")
            }
            if let problem = ArgumentValidator.validate(module, call.arguments) {
                if let param = module.params.first(where: { problem == "missing \($0.name)" }),
                   let question = param.missingQuestion {
                    return result(.speak(.clarify, question), call)
                }
                return result(.speak(.error, "I didn't catch that. Try again."), call, "invalid arguments: \(problem)")
            }
            let args = call.arguments.objectValue ?? [:]

            do {
                switch module.kind {
                case .builtin:
                    if module.id == "clarify" {
                        return result(.speak(.clarify, args["question"]?.stringValue ?? "Can you say that again?"), call)
                    }
                    return result(.answer(args["text"]?.stringValue ?? "", followUp: args["follow_up"]?.stringValue), call)
                case .shortcut:
                    let prepared = execute ? try await executor?.prepare(module, arguments: args) ?? args : args
                    var summaryArguments = args
                    if let name = prepared["name"] { summaryArguments["recipient"] = name }
                    let decision = Decision(action: .run, shortcut: module.id, input: prepared,
                                            confirm: module.requiresConfirmation, summary: module.renderSummary(summaryArguments))
                    return result(decision, call)
                case .native:
                    guard execute, let executor else {
                        return result(.speak(.done, "Would run: \(module.renderSummary(args))"), call)
                    }
                    switch try await executor.execute(module, arguments: args) {
                    case .spoken(let text):
                        return result(.speak(.done, text), call)
                    case .openURL(let url, let summary):
                        let id = module.requiresConfirmation ? Decision.openURLConfirmID : Decision.openURLID
                        return result(Decision(action: .run, shortcut: id, input: ["url": .string(url)],
                                               confirm: module.requiresConfirmation, summary: summary), call)
                    case .toolResult(let output):
                        toolMessages.append(.tool(call: call, output: output))
                    }
                }
            } catch ExecutionError.clarify(let question) {
                return result(.speak(.clarify, question), call)
            } catch ExecutionError.unavailable(let message) {
                return result(.speak(.error, message), call, message)
            } catch {
                return result(.speak(.error, "That didn't work."), call, "\(error)")
            }
        }
        return result(.speak(.error, "I couldn't finish looking that up. Try a more specific request."), nil, "tool limit reached")
    }

    static func spokenError(_ e: ProviderError) -> String {
        switch e {
        case .http(let status, _) where status == 401 || status == 403: return "Your API key was rejected."
        case .http(let status, _) where status == 429: return "The provider is rate limiting. Try again."
        case .transport: return "I couldn't reach the network."
        default: return "I couldn't work out what to do."
        }
    }
}

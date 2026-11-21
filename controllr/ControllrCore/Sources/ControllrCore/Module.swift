import Foundation

public enum ParamType: Sendable, Equatable {
    case string
    case number
    case integer
    case bool
    case oneOf([String])
}

public struct Param: Sendable {
    public let name: String
    public let type: ParamType
    public let description: String
    public let required: Bool
    public init(_ name: String, _ type: ParamType, _ description: String, required: Bool = true) {
        self.name = name; self.type = type; self.description = description; self.required = required
    }
}

public enum ModuleKind: Sendable, Equatable {
    /// Executed by the router shortcut (inline If branch). `shortcutName` is the router's module id label.
    case shortcut
    /// Executed by our app inside the Decide intent.
    case native
    /// Answered by the router/LLM directly (answer, clarify).
    case builtin
}

public struct Module: Sendable {
    public let id: String
    public let description: String
    public let kind: ModuleKind
    public let params: [Param]
    public let requiresConfirmation: Bool
    /// Spoken/alert summary template; `{name}` placeholders are filled from arguments.
    public let summary: String

    public init(id: String, description: String, kind: ModuleKind, params: [Param] = [],
                requiresConfirmation: Bool = false, summary: String) {
        self.id = id; self.description = description; self.kind = kind; self.params = params
        self.requiresConfirmation = requiresConfirmation; self.summary = summary
    }

    /// JSON Schema for this module's arguments.
    public var parametersSchema: JSONValue {
        var props: [String: JSONValue] = [:]
        for p in params {
            var s: [String: JSONValue] = ["description": .string(p.description)]
            switch p.type {
            case .string: s["type"] = .string("string")
            case .number: s["type"] = .string("number")
            case .integer: s["type"] = .string("integer")
            case .bool: s["type"] = .string("boolean")
            case .oneOf(let values):
                s["type"] = .string("string")
                s["enum"] = .array(values.map { .string($0) })
            }
            props[p.name] = .object(s)
        }
        return .object([
            "type": .string("object"),
            "properties": .object(props),
            "required": .array(params.filter(\.required).map { .string($0.name) }),
            "additionalProperties": .bool(false),
        ])
    }

    public func renderSummary(_ args: [String: JSONValue]) -> String {
        var out = summary
        for (k, v) in args {
            let text = v.stringValue ?? v.numberValue.map { $0 == $0.rounded() ? String(Int($0)) : String($0) } ?? v.boolValue.map(String.init) ?? ""
            out = out.replacingOccurrences(of: "{\(k)}", with: text)
        }
        return out
    }
}

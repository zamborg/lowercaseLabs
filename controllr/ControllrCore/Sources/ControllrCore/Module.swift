import Foundation

public enum ParamType: Sendable, Equatable {
    case string
    case number
    case integer
    case bool
    case oneOf([String])
}

public struct Param: Decodable, Sendable {
    public let name: String
    public let type: ParamType
    public let description: String
    public let required: Bool
    public let missingQuestion: String?
    private enum CodingKeys: String, CodingKey { case name, type, description, required, values, missingQuestion }

    public init(from decoder: Decoder) throws {
        let fields = try decoder.container(keyedBy: CodingKeys.self)
        name = try fields.decode(String.self, forKey: .name)
        description = try fields.decode(String.self, forKey: .description)
        required = try fields.decode(Bool.self, forKey: .required)
        missingQuestion = try fields.decodeIfPresent(String.self, forKey: .missingQuestion)
        switch try fields.decode(String.self, forKey: .type) {
        case "string": type = .string
        case "number": type = .number
        case "integer": type = .integer
        case "bool": type = .bool
        case "oneOf": type = .oneOf(try fields.decode([String].self, forKey: .values))
        default: throw DecodingError.dataCorruptedError(forKey: .type, in: fields, debugDescription: "Unknown parameter type")
        }
    }
}

public enum ModuleKind: String, Decodable, Sendable, Equatable {
    /// Executed by the router shortcut (inline If branch). `shortcutName` is the router's module id label.
    case shortcut
    /// Executed by our app inside the Decide intent.
    case native
    /// Conversation tool handled by the router.
    case builtin
    /// Plain assistant replies, included in the eval suite.
    case response
}

public enum ModuleOperation: String, Decodable, Sendable {
    case read, change, handoff
}

public enum ModulePermission: String, Decodable, Sendable {
    case calendar, contacts, reminders, shortcuts, timers
}

public struct ActionEval: Decodable, Sendable {
    public let utterance: String
    public let arguments: [String: JSONValue]?
    public let history: [Turn]?

    public struct Turn: Decodable, Sendable {
        public enum Role: String, Decodable, Sendable { case user, assistant }

        public let role: Role
        public let text: String

        public var message: ConversationMessage {
            switch role {
            case .user: .user(text)
            case .assistant: .assistant(text)
            }
        }
    }
}

public struct Module: Decodable, Sendable {
    public let id: String
    public let description: String
    public let kind: ModuleKind
    public let params: [Param]
    public let requiresConfirmation: Bool
    /// Spoken/alert summary template; `{name}` placeholders are filled from arguments.
    public let summary: String

    public let shortcutHandler: String?
    public let supportedSeconds: [Int]?
    public let operation: ModuleOperation?
    public let permissions: [ModulePermission]?
    public let evals: [ActionEval]

    public var hasConfirmationSetting: Bool {
        (kind == .native || kind == .shortcut) && operation != .read
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
            let text = v.stringValue ?? v.numberValue.map { Int(exactly: $0).map(String.init) ?? String($0) } ?? v.boolValue.map(String.init) ?? ""
            out = out.replacingOccurrences(of: "{\(k)}", with: text)
        }
        return out
    }
}

import Foundation
import Security

public struct EndpointModel: Sendable, Equatable {
    public var provider: String
    public var id: String
    /// What an agent file's `model:` holds.
    public var qualified: String { "\(provider)/\(id)" }
}

/// API keys live in the Keychain; `PARLOR_KEY_<NAME>` in the environment wins, for scripts and tests.
public enum KeyStore {
    static func envName(_ ref: String) -> String {
        "PARLOR_KEY_" + String(ref.uppercased().map { $0.isLetter || $0.isNumber ? $0 : "_" })
    }

    public static func get(_ ref: String) -> String? {
        if let v = ProcessInfo.processInfo.environment[envName(ref)], !v.isEmpty { return v }
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "parlor",
                                kSecAttrAccount as String: ref, kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        var out: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data else { return nil }
        return String(data: d, encoding: .utf8)
    }

    public static func has(_ ref: String) -> Bool { get(ref) != nil }

    @discardableResult
    public static func set(_ ref: String, _ value: String) -> Bool {
        let base: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "parlor", kSecAttrAccount as String: ref]
        SecItemDelete(base as CFDictionary)
        var add = base
        add[kSecValueData as String] = Data(value.utf8)
        return SecItemAdd(add as CFDictionary, nil) == errSecSuccess
    }

    public static func remove(_ ref: String) {
        SecItemDelete([kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "parlor", kSecAttrAccount as String: ref] as CFDictionary)
    }
}


// MARK: - OpenAI wire format

/// Translation between parlor's request types and OpenAI chat-completions JSON, in both directions:
/// the provider client writes requests and reads chunks, the runtime API reads requests and writes chunks.
enum OpenAIWire {
    /// Tool names reach OpenAI-style callers exactly as they wrote them; inside parlor they carry this server prefix.
    static let externalServer = "ext"

    static func toolSpecs(_ tools: [ToolDef]) -> [JSON] {
        tools.map { t in
            .object(["type": .string("function"), "function": .object([
                "name": .string(t.wireName), "description": .string(t.description), "parameters": t.schema])])
        }
    }

    static func messages(_ msgs: [LLMMessage]) -> [JSON] {
        msgs.map { m in
            switch m.role {
            case .system: return .object(["role": .string("system"), "content": .string(m.text)])
            case .user: return .object(["role": .string("user"), "content": .string(m.text)])
            case .assistant:
                var o: [String: JSON] = ["role": .string("assistant"), "content": m.text.isEmpty && !m.toolCalls.isEmpty ? .null : .string(m.text)]
                if !m.toolCalls.isEmpty {
                    o["tool_calls"] = .array(m.toolCalls.map { c in
                        .object(["id": .string(c.id), "type": .string("function"),
                                 "function": .object(["name": .string(c.name), "arguments": .string(c.arguments.text)])])
                    })
                }
                return .object(o)
            case .tool:
                return .object(["role": .string("tool"), "tool_call_id": .string(m.toolCallID ?? ""), "content": .string(m.text)])
            }
        }
    }

    // Server side: an OpenAI request into parlor's types.

    static func parseTools(_ v: JSON?) -> [ToolDef] {
        guard case .array(let a)? = v else { return [] }
        return a.compactMap { t in
            guard let f = t.objectValue?["function"]?.objectValue, let name = f["name"]?.stringValue else { return nil }
            return ToolDef(server: externalServer, name: name, description: f["description"]?.stringValue ?? "",
                           schema: f["parameters"] ?? .object(["type": .string("object"), "properties": .object([:])]))
        }
    }

    static func text(of content: JSON?) -> String {
        switch content {
        case .string(let s)?: return s
        case .array(let parts)?:
            return parts.compactMap { p -> String? in
                guard let o = p.objectValue, o["type"]?.stringValue == "text" || o["type"] == nil else { return nil }
                return o["text"]?.stringValue
            }.joined()
        default: return ""
        }
    }

    static func parseMessages(_ v: JSON?) -> [LLMMessage]? {
        guard case .array(let a)? = v else { return nil }
        var names: [String: String] = [:]
        var out: [LLMMessage] = []
        for item in a {
            guard let o = item.objectValue, let role = o["role"]?.stringValue else { return nil }
            switch role {
            case "system", "developer": out.append(LLMMessage(role: .system, text: text(of: o["content"])))
            case "user": out.append(LLMMessage(role: .user, text: text(of: o["content"])))
            case "assistant":
                var calls: [ToolCallRecord] = []
                if case .array(let tcs)? = o["tool_calls"] {
                    for tc in tcs {
                        guard let t = tc.objectValue, let f = t["function"]?.objectValue, let n = f["name"]?.stringValue else { continue }
                        let args: JSON
                        switch f["arguments"] {
                        case .string(let s)?: args = JSON.parse(s) ?? .object([:])
                        case let j?: args = j
                        default: args = .object([:])
                        }
                        let id = t["id"]?.stringValue ?? UUID().uuidString
                        names[id] = n
                        calls.append(ToolCallRecord(id: id, name: externalServer + "__" + n, arguments: args))
                    }
                }
                out.append(LLMMessage(role: .assistant, text: text(of: o["content"]), toolCalls: calls))
            case "tool":
                let id = o["tool_call_id"]?.stringValue
                out.append(LLMMessage(role: .tool, text: text(of: o["content"]), toolCallID: id,
                                      toolName: (o["name"]?.stringValue ?? id.flatMap { names[$0] }).map { externalServer + "__" + $0 }))
            default: return nil
            }
        }
        return out
    }

    /// The name an OpenAI-style caller gave the tool.
    static func externalName(_ wire: String) -> String {
        let p = externalServer + "__"
        return wire.hasPrefix(p) ? String(wire.dropFirst(p.count)) : wire
    }

    static func encode(_ j: JSON) -> Data { (try? JSONEncoder().encode(j)) ?? Data() }
}


import CryptoKit
import Foundation

/// `tools/<id>/tool.json`: a tool server (MCP over stdio or http) and how to run it. A tool whose program lives in its own
/// folder is what used to be a module; there is no separate manifest.
public struct ToolSpec: Noun, Equatable {
    public static let kind = "tool"
    public var version: Int = 1
    public var id: String
    public var title: String
    public var description: String = ""
    /// `stdio` (a program) or `http` (a running server at `url`).
    public var transport: String
    public var command: String?
    public var args: [String] = []
    public var env: [String: String] = [:]
    public var cwd: String?
    public var url: String?
    /// `always` starts with the app; `onDemand` starts on first use and stops after `idleMinutes`.
    public var lifecycle: String = "always"
    public var idleMinutes: Int?
    /// How much of parlor itself the tool may control.
    public var tier: String = "none"
    public var enabled: Bool = true
    /// A JSON Schema (object with `properties`) rendered as a form.
    public var settings: JSON?
    /// What the user chose in that form.
    public var values: [String: JSON] = [:]

    enum CodingKeys: String, CodingKey { case version, id, title, description, transport, command, args, env, cwd, url, lifecycle, idleMinutes, tier, enabled, settings, values }

    public init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        version = try c.decodeIfPresent(Int.self, forKey: .version) ?? 1
        id = try c.decode(String.self, forKey: .id)
        title = try c.decodeIfPresent(String.self, forKey: .title) ?? id
        description = try c.decodeIfPresent(String.self, forKey: .description) ?? ""
        command = try c.decodeIfPresent(String.self, forKey: .command)
        args = try c.decodeIfPresent([String].self, forKey: .args) ?? []
        env = try c.decodeIfPresent([String: String].self, forKey: .env) ?? [:]
        cwd = try c.decodeIfPresent(String.self, forKey: .cwd)
        url = try c.decodeIfPresent(String.self, forKey: .url)
        transport = try c.decodeIfPresent(String.self, forKey: .transport) ?? (url != nil ? "http" : "stdio")
        lifecycle = try c.decodeIfPresent(String.self, forKey: .lifecycle) ?? "always"
        idleMinutes = try c.decodeIfPresent(Int.self, forKey: .idleMinutes)
        tier = try c.decodeIfPresent(String.self, forKey: .tier) ?? "none"
        enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
        settings = try c.decodeIfPresent(JSON.self, forKey: .settings)
        values = try c.decodeIfPresent([String: JSON].self, forKey: .values) ?? [:]
    }

    public static func check(_ d: [String: JSON]) -> [Issue] {
        var out: [Issue] = []
        if let id = d["id"]?.stringValue, id.range(of: "^[A-Za-z0-9_-]+$", options: .regularExpression) == nil {
            out.append(Issue(path: "id", message: "ids use letters, digits, - and _"))
        }
        let t = d["transport"]?.stringValue ?? (d["url"] != nil ? "http" : "stdio")
        if !["stdio", "http"].contains(t) { out.append(Issue(path: "transport", message: "stdio or http")) }
        if t == "stdio", (d["command"]?.stringValue ?? "").isEmpty { out.append(Issue(path: "command", message: "a stdio tool needs a command")) }
        if t == "http", (d["url"]?.stringValue ?? "").isEmpty { out.append(Issue(path: "url", message: "an http tool needs a url")) }
        if let l = d["lifecycle"]?.stringValue, !["always", "onDemand"].contains(l) { out.append(Issue(path: "lifecycle", message: "always or onDemand")) }
        if let t = d["tier"]?.stringValue, Tier(rawValue: t) == nil {
            out.append(Issue(path: "tier", message: "unknown tier \"\(t)\"", hint: "one of " + Tier.allCases.map(\.rawValue).joined(separator: ", ")))
        }
        return out
    }

    /// The keys that decide what runs. Changing one needs the user's yes again.
    public static let codeKeys = ["transport", "command", "args", "env", "cwd", "url", "tier"]

    public var tierValue: Tier { Tier(rawValue: tier) ?? .none }
}

public struct ToolInfo: Identifiable, Equatable, Sendable {
    public var id: String
    public var dir: URL
    public var spec: ToolSpec?
    /// Why the file did not load, as one line.
    public var error: String?
    /// Fingerprint of the keys that decide what runs.
    public var hash: String
    public var approved: Bool

    /// A property of the settings schema.
    public struct Field: Identifiable, Sendable { public var key: String; public var type: String; public var title: String; public var def: JSON?; public var id: String { key } }

    public var fields: [Field] {
        guard let props = spec?.settings?.objectValue?["properties"]?.objectValue else { return [] }
        return props.keys.sorted().map { k in
            let p = props[k]?.objectValue ?? [:]
            return Field(key: k, type: p["type"]?.stringValue ?? "string", title: p["description"]?.stringValue ?? k, def: p["default"])
        }
    }

    /// Stored values over schema defaults.
    public var effectiveValues: [String: JSON] {
        var v: [String: JSON] = [:]
        for f in fields { if let d = f.def { v[f.key] = d } }
        for (k, x) in spec?.values ?? [:] { v[k] = x }
        return v
    }
}

public enum Tools {
    /// Fingerprint of what a tool runs, from its raw file.
    public static func hash(_ raw: [String: JSON]) -> String {
        var o: [String: JSON] = [:]
        for k in ToolSpec.codeKeys { if let v = raw[k] { o[k] = v } }
        let data = Data(JSON.object(o).pretty.utf8)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public static func scan(store: NounStore<ToolSpec>, home: ParlorHome, approvals: [String: String]) -> [ToolInfo] {
        store.scan().map { e in
            let dir = home.toolsDir.appendingPathComponent(e.id)
            let h = hash(e.raw)
            var info = ToolInfo(id: e.id, dir: dir, spec: e.value, error: nil, hash: h, approved: approvals[e.id] == h)
            if e.value == nil { info.error = e.issues.first.map { "tool.json: \($0.message)" } ?? "tool.json does not load" }
            else if let bad = e.issues.errors.first { info.error = "tool.json: \(bad.path.isEmpty ? "" : bad.path + " ")\(bad.message)"; info.spec = nil }
            return info
        }
    }

    /// How to launch an approved tool.
    public static func config(for t: ToolInfo, parlorURL: String?, token: String?) -> MCPServerConfig? {
        guard let s = t.spec else { return nil }
        var env: [String: String] = s.env
        env["PARLOR_TOOL_SETTINGS"] = JSON.object(t.effectiveValues).pretty.replacingOccurrences(of: "\n", with: "")
        env["PARLOR_TOOL_DIR"] = t.dir.path
        if let parlorURL, let token { env["PARLOR_URL"] = parlorURL; env["PARLOR_TOKEN"] = token }
        // The helper `parlor_tool.py` lives next to the tool folders.
        env["PYTHONPATH"] = t.dir.deletingLastPathComponent().path
        env["PYTHONUNBUFFERED"] = "1"
        let cwd = s.cwd.map { ($0 as NSString).expandingTildeInPath } ?? t.dir.path
        return MCPServerConfig(command: s.transport == "stdio" ? s.command : nil, args: s.args, env: env, cwd: cwd,
                               url: s.transport == "http" ? s.url : nil, disabled: s.enabled ? nil : true,
                               lifecycle: s.lifecycle, idleMinutes: s.idleMinutes)
    }
}

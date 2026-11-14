import CryptoKit
import Foundation

/// `modules/<name>/module.json`: what a module is, what it needs, how to launch it.
public struct ModuleManifest: Codable, Equatable, Sendable {
    public struct Needs: Codable, Equatable, Sendable {
        public var network: Bool?
        public var files: String?
        public var tier: String?
        public init() {}
    }
    public var version: Int?
    public var name: String
    public var description: String?
    public var command: String
    public var args: [String]?
    public var needs: Needs?
    /// JSON Schema (object with `properties`) rendered as a form.
    public var settings: JSON?
    public var lifecycle: String?

    public var tier: Tier { Tier(rawValue: needs?.tier ?? "none") ?? .none }
}

public struct ModuleInfo: Identifiable, Equatable, Sendable {
    public var name: String
    public var dir: URL
    public var manifest: ModuleManifest?
    /// Why the manifest could not be read.
    public var error: String?
    public var hash: String
    public var approved: Bool
    public var values: [String: JSON]
    public var id: String { name }

    /// A property of the settings schema.
    public struct Field: Identifiable, Sendable { public var key: String; public var type: String; public var title: String; public var def: JSON?; public var id: String { key } }

    public var fields: [Field] {
        guard let props = manifest?.settings?.objectValue?["properties"]?.objectValue else { return [] }
        return props.keys.sorted().map { k in
            let p = props[k]?.objectValue ?? [:]
            return Field(key: k, type: p["type"]?.stringValue ?? "string", title: p["description"]?.stringValue ?? k, def: p["default"])
        }
    }
}

public enum Modules {
    public static func scan(home: ParlorHome, approvals: [String: String]) -> [ModuleInfo] {
        let fm = FileManager.default
        var out: [ModuleInfo] = []
        for d in ((try? fm.contentsOfDirectory(atPath: home.modulesDir.path)) ?? []).sorted() {
            let dir = home.modulesDir.appendingPathComponent(d)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else { continue }
            let file = dir.appendingPathComponent("module.json")
            guard let data = try? Data(contentsOf: file) else { continue }
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            var info = ModuleInfo(name: d, dir: dir, manifest: nil, error: nil, hash: hash, approved: false, values: [:])
            do {
                let m = try JSONDecoder().decode(ModuleManifest.self, from: data)
                guard m.name == d else { throw ParlorError.message("\"name\" is \"\(m.name)\" but the folder is \"\(d)\".") }
                guard d.range(of: "^[A-Za-z0-9_-]+$", options: .regularExpression) != nil else { throw ParlorError.message("Module names use letters, digits, - and _.") }
                if let l = m.lifecycle, !["always", "onDemand"].contains(l) { throw ParlorError.message("\"lifecycle\" is always or onDemand.") }
                info.manifest = m
                info.approved = approvals[d] == hash
                info.values = loadValues(info)
            } catch let DecodingError.keyNotFound(k, _) { info.error = "module.json is missing \"\(k.stringValue)\"."
            } catch { info.error = "module.json: \(error.localizedDescription)" }
            out.append(info)
        }
        return out
    }

    static func settingsURL(_ m: ModuleInfo) -> URL { m.dir.appendingPathComponent("settings.json") }

    /// Stored values over schema defaults.
    static func loadValues(_ m: ModuleInfo) -> [String: JSON] {
        var v: [String: JSON] = [:]
        for f in m.fields { if let d = f.def { v[f.key] = d } }
        if let d = try? Data(contentsOf: settingsURL(m)), let o = try? JSONDecoder().decode(JSON.self, from: d).objectValue {
            for (k, x) in o { v[k] = x }
        }
        return v
    }

    public static func saveValues(_ m: ModuleInfo, _ values: [String: JSON]) {
        guard let d = try? JSONSerialization.data(withJSONObject: JSON.object(values).any, options: [.prettyPrinted, .sortedKeys]) else { return }
        try? d.write(to: settingsURL(m), options: .atomic)
    }

    /// How to launch an approved module.
    public static func config(for m: ModuleInfo, parlorURL: String?, token: String?) -> MCPServerConfig? {
        guard let man = m.manifest else { return nil }
        var env: [String: String] = [:]
        env["PARLOR_MODULE_SETTINGS"] = JSON.object(m.values).pretty.replacingOccurrences(of: "\n", with: "")
        env["PARLOR_MODULE_DIR"] = m.dir.path
        if let parlorURL, let token { env["PARLOR_URL"] = parlorURL; env["PARLOR_TOKEN"] = token }
        return MCPServerConfig(command: man.command, args: man.args, env: env, cwd: m.dir.path, lifecycle: man.lifecycle)
    }
}

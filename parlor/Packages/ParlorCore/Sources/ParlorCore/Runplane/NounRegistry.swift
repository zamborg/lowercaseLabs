import Foundation

/// One entry per noun: where it lives, how it is laid out, and whether it runs code. A new noun is one entry here plus
/// its struct and schema; the watcher, `schema.get` and the Library read this list.
public struct NounKind: Sendable {
    public var id: String
    public var folder: String
    public var layout: NounLayout
    public var runsCode: Bool
    /// Seeded files carry a `managed` marker and are read-only to verbs.
    public var canBeSeeded: Bool

    public static let engine = NounKind(id: "engine", folder: "engines", layout: .folder("engine.json"), runsCode: true, canBeSeeded: true)
    public static let host = NounKind(id: "host", folder: "hosts", layout: .file, runsCode: false, canBeSeeded: false)
    public static let harness = NounKind(id: "harness", folder: "harnesses", layout: .folder("harness.json"), runsCode: true, canBeSeeded: true)
    public static let agent = NounKind(id: "agent", folder: "agents", layout: .folder("agent.json"), runsCode: false, canBeSeeded: false)
    public static let tool = NounKind(id: "tool", folder: "tools", layout: .folder("tool.json"), runsCode: true, canBeSeeded: false)
}

public enum NounRegistry {
    public static let all: [NounKind] = [.agent, .host, .harness, .engine, .tool]
    public static var ids: [String] { all.map(\.id) }
    public static func kind(_ id: String) -> NounKind? { all.first { $0.id == id } }

    /// The nouns whose folders contain any of these changed paths. Noise (bytecode caches, bench results, Finder files) is ignored.
    public static func folders(changed paths: [String], home: ParlorHome) -> Set<String> {
        let roots = [home.root.path, ParlorEngine.realPath(home.root.path)]
        var out: Set<String> = []
        for p in paths {
            if p.contains("/__pycache__") || p.hasSuffix(".bench.json") || p.hasSuffix(".DS_Store") || p.hasSuffix("~") { continue }
            for r in roots where p.hasPrefix(r + "/") {
                let first = p.dropFirst(r.count + 1).split(separator: "/").first.map(String.init) ?? ""
                if let k = all.first(where: { $0.folder == first }) { out.insert(k.id) }
            }
        }
        return out
    }
}

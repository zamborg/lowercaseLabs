import Foundation

/// An engine serving a model at the gateway. This is the file, `hosts/<id>.json`.
public struct HostSpec: Noun, Equatable {
    public static let kind = "host"
    public var version: Int = 1
    public var id: String
    public var engine: String
    /// The model repo (for engines that serve local weights).
    public var model: String?
    /// How the model runs: placeholders in the engine's launch (context, GPU layers, sampler, steps).
    public var settings: [String: JSON] = [:]
    /// How it is asked by default: sampling parameters used when an agent's config is silent.
    public var defaults: [String: JSON] = [:]
    public var enabled: Bool = true
    /// Remote hosts: where to talk, the Keychain item holding the key, and the models to offer (blank asks the server).
    public var url: String?
    public var keyRef: String?
    public var models: [String]?

    enum CodingKeys: String, CodingKey { case version, id, engine, model, settings, defaults, enabled, url, keyRef, models }

    public init(id: String, engine: String, model: String? = nil, settings: [String: JSON] = [:], defaults: [String: JSON] = [:],
                enabled: Bool = true, url: String? = nil, keyRef: String? = nil, models: [String]? = nil) {
        self.id = id; self.engine = engine; self.model = model; self.settings = settings; self.defaults = defaults
        self.enabled = enabled; self.url = url; self.keyRef = keyRef; self.models = models
    }

    public init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        version = try c.decodeIfPresent(Int.self, forKey: .version) ?? 1
        id = try c.decode(String.self, forKey: .id)
        engine = try c.decode(String.self, forKey: .engine)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        settings = try c.decodeIfPresent([String: JSON].self, forKey: .settings) ?? [:]
        defaults = try c.decodeIfPresent([String: JSON].self, forKey: .defaults) ?? [:]
        enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
        url = try c.decodeIfPresent(String.self, forKey: .url)
        keyRef = try c.decodeIfPresent(String.self, forKey: .keyRef)
        models = try c.decodeIfPresent([String]?.self, forKey: .models) ?? nil
    }

    public func encode(to e: Encoder) throws {
        var c = e.container(keyedBy: CodingKeys.self)
        try c.encode(version, forKey: .version); try c.encode(id, forKey: .id); try c.encode(engine, forKey: .engine)
        try c.encodeIfPresent(model, forKey: .model)
        try c.encode(settings, forKey: .settings); try c.encode(defaults, forKey: .defaults)
        try c.encode(enabled, forKey: .enabled)
        try c.encodeIfPresent(url, forKey: .url); try c.encodeIfPresent(keyRef, forKey: .keyRef); try c.encodeIfPresent(models, forKey: .models)
    }
}

/// A host as the rest of the app sees it: explicit (a file) or auto (computed from an installed model).
public struct HostInfo: Identifiable, Equatable, Sendable {
    public var spec: HostSpec
    public var isAuto: Bool
    public var id: String { spec.id }
}

public enum HostRegistry {
    /// Explicit hosts plus an auto host for every installed model whose format has exactly one engine.
    /// An auto host's id is `slug(shortName)`, with `-2`, `-3` for collisions assigned in sorted repo order.
    /// A file with the same id wins. Auto hosts are never written.
    public static func build(explicit: [HostSpec], installed: [InstalledModel], engines: [EngineManifest]) -> [HostInfo] {
        var out = explicit.map { HostInfo(spec: $0, isAuto: false) }
        var taken = Set(explicit.map(\.id))
        let explicitModels = Set(explicit.compactMap { h in h.model.map { "\(h.engine)|\($0)" } })
        for m in installed.sorted(by: { $0.repo < $1.repo }) {
            let matching = engines.filter { $0.formats.contains(m.format.rawValue) }
            guard matching.count == 1, let engine = matching.first else { continue }
            // A model that already has an explicit host on its engine keeps just that host in pickers.
            let base = Agent.slug(m.shortName).isEmpty ? "model" : Agent.slug(m.shortName)
            var id = base, n = 2
            while taken.contains(id) { id = "\(base)-\(n)"; n += 1 }
            if explicitModels.contains("\(engine.id)|\(m.repo)"), explicit.contains(where: { $0.id == base }) { continue }
            taken.insert(id)
            out.append(HostInfo(spec: HostSpec(id: id, engine: engine.id, model: m.repo), isAuto: true))
        }
        return out.sorted { $0.id < $1.id }
    }
}

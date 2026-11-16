import Foundation

public enum HostStatus: Equatable, Sendable {
    case ready, running, starting, off, missingModel
    case missingBinary(String)
    case unknownEngine
    case failed(String)

    public var label: String {
        switch self {
        case .ready: return "ready"
        case .running: return "running"
        case .starting: return "starting"
        case .off: return "off"
        case .missingModel: return "missing model"
        case .missingBinary(let hint): return hint.isEmpty ? "missing binary" : "missing binary (\(hint))"
        case .unknownEngine: return "unknown engine"
        case .failed(let why): return "failed (\(why))"
        }
    }

    /// Can a request be served now (possibly after starting it)?
    public var usable: Bool {
        switch self { case .ready, .running, .starting: return true; default: return false }
    }
}

/// The single answer to "is this available?" It is a plain value rebuilt from what is on disk and what the Supervisor
/// reports; nothing else in the app decides availability.
public struct Catalog: Sendable {
    public var models: [InstalledModel]
    public var engines: [EngineManifest]
    public var hosts: [HostInfo]
    /// Live state per host id (running, failed...), from the Supervisor and the MLX runtime.
    public var states: [String: ServiceState]
    /// Where each engine's folder lives, for binary lookup.
    public var engineDirs: [String: URL]

    public init(models: [InstalledModel], engines: [EngineManifest], hosts: [HostInfo], states: [String: ServiceState] = [:], engineDirs: [String: URL] = [:]) {
        self.models = models; self.engines = engines; self.hosts = hosts; self.states = states; self.engineDirs = engineDirs
    }

    public func engine(_ id: String) -> EngineManifest? { engines.first { $0.id == id } }
    public func host(_ id: String) -> HostInfo? { hosts.first { $0.id == id } }
    public func model(_ repo: String) -> InstalledModel? { models.first { $0.repo == repo } }
    public func isInstalled(_ repo: String) -> Bool { model(repo) != nil }

    public func capabilities(of host: HostInfo) -> [String] { engine(host.spec.engine)?.capabilities ?? [] }

    /// Hosts that offer a capability (`chat`, `image`), for pickers.
    public func hosts(capability: String) -> [HostInfo] { hosts.filter { capabilities(of: $0).contains(capability) } }

    /// The host that serves a model repo by default (its auto host, or an explicit host on it).
    public func host(forModel repo: String) -> HostInfo? {
        hosts.first { $0.spec.model == repo && !$0.isAuto } ?? hosts.first { $0.spec.model == repo }
    }

    /// Hosts that use a model, for the Library's "used by".
    public func hosts(usingModel repo: String) -> [HostInfo] { hosts.filter { $0.spec.model == repo } }

    public func binary(for engine: EngineManifest) -> URL? {
        guard let name = engine.requires?.binary else { return nil }
        return EngineBinary.find(name, engineDir: engineDirs[engine.id])
    }

    public func status(host id: String) -> HostStatus {
        guard let h = host(id) else { return .failed("no such host") }
        guard let e = engine(h.spec.engine) else { return .unknownEngine }
        if !h.spec.enabled { return .off }
        if e.remote {
            if (h.spec.url ?? "").isEmpty { return .failed("needs a url") }
        } else if !e.formats.isEmpty {
            guard let repo = h.spec.model, isInstalled(repo) else { return .missingModel }
        }
        if let r = e.requires, binary(for: e) == nil { return .missingBinary(r.hint ?? r.binary) }
        switch states[id] {
        case .starting?: return .starting
        case .running?, .idle?: return .running
        case .failed(let why)?: return .failed(why)
        default: return .ready
        }
    }

    /// The first thing that stops a host from serving, as a sentence; nil when it can.
    public func problem(host id: String) -> String? {
        let s = status(host: id)
        if s.usable { return nil }
        let name = id
        switch s {
        case .off: return "Host \(name) is off. Turn it on in Library."
        case .missingModel: return "\(host(id)?.spec.model.map { $0.split(separator: "/").last.map(String.init) ?? $0 } ?? name) is not downloaded."
        case .missingBinary(let hint): return "Host \(name) needs a program that is not installed (\(hint))."
        case .unknownEngine: return "Host \(name) uses an engine that is not installed."
        case .failed(let why): return "Host \(name) failed: \(why)"
        default: return nil
        }
    }
}

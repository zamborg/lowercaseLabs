import Foundation

public enum AdapterError: LocalizedError {
    case unsupported(String)
    public var errorDescription: String? {
        switch self { case .unsupported(let what): return "This engine cannot do \(what)." }
    }
}

/// What a host costs right now, for the memory budget and the dashboard.
public struct HostUsage: Sendable, Equatable {
    public var alive: Bool
    public var active: Int
    public var lastUsed: Date?
    public var bytes: Int64
    public init(alive: Bool, active: Int = 0, lastUsed: Date? = nil, bytes: Int64 = 0) {
        self.alive = alive; self.active = active; self.lastUsed = lastUsed; self.bytes = bytes
    }
}

/// One per wire protocol (`inprocess-mlx`, `http-openai`, `command`, `mcp-stdio`). The gateway is the only caller:
/// it resolves a host through the catalog, then asks the engine's adapter to do the work.
public protocol Adapter: AnyObject, Sendable {
    var id: String { get }
    /// The adapter keeps its own memory budget (in-process models do), so the gateway does not admit for it.
    var managesOwnBudget: Bool { get }

    func chat(_ req: LLMRequest, host: HostInfo, engine: EngineManifest, model: String?) -> AsyncThrowingStream<LLMEvent, Error>
    func image(_ req: ImageRequest, host: HostInfo, engine: EngineManifest) async throws -> ImageResult

    /// Called after every reload of engines, hosts and models, so a new host is picked up live.
    func reconfigure(catalog: Catalog)
    /// Re-read live state (what is loaded) before the catalog is read.
    func refresh() async
    /// The live state of a host, nil when this adapter has nothing to say (idle and in-process).
    func state(of host: HostInfo) -> ServiceState?
    func usage(of host: HostInfo) async -> HostUsage
    /// Bytes this adapter holds in total, for the budget.
    func residentBytes() async -> Int64
    func start(_ host: HostInfo) async throws
    func stop(_ host: HostInfo) async
    func shutdown() async
}

public extension Adapter {
    var managesOwnBudget: Bool { false }
    func chat(_ req: LLMRequest, host: HostInfo, engine: EngineManifest, model: String?) -> AsyncThrowingStream<LLMEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: AdapterError.unsupported("chat")) }
    }
    func image(_ req: ImageRequest, host: HostInfo, engine: EngineManifest) async throws -> ImageResult { throw AdapterError.unsupported("images") }
    func reconfigure(catalog: Catalog) {}
    func refresh() async {}
    func state(of host: HostInfo) -> ServiceState? { nil }
    func usage(of host: HostInfo) async -> HostUsage { HostUsage(alive: false) }
    func residentBytes() async -> Int64 { 0 }
    func start(_ host: HostInfo) async throws {}
    func stop(_ host: HostInfo) async {}
    func shutdown() async {}
}

/// Adapter id → instance. A drop-in engine names an adapter that is registered here; it needs no Swift.
public final class AdapterRegistry: @unchecked Sendable {
    private let lock = NSLock()
    private var adapters: [String: Adapter] = [:]
    public init() {}
    public func register(_ a: Adapter) { lock.lock(); adapters[a.id] = a; lock.unlock() }
    public subscript(id: String) -> Adapter? { lock.lock(); defer { lock.unlock() }; return adapters[id] }
    public var all: [Adapter] { lock.lock(); defer { lock.unlock() }; return adapters.values.sorted { $0.id < $1.id } }
}

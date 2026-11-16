import Foundation

/// `inprocess-mlx`: the model runs inside the app. A host's route is its model repo. With a substitute runtime
/// (tests) the same adapter wraps it.
public final class MLXAdapter: Adapter, @unchecked Sendable {
    public let id = "inprocess-mlx"
    public var managesOwnBudget: Bool { true }
    let runtime: LanguageModelRuntime
    let mlx: MLXRuntime?
    private let lock = NSLock()
    private var loaded: Set<String> = []

    init(runtime: LanguageModelRuntime, mlx: MLXRuntime?) { self.runtime = runtime; self.mlx = mlx }

    public func chat(_ req: LLMRequest, host: HostInfo, engine: EngineManifest, model: String?) -> AsyncThrowingStream<LLMEvent, Error> {
        guard let repo = host.spec.model else {
            return AsyncThrowingStream { $0.finish(throwing: ParlorError.message("Host \(host.id) names no model.")) }
        }
        return runtime.generate(model: repo, request: req)
    }

    public func refresh() async {
        let now: Set<String> = if let mlx { Set(await mlx.loadedModels()) } else if let m = await runtime.loadedModel() { [m] } else { [] }
        lock.withLock { loaded = now }
    }

    public func state(of host: HostInfo) -> ServiceState? {
        guard let m = host.spec.model else { return nil }
        lock.lock(); defer { lock.unlock() }
        return loaded.contains(m) ? .running : nil
    }

    public var loadedModels: [String] { lock.lock(); defer { lock.unlock() }; return loaded.sorted() }

    public func usage(of host: HostInfo) async -> HostUsage {
        guard let m = host.spec.model else { return HostUsage(alive: false) }
        if let mlx { return HostUsage(alive: await mlx.loadedModels().contains(m), bytes: await mlx.residentBytes(of: m)) }
        return HostUsage(alive: await runtime.loadedModel() == m)
    }

    public func residentBytes() async -> Int64 { await mlx?.residentBytes() ?? 0 }

    /// Warm: one token, so the weights are in memory.
    public func start(_ host: HostInfo) async throws {
        guard let repo = host.spec.model else { throw ParlorError.message("Host \(host.id) names no model.") }
        let req = LLMRequest(messages: [LLMMessage(role: .user, text: "Hi")], temperature: 0, maxTokens: 1, priority: .background)
        for try await _ in runtime.generate(model: repo, request: req) {}
        await refresh()
    }

    /// Unload this host's model only; other residents stay.
    public func stop(_ host: HostInfo) async {
        if let mlx, let m = host.spec.model { await mlx.unload(model: m) }
        else if let m = host.spec.model, await runtime.loadedModel() == m { await runtime.unload() }
        await refresh()
    }

    public func unloadAll() async { await runtime.unload(); await refresh() }
    public func shutdown() async { await runtime.unload() }
}

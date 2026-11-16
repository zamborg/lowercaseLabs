import Foundation

/// Crash policy, in one place. After each unexpected exit the owner asks what to do: wait and restart, or give up
/// (`failed`) once the delays run out inside the window.
public struct RestartPolicy: Sendable {
    public enum Decision: Equatable, Sendable { case restart(after: TimeInterval), fail }
    /// Delay before each automatic restart (1-based by crash count inside `window`).
    public var delays: [TimeInterval] = [1, 5, 30]
    public var window: TimeInterval = 600
    public init(delays: [TimeInterval] = [1, 5, 30], window: TimeInterval = 600) { self.delays = delays; self.window = window }

    /// Record a crash at `now` in `crashes` (older ones fall out of the window) and decide.
    public func decide(crashes: inout [Date], now: Date = Date()) -> Decision {
        crashes = crashes.filter { now.timeIntervalSince($0) < window } + [now]
        return crashes.count > delays.count ? .fail : .restart(after: delays[crashes.count - 1])
    }
}

/// One idle countdown: re-arm on use, fires once when nothing used it for `seconds`.
public final class IdleTimer: @unchecked Sendable {
    private let lock = NSLock()
    private var task: Task<Void, Never>?
    public init() {}
    public func arm(after seconds: TimeInterval, _ fire: @escaping @Sendable () async -> Void) {
        lock.lock(); task?.cancel()
        task = Task {
            try? await Task.sleep(nanoseconds: UInt64(max(0.01, seconds) * 1_000_000_000))
            if !Task.isCancelled { await fire() }
        }
        lock.unlock()
    }
    public func cancel() { lock.lock(); task?.cancel(); task = nil; lock.unlock() }
}

/// One supervised server process: started on demand, shown ready when it answers, stopped when idle, `failed` when it
/// crashes (the next use starts it again). A server that already answers where it should is used, not owned.
public final class ManagedService: @unchecked Sendable {
    public struct Spec: Sendable {
        public var id: String                      // "host:stheno"
        public var launch: Launch
        /// Ready when this answers 200; nil is ready once spawned.
        public var healthURL: URL?
        public var headers: @Sendable () -> [String: String] = { [:] }
        /// Stop after this long idle; nil keeps it running.
        public var idleSeconds: TimeInterval?
        public var startupSeconds: Int = 180
        public var banner: String = ""
        public init(id: String, launch: Launch, healthURL: URL? = nil, idleSeconds: TimeInterval? = nil, startupSeconds: Int = 180, banner: String = "") {
            self.id = id; self.launch = launch; self.healthURL = healthURL; self.idleSeconds = idleSeconds
            self.startupSeconds = startupSeconds; self.banner = banner
        }
    }

    public let spec: Spec
    let home: ParlorHome
    private let lock = NSLock()
    private var proc: ManagedProcess?
    private var starting: Task<Void, Error>?
    private var _state: ServiceState = .stopped
    private var _lastUsed: Date?
    private var active = 0
    private let idle = IdleTimer()
    private var external = false
    private var _restarts = 0
    /// Tests shorten the idle window below one second.
    var idleSecondsOverride: TimeInterval?

    public var serviceID: String { spec.id }

    public init(spec: Spec, home: ParlorHome) { self.spec = spec; self.home = home }

    public struct Snapshot: Sendable { public var state: ServiceState; public var pid: Int32?; public var lastUsed: Date?; public var external: Bool; public var restarts: Int; public var active: Int }
    public func snapshot() -> Snapshot {
        lock.lock(); defer { lock.unlock() }
        return Snapshot(state: _state, pid: proc?.pid, lastUsed: _lastUsed, external: external, restarts: _restarts, active: active)
    }

    /// Bytes this service holds, for the memory budget.
    public func footprint() -> Int64 {
        guard let pid = snapshot().pid, let b = ProcessMemory.footprint(pid: pid) else { return 0 }
        return Int64(b)
    }

    func touch() { lock.lock(); _lastUsed = Date(); lock.unlock() }
    /// Count a user; `end` drops it and starts the idle countdown.
    func begin() { lock.lock(); active += 1; _lastUsed = Date(); lock.unlock() }
    func end() { lock.lock(); active -= 1; _lastUsed = Date(); lock.unlock(); scheduleIdle() }

    private func answers() async -> Bool {
        guard let url = spec.healthURL else { return true }
        var r = URLRequest(url: url); r.timeoutInterval = 1.5
        for (k, v) in spec.headers() { r.setValue(v, forHTTPHeaderField: k) }
        guard let (_, resp) = try? await URLSession.shared.data(for: r) else { return false }
        return (resp as? HTTPURLResponse)?.statusCode == 200
    }

    public func ensureRunning() async throws {
        enum Plan { case ready, wait(Task<Void, Error>), launch(Task<Void, Error>) }
        let plan: Plan = lock.withLock {
            if _state == .running || _state == .idle, proc?.isRunning == true || external { _lastUsed = Date(); return .ready }
            if let s = starting { return .wait(s) }
            let t = Task { try await self.launch() }
            starting = t
            if case .failed = _state { _restarts += 1 }
            _state = .starting
            return .launch(t)
        }
        switch plan {
        case .ready: return
        case .wait(let s): try await s.value
        case .launch(let t):
            do { try await t.value } catch { lock.withLock { starting = nil }; throw error }
            lock.withLock { starting = nil }
        }
    }

    private func launch() async throws {
        // Something already answers where it should (the user started it): use it, don't own it.
        if spec.healthURL != nil, await answers() {
            lock.withLock { external = true; _state = .running; _lastUsed = Date() }
            return
        }
        let log = ServiceLog.open(home: home, service: spec.id, banner: spec.banner.isEmpty ? "starting" : "starting: \(spec.banner)")
        let p: ManagedProcess
        do {
            p = try Supervisor.spawn(spec.launch, streams: Streams(stdout: log, stderr: log)) { [weak self] proc, status in
                guard let self else { return }
                self.lock.lock()
                let wasStopping = self._state == .stopped
                if self.proc === proc {
                    self.proc = nil
                    if !wasStopping { self._state = .failed("exited with status \(status)") }
                }
                self.lock.unlock()
                try? log?.write(contentsOf: Data("--- exited \(status)\n".utf8))
            }
        } catch {
            lock.withLock { _state = .failed(error.localizedDescription) }
            throw ParlorError.message("Could not start \(spec.id): \(error.localizedDescription)")
        }
        lock.withLock { proc = p; external = false }
        let deadline = Date().addingTimeInterval(TimeInterval(max(5, spec.startupSeconds)))
        while Date() < deadline {
            if !p.isRunning {
                let tail = ServiceLog.tail(home: home, service: spec.id, lines: 6)
                lock.withLock { _state = .failed("exited with status \(p.terminationStatus)") }
                throw ParlorError.message("\(spec.id) exited while starting (status \(p.terminationStatus)).\n\(tail)")
            }
            if await answers() {
                lock.withLock { _state = .running; _lastUsed = Date() }
                scheduleIdle()
                return
            }
            try await Task.sleep(nanoseconds: 400_000_000)
        }
        stop()
        throw ParlorError.message("\(spec.id) did not answer within \(spec.startupSeconds)s. See its log in logs/.")
    }

    private func scheduleIdle() {
        guard let limit = idleSecondsOverride ?? spec.idleSeconds else { return }
        idle.arm(after: min(limit, 30)) { [weak self] in
            guard let self else { return }
            let quiet = self.lock.withLock { self.active == 0 && Date().timeIntervalSince(self._lastUsed ?? .distantPast) >= limit }
            if quiet { self.stop() } else { self.scheduleIdle() }
        }
    }

    /// Stop the server, if the app started it.
    public func stop() {
        lock.lock()
        idle.cancel()
        let p = proc
        let wasExternal = external
        _state = .stopped; external = false
        lock.unlock()
        guard !wasExternal, let p, p.isRunning else { return }
        p.terminate(grace: 3)
    }
}

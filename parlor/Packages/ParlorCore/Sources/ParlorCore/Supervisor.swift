import Darwin
import Foundation

// MARK: - what a service is

/// How to start a child process.
public struct Launch: Sendable {
    public var executable: URL
    public var arguments: [String] = []
    /// Extra variables on top of the app's environment.
    public var environment: [String: String] = [:]
    public var cwd: URL?

    public init(executable: URL, arguments: [String] = [], environment: [String: String] = [:], cwd: URL? = nil) {
        self.executable = executable; self.arguments = arguments; self.environment = environment; self.cwd = cwd
    }

    /// A shell command line, run through a login shell so Homebrew/uv/npm paths resolve the way they do in Terminal.
    public static func shell(_ command: String, environment: [String: String] = [:], cwd: URL? = nil) -> Launch {
        Launch(executable: URL(fileURLWithPath: "/bin/zsh"), arguments: ["-l", "-c", "exec " + command], environment: environment, cwd: cwd)
    }

    /// One program with literal arguments, through a login shell.
    public static func program(_ command: String, _ args: [String], environment: [String: String] = [:], cwd: URL? = nil) -> Launch {
        Launch(executable: URL(fileURLWithPath: "/bin/zsh"), arguments: ["-l", "-c", "exec \"$0\" \"$@\"", command] + args, environment: environment, cwd: cwd)
    }
}

/// A supervised thing, described as data: what it runs, how we know it is ready, when it stops, how it restarts.
public struct ServiceSpec: Sendable {
    public enum Kind: String, Sendable { case host, harness, tools, server }
    public enum Ready: Sendable { case http(URL), stdio, none }
    public enum Lifecycle: Equatable, Sendable { case always, onDemand(idleMinutes: Int), perRequest }
    public struct RestartPolicy: Sendable {
        /// Delay before each automatic restart; `failed` once these run out inside `window`.
        public var backoff: [TimeInterval] = [1, 5, 30]
        public var window: TimeInterval = 600
        public init() {}
    }

    public var id: String            // "host:stheno", "harness:wren", "mcp:fs", "module:dice", "provider:llama"
    public var kind: Kind
    public var launch: Launch?       // nil for in-process (MLX) and for external servers
    public var ready: Ready
    public var lifecycle: Lifecycle
    public var restart = RestartPolicy()

    public init(id: String, kind: Kind, launch: Launch? = nil, ready: Ready = .none, lifecycle: Lifecycle = .always) {
        self.id = id; self.kind = kind; self.launch = launch; self.ready = ready; self.lifecycle = lifecycle
    }
}

public enum ServiceAction: Sendable { case start, stop, restart }

/// One dashboard row and the verbs that act on it. The Supervisor never switches on an id prefix: the owner of
/// a service describes it here and brings its own start/stop.
public struct ServiceEntry: Sendable {
    public var info: ServiceInfo
    public var start: (@MainActor @Sendable () async throws -> Void)?
    public var stop: (@MainActor @Sendable () async throws -> Void)?
    public var restart: (@MainActor @Sendable () async throws -> Void)?

    public init(info: ServiceInfo, start: (@MainActor @Sendable () async throws -> Void)? = nil,
                stop: (@MainActor @Sendable () async throws -> Void)? = nil, restart: (@MainActor @Sendable () async throws -> Void)? = nil) {
        self.info = info; self.start = start; self.stop = stop; self.restart = restart
    }
}

// MARK: - a child process

/// A running child. The only way to get one is `Supervisor.spawn`.
public final class ManagedProcess: @unchecked Sendable {
    private let process: Process
    private let lock = NSLock()
    private var stopRequested = false

    fileprivate init(_ p: Process) { process = p }

    public var pid: Int32? { process.isRunning ? process.processIdentifier : nil }
    public var isRunning: Bool { process.isRunning }
    public var terminationStatus: Int32 { process.terminationStatus }
    /// True once `terminate` was called, so an exit handler can tell a stop from a crash.
    public var wasStopped: Bool { lock.lock(); defer { lock.unlock() }; return stopRequested }

    /// Ask the process to end; kill it if it is still there after `grace` seconds.
    public func terminate(grace: TimeInterval = 3) {
        lock.lock(); stopRequested = true; lock.unlock()
        guard process.isRunning else { return }
        process.terminate()
        let p = process
        DispatchQueue.global().asyncAfter(deadline: .now() + grace) { if p.isRunning { kill(p.processIdentifier, SIGKILL) } }
    }
}

/// How the child's standard streams are wired.
public struct Streams {
    public var stdin: Any? = FileHandle.nullDevice
    public var stdout: Any?
    public var stderr: Any?
    public init(stdin: Any? = FileHandle.nullDevice, stdout: Any? = nil, stderr: Any? = nil) {
        self.stdin = stdin; self.stdout = stdout; self.stderr = stderr
    }
}

// MARK: - the supervisor

/// Owns every child process (hosts, harnesses, tool servers, one-off runs) and projects them onto the dashboard.
public actor Supervisor {
    private var order: [String] = []
    private var sources: [String: @Sendable () async -> [ServiceEntry]] = [:]

    public init() {}

    // MARK: processes

    /// Start a child. `onExit` runs on termination with the status. Throws when it cannot start.
    public static func spawn(_ launch: Launch, streams: Streams = Streams(), onExit: (@Sendable (ManagedProcess, Int32) -> Void)? = nil) throws -> ManagedProcess {
        let p = Process()
        p.executableURL = launch.executable
        p.arguments = launch.arguments
        var env = ProcessInfo.processInfo.environment
        for (k, v) in launch.environment { env[k] = v }
        p.environment = env
        if let cwd = launch.cwd { p.currentDirectoryURL = cwd }
        p.standardInput = streams.stdin
        if let o = streams.stdout { p.standardOutput = o }
        if let e = streams.stderr { p.standardError = e }
        let managed = ManagedProcess(p)
        p.terminationHandler = { proc in onExit?(managed, proc.terminationStatus) }
        try p.run()
        return managed
    }

    /// Run a program to completion, with its output going to `output` (or nowhere), and return its status. Cancelling the task stops it;
    /// `timeout` ends it too. Safe to await from any task.
    public static func run(_ launch: Launch, output: FileHandle? = nil, timeout: TimeInterval = 1500) async throws -> Int32 {
        let sink: Any = output ?? FileHandle.nullDevice
        let streams = Streams(stdout: sink, stderr: sink)
        let box = ProcessBox()
        let status: Int32 = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Int32, Error>) in
                do {
                    let p = try spawn(launch, streams: streams) { _, status in cont.resume(returning: status) }
                    box.set(p)
                    DispatchQueue.global().asyncAfter(deadline: .now() + timeout) { if p.isRunning { p.terminate() } }
                } catch { cont.resume(throwing: error) }
            }
        } onCancel: { box.get()?.terminate() }
        if Task.isCancelled { throw CancellationError() }
        return status
    }

    /// Blocking variant for code that already runs off the main thread (git). Waits on a semaphore, never on
    /// `waitUntilExit`, which spins the run loop and re-enters the UI.
    public static func runBlocking(_ launch: Launch) -> (status: Int32, output: String) {
        let pipe = Pipe()
        let done = DispatchSemaphore(value: 0)
        guard let p = try? spawn(launch, streams: Streams(stdout: pipe, stderr: pipe), onExit: { _, _ in done.signal() }) else { return (-1, "") }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        done.wait()
        return (p.terminationStatus, String(decoding: data, as: UTF8.self))
    }

    private final class ProcessBox: @unchecked Sendable {
        private let lock = NSLock(); private var p: ManagedProcess?
        func set(_ v: ManagedProcess) { lock.lock(); p = v; lock.unlock() }
        func get() -> ManagedProcess? { lock.lock(); defer { lock.unlock() }; return p }
    }

    // MARK: dashboard

    /// Register something that lists services. Called once per source; the closure is asked on every snapshot.
    public func register(source name: String, _ list: @escaping @Sendable () async -> [ServiceEntry]) {
        if sources[name] == nil { order.append(name) }
        sources[name] = list
    }

    public func entries() async -> [ServiceEntry] {
        var out: [ServiceEntry] = []
        for name in order { if let s = sources[name] { out += await s() } }
        return out
    }

    public func snapshot() async -> [ServiceInfo] { await entries().map(\.info) }

    public func control(_ id: String, _ action: ServiceAction) async throws {
        guard let e = await entries().first(where: { $0.info.id == id }) else {
            throw CommandError("No service \"\(id)\". Use services.list.")
        }
        switch action {
        case .start: try await e.start?()
        case .stop: try await e.stop?()
        case .restart:
            if let r = e.restart { try await r() } else { try await e.stop?(); try await e.start?() }
        }
    }
}

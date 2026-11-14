import Foundation

/// Automatic undo for the hand-editable state: `~/parlor/.git` records agents, memory, settings, mcp.json,
/// module manifests and conversation configs. Commits are debounced, attributed to whoever wrote (the app,
/// an agent, a token), and never include models, media, logs, workspaces or chat logs.
public final class HomeHistory: @unchecked Sendable {
    public struct Entry: Sendable, Equatable {
        public var hash: String
        public var author: String
        public var date: Date
        public var subject: String
        public var files: [String]
    }

    static let gitPath = "/usr/bin/git"
    public static var gitAvailable: Bool { FileManager.default.isExecutableFile(atPath: gitPath) }

    static let ignore = """
    # managed by parlor: only hand-editable state is versioned
    /*
    !/.gitignore
    !/agents/
    !/memory/
    !/settings.json
    !/mcp.json
    !/modules/
    !/chats/
    /modules/*/*
    !/modules/*/module.json
    /chats/*/*
    !/chats/*/conversation.json
    .DS_Store
    """

    public let root: URL
    private let queue = DispatchQueue(label: "parlor.history")
    private let lock = NSLock()
    private var pending: (actor: String, message: String)?
    private var timer: DispatchWorkItem?
    private var ready = false

    public init(root: URL) { self.root = root }

    public var isEnabled: Bool { Self.gitAvailable }

    @discardableResult
    private func git(_ args: [String], actor: String = "parlor", input: Data? = nil) -> (status: Int32, out: String) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: Self.gitPath)
        p.arguments = ["-C", root.path] + args
        var env = ProcessInfo.processInfo.environment
        let name = actor.isEmpty ? "parlor" : actor
        env["GIT_AUTHOR_NAME"] = name; env["GIT_AUTHOR_EMAIL"] = "\(Agent.slug(name))@parlor.local"
        env["GIT_COMMITTER_NAME"] = "parlor"; env["GIT_COMMITTER_EMAIL"] = "parlor@parlor.local"
        env["GIT_CONFIG_GLOBAL"] = "/dev/null"; env["GIT_CONFIG_SYSTEM"] = "/dev/null"
        env["GIT_TERMINAL_PROMPT"] = "0"
        p.environment = env
        let out = Pipe()
        p.standardOutput = out; p.standardError = out
        p.standardInput = FileHandle.nullDevice
        // Wait on a semaphore, not waitUntilExit: that spins the run loop, which re-enters the UI when called on the main thread.
        let done = DispatchSemaphore(value: 0)
        p.terminationHandler = { _ in done.signal() }
        do { try p.run() } catch { return (-1, "") }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        done.wait()
        return (p.terminationStatus, String(decoding: data, as: UTF8.self))
    }

    /// Create the repository on first use and commit what is already there. Safe to call every launch.
    public func setup() {
        guard isEnabled else { return }
        queue.sync {
            let fm = FileManager.default
            try? fm.createDirectory(at: root, withIntermediateDirectories: true)
            if !fm.fileExists(atPath: root.appendingPathComponent(".git").path) {
                guard git(["init", "-q", "-b", "main"]).status == 0 else { return }
                git(["config", "core.autocrlf", "false"])
            }
            let ig = root.appendingPathComponent(".gitignore")
            if (try? String(contentsOf: ig, encoding: .utf8)) != Self.ignore { try? Self.ignore.write(to: ig, atomically: true, encoding: .utf8) }
            ready = true
            commitNow(actor: "parlor", message: "Start history")
        }
    }

    /// Call just before writing a file for `actor`: anything waiting from someone else is committed first, so their
    /// commit does not pick up this write.
    public func willChange(actor: String) {
        guard isEnabled else { return }
        lock.lock(); let other = pending.map { $0.actor != actor } ?? false; lock.unlock()
        if other { flush() }
    }

    /// Note that something changed. Many calls in a short time make one commit, credited to the first caller.
    public func record(actor: String, _ message: String) {
        guard isEnabled else { return }
        lock.lock()
        // A different actor gets its own commit: what is waiting is committed first.
        if let p = pending, p.actor != actor { lock.unlock(); flush(); lock.lock() }
        if pending == nil { pending = (actor, message) }
        timer?.cancel()
        let item = DispatchWorkItem { [weak self] in self?.commitPending() }
        timer = item
        lock.unlock()
        queue.asyncAfter(deadline: .now() + 0.6, execute: item)
    }

    /// Commit anything pending now.
    public func flush() { queue.sync { commitPending() } }

    /// Runs on `queue`.
    private func commitPending() {
        lock.lock(); let p = pending; pending = nil; timer?.cancel(); timer = nil; lock.unlock()
        if let p, ready { commitNow(actor: p.actor, message: p.message) }
    }

    private func commitNow(actor: String, message: String) {
        git(["add", "-A"])
        guard git(["diff", "--cached", "--quiet"]).status != 0 else { return }
        git(["commit", "-q", "-m", message], actor: actor)
    }

    /// Recent commits, newest first, optionally only those touching `path`.
    public func log(limit: Int = 20, path: String? = nil) -> [Entry] {
        guard isEnabled else { return [] }
        flush()
        return queue.sync {
            var args = ["log", "-n", String(max(1, min(200, limit))), "--name-only", "--format=\u{1e}%H\u{1f}%an\u{1f}%at\u{1f}%s"]
            if let path { args += ["--", path] }
            let r = git(args)
            guard r.status == 0 else { return [] }
            return r.out.components(separatedBy: "\u{1e}").compactMap { block in
                guard !block.isEmpty else { return nil }
                let lines = block.components(separatedBy: "\n")
                let head = lines[0].components(separatedBy: "\u{1f}")
                guard head.count == 4 else { return nil }
                return Entry(hash: head[0], author: head[1], date: Date(timeIntervalSince1970: TimeInterval(head[2]) ?? 0), subject: head[3],
                             files: lines.dropFirst().filter { !$0.isEmpty })
            }
        }
    }

    /// Put files back as they were at `rev` (all tracked files when `paths` is empty) and commit that as a new step.
    /// Files that did not exist then are removed. Returns what changed.
    public func restore(rev: String, paths: [String], actor: String) throws -> [String] {
        guard isEnabled else { throw CommandError("History needs git, which is not installed.") }
        guard rev.range(of: "^[0-9a-fA-F]{4,40}$", options: .regularExpression) != nil else { throw CommandError("\"rev\" is a commit hash from home.history.") }
        flush()
        return try queue.sync {
            guard git(["cat-file", "-e", rev + "^{commit}"]).status == 0 else { throw CommandError("No such commit \(rev).") }
            for p in paths where p.hasPrefix("/") || p.contains("..") { throw CommandError("Paths are relative to the parlor folder.") }
            let scope = paths.isEmpty ? ["--", "."] : ["--"] + paths
            // Anything the commit lacks goes; everything it has comes back.
            let current = git(["ls-files"] + scope).out.split(separator: "\n").map(String.init)
            let then = Set(git(["ls-tree", "-r", "--name-only", rev] + scope).out.split(separator: "\n").map(String.init))
            var changed: [String] = []
            for f in current where !then.contains(f) {
                try? FileManager.default.removeItem(at: root.appendingPathComponent(f)); changed.append(f)
            }
            if !then.isEmpty {
                let r = git(["checkout", rev, "--"] + then.sorted())
                guard r.status == 0 else { throw CommandError("Could not restore: \(r.out.prefix(200))") }
                changed += then.sorted()
            }
            git(["add", "-A"])
            if git(["diff", "--cached", "--quiet"]).status != 0 {
                git(["commit", "-q", "-m", "Restore to \(rev.prefix(8))"], actor: actor)
            }
            return Array(Set(changed)).sorted()
        }
    }
}

/// `memory/<id>.md`: durable notes an agent keeps across all its conversations.
public struct MemoryStore: Sendable {
    public let dir: URL
    /// Characters kept in the system prompt; longer memory is refused on append so it never silently eats the context.
    public static let maxChars = 6000

    public init(home: ParlorHome) { dir = home.memoryDir }

    public func url(_ id: String) -> URL { dir.appendingPathComponent(id + ".md") }

    public func read(_ id: String) -> String {
        (try? String(contentsOf: url(id), encoding: .utf8)) ?? ""
    }

    public func write(_ id: String, _ text: String) throws {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if t.isEmpty { try? FileManager.default.removeItem(at: url(id)); return }
        try (t + "\n").write(to: url(id), atomically: true, encoding: .utf8)
    }

    public func append(_ id: String, _ text: String) throws {
        let add = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !add.isEmpty else { throw CommandError("Nothing to remember.") }
        let cur = read(id).trimmingCharacters(in: .whitespacesAndNewlines)
        let next = cur.isEmpty ? add : cur + "\n" + add
        guard next.count <= Self.maxChars else {
            throw CommandError("Memory is full (\(cur.count) of \(Self.maxChars) characters). Condense it with memory.replace, then append.")
        }
        try write(id, next)
    }
}

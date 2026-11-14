import Foundation

/// Starts a conversation's scheduler with no user message: on a clock (`every`, `at`) or when a file changes (`onFile`).
public struct ConversationTrigger: Codable, Hashable, Sendable {
    /// "30m", "2h" or "1d". With `at`, whole days (`1d` is the default).
    public var every: String?
    /// "HH:MM" local time of day.
    public var at: String?
    /// Glob relative to the parlor folder, e.g. `workspaces/mabel/inbox/*`.
    public var onFile: String?

    public init(every: String? = nil, at: String? = nil, onFile: String? = nil) { self.every = every; self.at = at; self.onFile = onFile }

    /// Seconds in `every`, at least a minute. nil when absent or unreadable.
    var interval: TimeInterval? {
        guard let e = every?.trimmingCharacters(in: .whitespaces).lowercased(), let unit = e.last, let n = Double(e.dropLast()), n > 0 else { return nil }
        let s: Double
        switch unit { case "s": s = n; case "m": s = n * 60; case "h": s = n * 3600; case "d": s = n * 86400; default: return nil }
        return max(60, s)
    }

    var timeOfDay: (hour: Int, minute: Int)? {
        guard let at else { return nil }
        let p = at.split(separator: ":").compactMap { Int($0) }
        guard p.count == 2, (0..<24).contains(p[0]), (0..<60).contains(p[1]) else { return nil }
        return (p[0], p[1])
    }

    public var isClock: Bool { onFile == nil && (interval != nil || timeOfDay != nil) }

    /// The next time this clock trigger is due, counting from the last run (or from when it was first seen).
    public func nextFire(after last: Date, calendar: Calendar = .current) -> Date? {
        guard isClock else { return nil }
        if let t = timeOfDay {
            let days = max(1, Int(((interval ?? 86400) / 86400).rounded()))
            let day = calendar.startOfDay(for: last)
            // The first HH:MM after `last`, and no sooner than `days` days after the day of `last`.
            guard let today = calendar.date(bySettingHour: t.hour, minute: t.minute, second: 0, of: day),
                  let spaced = calendar.date(byAdding: .day, value: days, to: today) else { return nil }
            let next = today > last ? today : (calendar.date(byAdding: .day, value: 1, to: today) ?? today)
            return days > 1 ? max(next, spaced) : next
        }
        return interval.map { last.addingTimeInterval($0) }
    }

    /// Whether a changed file (path relative to the parlor folder) matches `onFile`.
    public func matches(relativePath: String) -> Bool {
        guard let g = onFile, !g.isEmpty else { return false }
        return fnmatch(g, relativePath, 0) == 0
    }
}

/// Fires conversation triggers while the app is open. A time that passes while it is closed is not made up.
@MainActor
final class TriggerScheduler {
    unowned let engine: ParlorEngine
    private var timer: Timer?
    private var lastFired: [String: Date] = [:]      // "<chat>#<index>"
    /// A file trigger will not fire again within this long, so an agent writing files cannot start itself in a loop.
    var fileCooldown: TimeInterval = 30

    init(engine: ParlorEngine) { self.engine = engine }

    func start() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 20, repeats: true) { [weak self] _ in
            Task { @MainActor in _ = self?.tick() }
        }
    }
    func stop() { timer?.invalidate(); timer = nil }

    private func configs() -> [(id: String, config: ConversationConfig)] {
        let fm = FileManager.default
        return ((try? fm.contentsOfDirectory(atPath: engine.home.chatsDir.path)) ?? []).compactMap { d in
            let dir = engine.home.chatDir(for: d)
            guard fm.fileExists(atPath: dir.appendingPathComponent("conversation.json").path) else { return nil }
            let c = ConversationConfig.load(dir: dir, agentID: d)
            return c.triggers.isEmpty ? nil : (d, c)
        }
    }

    private func stateURL(_ id: String) -> URL { engine.home.chatDir(for: id).appendingPathComponent("triggers.json") }

    private func loadState(_ id: String) {
        guard let d = try? Data(contentsOf: stateURL(id)), let o = try? JSONDecoder().decode([String: Double].self, from: d) else { return }
        for (k, v) in o where lastFired["\(id)#\(k)"] == nil { lastFired["\(id)#\(k)"] = Date(timeIntervalSince1970: v) }
    }

    private func saveState(_ id: String, _ count: Int) {
        var o: [String: Double] = [:]
        for i in 0..<count { if let t = lastFired["\(id)#\(i)"] { o[String(i)] = t.timeIntervalSince1970 } }
        try? JSONEncoder().encode(o).write(to: stateURL(id), options: .atomic)
    }

    /// Run every clock trigger that is due. Returns the conversations it started.
    @discardableResult
    func tick(now: Date = Date()) -> [String] {
        var started: [String] = []
        for (id, config) in configs() {
            loadState(id)
            var dirty = false
            for (i, t) in config.triggers.enumerated() where t.isClock {
                let key = "\(id)#\(i)"
                guard let last = lastFired[key] else { lastFired[key] = now; dirty = true; continue }   // first sight: count from now
                guard let due = t.nextFire(after: last), due <= now else { continue }
                lastFired[key] = now; dirty = true
                if fire(id) { started.append(id) }
            }
            if dirty { saveState(id, config.triggers.count) }
        }
        return started
    }

    /// Files changed under the parlor folder: run matching `onFile` triggers.
    @discardableResult
    func filesChanged(_ paths: [String], now: Date = Date()) -> [String] {
        let roots = [engine.home.root.path, ParlorEngine.realPath(engine.home.root.path)]
        let rel: [String] = paths.compactMap { p in
            guard let r = roots.first(where: { p.hasPrefix($0 + "/") }) else { return nil }
            let s = String(p.dropFirst(r.count + 1))
            return s.hasPrefix(".git/") || s.hasPrefix("logs/") ? nil : s
        }
        guard !rel.isEmpty else { return [] }
        var started: [String] = []
        for (id, config) in configs() {
            let mine = rel.filter { !$0.hasPrefix("chats/\(id)/") }
            for (i, t) in config.triggers.enumerated() where t.onFile != nil && mine.contains(where: t.matches(relativePath:)) {
                let key = "\(id)#\(i)"
                if let last = lastFired[key], now.timeIntervalSince(last) < fileCooldown { continue }
                lastFired[key] = now
                if fire(id) { started.append(id) }
            }
        }
        return started
    }

    private func fire(_ id: String) -> Bool {
        let room = engine.room(for: id)
        guard !room.isBusy else { return false }
        room.run()
        return true
    }
}

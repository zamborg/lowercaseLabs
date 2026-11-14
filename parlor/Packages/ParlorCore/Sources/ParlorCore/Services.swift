import Darwin
import Foundation

/// `stopped → starting → running ⇄ idle → stopped`, and any state → `failed`.
public enum ServiceState: Equatable, Sendable {
    case stopped, starting, running, idle
    case failed(String)

    public var label: String {
        switch self {
        case .stopped: return "stopped"
        case .starting: return "starting"
        case .running: return "running"
        case .idle: return "idle"
        case .failed: return "failed"
        }
    }
    public var isAlive: Bool { self == .running || self == .idle || self == .starting }
}

/// One supervised thing: the language model runtime, the image worker, a mounted MCP server.
public struct ServiceInfo: Identifiable, Equatable, Sendable {
    public var id: String            // "llm", "image", "mcp:<name>"
    public var kind: String          // "language model", "image", "mcp"
    public var title: String
    public var detail: String
    public var state: ServiceState
    public var memoryBytes: UInt64?
    public var lastUsed: Date?
    public var queueDepth: Int?
    public var lifecycle: String?    // "always" | "onDemand" for mounts
    public var restarts: Int
    public var canStart: Bool
    public var canStop: Bool
    public var canRestart: Bool
    public var logURL: URL?
}

public enum ServiceLog {
    static let maxBytes = 1_000_000
    static let keep = 3

    public static func url(home: ParlorHome, service id: String) -> URL {
        let safe = id.replacingOccurrences(of: ":", with: "-").replacingOccurrences(of: "/", with: "_")
        return home.logsDir.appendingPathComponent(safe + ".log")
    }

    /// Open a service's log for appending, rotating `x.log → x.log.1 → x.log.2` past 1 MB. Writes a start line.
    public static func open(home: ParlorHome, service id: String, banner: String) -> FileHandle? {
        let fm = FileManager.default
        try? fm.createDirectory(at: home.logsDir, withIntermediateDirectories: true)
        let u = url(home: home, service: id)
        if let size = (try? fm.attributesOfItem(atPath: u.path))?[.size] as? Int, size > maxBytes { rotate(u) }
        if !fm.fileExists(atPath: u.path) { fm.createFile(atPath: u.path, contents: nil) }
        guard let h = try? FileHandle(forWritingTo: u) else { return nil }
        _ = try? h.seekToEnd()
        let stamp = ISO8601DateFormatter().string(from: Date())
        try? h.write(contentsOf: Data("--- \(stamp) \(banner)\n".utf8))
        return h
    }

    static func rotate(_ u: URL) {
        let fm = FileManager.default
        for i in stride(from: keep - 1, through: 1, by: -1) {
            let from = i == 1 ? u : URL(fileURLWithPath: u.path + ".\(i - 1)")
            let to = URL(fileURLWithPath: u.path + ".\(i)")
            try? fm.removeItem(at: to)
            if fm.fileExists(atPath: from.path) { try? fm.moveItem(at: from, to: to) }
        }
    }

    /// The last `lines` lines of a service's log.
    public static func tail(home: ParlorHome, service id: String, lines: Int = 8) -> String {
        guard let text = try? String(contentsOf: url(home: home, service: id), encoding: .utf8) else { return "" }
        return text.split(separator: "\n", omittingEmptySubsequences: true)
            .filter { !$0.hasPrefix("---") }.suffix(lines).joined(separator: "\n")
    }
}

public enum ProcessMemory {
    /// Physical footprint of a child process, as Activity Monitor reports it.
    public static func footprint(pid: Int32) -> UInt64? {
        var info = rusage_info_v4()
        let r = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(pid, RUSAGE_INFO_V4, $0) }
        }
        return r == 0 ? info.ri_phys_footprint : nil
    }
}

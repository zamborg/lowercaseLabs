import Foundation

/// Root of all parlor state on disk. Files are the database.
public struct ParlorHome: Sendable {
    public let root: URL

    public init(root: URL) { self.root = root }

    /// `PARLOR_HOME` env var wins, otherwise `~/parlor`.
    public static var `default`: ParlorHome {
        if let p = ProcessInfo.processInfo.environment["PARLOR_HOME"], !p.isEmpty {
            return ParlorHome(root: URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        }
        return ParlorHome(root: FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("parlor"))
    }

    public var agentsDir: URL { root.appendingPathComponent("agents") }
    public var chatsDir: URL { root.appendingPathComponent("chats") }
    public var mediaDir: URL { root.appendingPathComponent("media") }
    public var modelsDir: URL {
        if let p = ProcessInfo.processInfo.environment["PARLOR_MODELS"], !p.isEmpty {
            return URL(fileURLWithPath: (p as NSString).expandingTildeInPath)
        }
        return root.appendingPathComponent("models")
    }
    public var modulesDir: URL { root.appendingPathComponent("modules") }
    /// Working directory of an external harness: `workspaces/<agent>/`.
    public var workspacesDir: URL { root.appendingPathComponent("workspaces") }
    public var harnessesDir: URL { root.appendingPathComponent("harnesses") }
    public var logsDir: URL { root.appendingPathComponent("logs") }
    /// Derived data that is safe to delete (cached tool lists of on-demand servers).
    public var cacheDir: URL { root.appendingPathComponent(".cache") }
    public var mcpConfig: URL { root.appendingPathComponent("mcp.json") }
    public var settingsFile: URL { root.appendingPathComponent("settings.json") }
    /// Present once the starter agents have been created (or an existing home was adopted).
    public var seededMarker: URL { root.appendingPathComponent(".parlor-seeded") }

    public func chatDir(for agentID: String) -> URL { chatsDir.appendingPathComponent(agentID) }

    public func ensure() throws {
        let fm = FileManager.default
        for d in [root, agentsDir, chatsDir, mediaDir, modelsDir, logsDir] {
            try fm.createDirectory(at: d, withIntermediateDirectories: true)
        }
    }
}

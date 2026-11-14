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
    public var mcpConfig: URL { root.appendingPathComponent("mcp.json") }
    public var settingsFile: URL { root.appendingPathComponent("settings.json") }

    public func chatDir(for agentID: String) -> URL { chatsDir.appendingPathComponent(agentID) }

    public func ensure() throws {
        let fm = FileManager.default
        for d in [root, agentsDir, chatsDir, mediaDir, modelsDir] {
            try fm.createDirectory(at: d, withIntermediateDirectories: true)
        }
    }
}

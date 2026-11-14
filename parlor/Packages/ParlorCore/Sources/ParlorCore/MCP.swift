import Foundation
import MCP
import System

/// One entry of `mcp.json` (Claude Desktop format).
public struct MCPServerConfig: Codable, Sendable, Equatable {
    public var command: String?
    public var args: [String]?
    public var env: [String: String]?
    public var cwd: String?
    public var url: String?
    public var disabled: Bool?

    public init(command: String? = nil, args: [String]? = nil, env: [String: String]? = nil,
                cwd: String? = nil, url: String? = nil, disabled: Bool? = nil) {
        self.command = command; self.args = args; self.env = env
        self.cwd = cwd; self.url = url; self.disabled = disabled
    }
}

public struct MCPConfigFile: Codable, Sendable, Equatable {
    public var mcpServers: [String: MCPServerConfig]
    public init(mcpServers: [String: MCPServerConfig] = [:]) { self.mcpServers = mcpServers }

    public static func load(_ url: URL) -> MCPConfigFile {
        guard let d = try? Data(contentsOf: url), let c = try? JSONDecoder().decode(MCPConfigFile.self, from: d) else {
            return MCPConfigFile()
        }
        return c
    }

    public func save(_ url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try enc.encode(self).write(to: url)
    }
}

/// A connection to one MCP server (stdio child process or streamable HTTP).
public actor MCPServerConnection: ToolProvider {
    public nonisolated let id: String
    private let config: MCPServerConfig
    private let home: ParlorHome
    private var client: Client?
    private var process: Process?
    private weak var hub: ToolHub?

    public init(id: String, config: MCPServerConfig, home: ParlorHome, hub: ToolHub) {
        self.id = id; self.config = config; self.home = home; self.hub = hub
    }

    /// Connect with a deadline so a hung or crashing server cannot block the app.
    public func start(timeout: TimeInterval = 25) async throws {
        do {
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask { try await self.connect() }
                group.addTask {
                    try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                    throw ParlorError.message("Timed out waiting for \(self.id) to start. Check the command in mcp.json.")
                }
                _ = try await group.next()
                group.cancelAll()
            }
        } catch {
            await stop()
            throw error
        }
    }

    private func connect() async throws {
        let client = Client(name: "parlor", version: "0.1.0")
        if let urlString = config.url, let url = URL(string: urlString) {
            _ = try await client.connect(transport: HTTPClientTransport(endpoint: url))
        } else if let command = config.command {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/bin/zsh")
            // Login shell so Homebrew/uv/npm paths resolve the way they do in Terminal.
            proc.arguments = ["-l", "-c", "exec \"$0\" \"$@\"", command] + (config.args ?? [])
            var env = ProcessInfo.processInfo.environment
            let extra = "/opt/homebrew/bin:/usr/local/bin:\(NSHomeDirectory())/.local/bin:\(NSHomeDirectory())/.cargo/bin"
            env["PATH"] = extra + ":" + (env["PATH"] ?? "/usr/bin:/bin")
            for (k, v) in config.env ?? [:] { env[k] = v }
            proc.environment = env
            if let cwd = config.cwd {
                proc.currentDirectoryURL = URL(fileURLWithPath: (cwd as NSString).expandingTildeInPath)
            } else {
                proc.currentDirectoryURL = home.root
            }
            let stdin = Pipe(), stdout = Pipe()
            proc.standardInput = stdin
            proc.standardOutput = stdout
            proc.standardError = FileHandle.nullDevice
            try proc.run()
            process = proc
            let transport = StdioTransport(
                input: FileDescriptor(rawValue: stdout.fileHandleForReading.fileDescriptor),
                output: FileDescriptor(rawValue: stdin.fileHandleForWriting.fileDescriptor))
            _ = try await client.connect(transport: transport)
        } else {
            throw ParlorError.message("Server \(id) needs a command or a url")
        }
        let me = id
        let hub = self.hub
        await client.onNotification(ToolListChangedNotification.self) { _ in
            await hub?.refresh(me)
        }
        self.client = client
    }

    public func stop() async {
        await client?.disconnect()
        client = nil
        if let p = process, p.isRunning { p.terminate() }
        process = nil
    }

    public func listTools() async throws -> [ToolDef] {
        guard let client else { throw ParlorError.message("Server \(id) is not connected") }
        var out: [ToolDef] = []
        var cursor: String?
        repeat {
            let page = try await client.listTools(cursor: cursor)
            for t in page.tools {
                let schema = (try? JSONEncoder().encode(t.inputSchema)).flatMap { try? JSONDecoder().decode(JSON.self, from: $0) } ?? .object([:])
                out.append(ToolDef(server: id, name: t.name, description: t.description ?? "", schema: schema))
            }
            cursor = page.nextCursor
        } while cursor != nil
        return out
    }

    public func call(tool: String, arguments: JSON) async throws -> ToolOutput {
        guard let client else { throw ParlorError.message("Server \(id) is not connected") }
        var args: [String: Value]?
        if let obj = arguments.objectValue {
            let data = try JSONEncoder().encode(JSON.object(obj))
            args = try JSONDecoder().decode([String: Value].self, from: data)
        }
        let result = try await client.callTool(name: tool, arguments: args)
        var texts: [String] = []
        var images: [String] = []
        for c in result.content {
            switch c {
            case .text(let t, _, _): texts.append(t)
            case .image(let data, let mime, _, _):
                if let path = Media.save(base64: data, mime: mime, home: home) {
                    images.append(path); texts.append("[image saved: \(path)]")
                }
            case .resource(let r, _, _):
                texts.append(r.text ?? "[resource \(r.uri)]")
            case .resourceLink(let uri, let name, _, _, _, _): texts.append("[\(name): \(uri)]")
            default: break
            }
        }
        return ToolOutput(text: texts.joined(separator: "\n"), images: images, isError: result.isError ?? false)
    }
}

enum Media {
    /// Save base64 image data under `media/YYYY-MM/` and return the home-relative path.
    static func save(base64: String, mime: String, home: ParlorHome) -> String? {
        guard let data = Data(base64Encoded: base64) else { return nil }
        let ext = mime.contains("png") ? "png" : mime.contains("jpeg") || mime.contains("jpg") ? "jpg" : "img"
        return save(data: data, ext: ext, home: home)
    }

    static func save(data: Data, ext: String, home: ParlorHome) -> String? {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM"
        let sub = "media/" + f.string(from: Date())
        let dir = home.root.appendingPathComponent(sub)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let name = UUID().uuidString.prefix(8).lowercased() + "." + ext
        do { try data.write(to: dir.appendingPathComponent(String(name))); return sub + "/" + name }
        catch { return nil }
    }
}

/// Keeps the hub in sync with `mcp.json`: starts, stops and restarts connections.
public actor MCPManager {
    private let hub: ToolHub
    private let home: ParlorHome
    private var connections: [String: MCPServerConnection] = [:]
    private var configs: [String: MCPServerConfig] = [:]

    public init(hub: ToolHub, home: ParlorHome) { self.hub = hub; self.home = home }

    public func reload() async {
        let file = MCPConfigFile.load(home.mcpConfig)
        let wanted = file.mcpServers.filter { $0.value.disabled != true }
        for id in connections.keys where wanted[id] == nil || wanted[id] != configs[id] {
            await stop(id)
        }
        let toStart = wanted.filter { connections[$0.key] == nil }
        await withTaskGroup(of: Void.self) { group in
            for (id, cfg) in toStart { group.addTask { await self.start(id, cfg) } }
        }
    }

    public func restart(_ id: String) async {
        let file = MCPConfigFile.load(home.mcpConfig)
        await stop(id)
        if let cfg = file.mcpServers[id] { await start(id, cfg) }
    }

    public func stopAll() async { for id in Array(connections.keys) { await stop(id) } }

    private func stop(_ id: String) async {
        if let c = connections[id] { await c.stop() }
        connections[id] = nil; configs[id] = nil
        await hub.unregister(id)
    }

    private func start(_ id: String, _ cfg: MCPServerConfig) async {
        let kind = cfg.url != nil ? "http" : "stdio"
        let detail = cfg.url ?? ([cfg.command ?? ""] + (cfg.args ?? [])).joined(separator: " ")
        await hub.setStatus(ServerStatus(id: id, kind: kind, detail: detail, state: .connecting, toolCount: 0, toolNames: []))
        let conn = MCPServerConnection(id: id, config: cfg, home: home, hub: hub)
        connections[id] = conn; configs[id] = cfg
        do {
            try await conn.start()
            await hub.register(conn, status: ServerStatus(id: id, kind: kind, detail: detail, state: .connecting, toolCount: 0, toolNames: []))
        } catch {
            await hub.setStatus(ServerStatus(id: id, kind: kind, detail: detail, state: .failed(error.localizedDescription), toolCount: 0, toolNames: []))
        }
    }
}

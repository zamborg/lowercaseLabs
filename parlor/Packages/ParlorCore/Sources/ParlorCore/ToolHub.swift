import Foundation

public struct ToolOutput: Sendable {
    public var text: String
    /// Paths relative to the parlor home.
    public var images: [String]
    public var isError: Bool
    public init(text: String, images: [String] = [], isError: Bool = false) {
        self.text = text; self.images = images; self.isError = isError
    }
}

public protocol ToolProvider: Sendable {
    var id: String { get }
    func listTools() async throws -> [ToolDef]
    func call(tool: String, arguments: JSON) async throws -> ToolOutput
}

public struct ServerStatus: Identifiable, Sendable, Equatable {
    public enum State: Sendable, Equatable { case connecting, ready, failed(String) }
    public var id: String
    public var kind: String          // "built in", "stdio", "http"
    public var detail: String
    public var state: State
    public var toolCount: Int
    public var toolNames: [String]
}

/// Union of every mounted provider's tools. Providers announce changes through `notifyChanged`.
public actor ToolHub {
    private var providers: [String: ToolProvider] = [:]
    private var statuses: [String: ServerStatus] = [:]
    private var cached: [String: [ToolDef]] = [:]
    private var listeners: [UUID: @Sendable () -> Void] = [:]
    public private(set) var version = 0

    public init() {}

    public func onChange(_ f: @escaping @Sendable () -> Void) -> UUID {
        let id = UUID(); listeners[id] = f; return id
    }

    public func setStatus(_ s: ServerStatus) {
        statuses[s.id] = s
        bump()
    }

    public func register(_ p: ToolProvider, status: ServerStatus) async {
        providers[p.id] = p
        var s = status
        do {
            let tools = try await p.listTools()
            cached[p.id] = tools
            s.state = .ready; s.toolCount = tools.count; s.toolNames = tools.map(\.name)
        } catch {
            cached[p.id] = []
            s.state = .failed(error.localizedDescription)
        }
        statuses[p.id] = s
        bump()
    }

    public func unregister(_ id: String) {
        providers[id] = nil; cached[id] = nil; statuses[id] = nil
        bump()
    }

    /// A provider calls this when its tool list changed (tools/list_changed).
    public func refresh(_ id: String) async {
        guard let p = providers[id] else { return }
        if let tools = try? await p.listTools() {
            cached[id] = tools
            if var s = statuses[id] { s.toolCount = tools.count; s.toolNames = tools.map(\.name); statuses[id] = s }
            bump()
        }
    }

    public func allTools() -> [ToolDef] { cached.values.flatMap { $0 }.sorted { $0.qualified < $1.qualified } }
    public func allStatuses() -> [ServerStatus] { statuses.values.sorted { $0.id < $1.id } }
    public func providerIDs() -> [String] { Array(providers.keys) }

    public func tools(for agent: Agent) -> [ToolDef] { allTools().filter { agent.allows($0.qualified) } }

    public func call(wireName: String, arguments: JSON, timeout: TimeInterval? = nil) async -> ToolOutput {
        guard let r = wireName.range(of: "__") else { return ToolOutput(text: "Unknown tool \(wireName)", isError: true) }
        let server = String(wireName[..<r.lowerBound]); let tool = String(wireName[r.upperBound...])
        guard let p = providers[server] else { return ToolOutput(text: "Server \(server) is not running", isError: true) }
        do {
            guard let timeout, timeout > 0 else { return try await p.call(tool: tool, arguments: arguments) }
            return try await withThrowingTaskGroup(of: ToolOutput.self) { group in
                group.addTask { try await p.call(tool: tool, arguments: arguments) }
                group.addTask {
                    try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                    return ToolOutput(text: "Tool \(server).\(tool) timed out after \(Int(timeout)) s.", isError: true)
                }
                defer { group.cancelAll() }
                return try await group.next()!
            }
        } catch { return ToolOutput(text: "Tool failed: \(error.localizedDescription)", isError: true) }
    }

    private func bump() {
        version += 1
        for l in listeners.values { l() }
    }
}

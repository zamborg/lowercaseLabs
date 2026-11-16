import Foundation

extension ParlorEngine {
    // MARK: tools

    /// The authoring guide and the Python helper, rewritten when this app version has newer ones.
    func seedToolFiles() {
        let fm = FileManager.default
        try? fm.createDirectory(at: home.toolsDir, withIntermediateDirectories: true)
        for (name, text) in [("CLAUDE.md", Seeds.toolGuide), ("parlor_tool.py", Seeds.toolHelper)] {
            let url = home.toolsDir.appendingPathComponent(name)
            let current = try? String(contentsOf: url, encoding: .utf8)
            if current == nil || !(current!.contains(Seeds.toolMarker)) { try? text.write(to: url, atomically: true, encoding: .utf8) }
        }
    }

    public func reloadToolList() {
        let found = Tools.scan(store: toolStore, home: home, approvals: settings.toolApprovals)
        if found != toolList { toolList = found }
    }

    /// Launch configs of approved tools, with a per-process token at the tier each asks for.
    func toolConfigs() -> [String: MCPServerConfig] {
        var out: [String: MCPServerConfig] = [:]
        for t in toolList where t.approved && t.spec != nil {
            var token: String?
            if t.spec!.tierValue != .none, endpoint.isRunning {
                if toolTokens[t.id] == nil { toolTokens[t.id] = endpoint.issueEphemeral(name: "tool:\(t.id)", tier: t.spec!.tierValue) }
                token = toolTokens[t.id]
            }
            let url = endpoint.isRunning ? "http://127.0.0.1:\(endpoint.port)/mcp" : nil
            if let c = Tools.config(for: t, parlorURL: url, token: token) { out[t.id] = c }
        }
        return out
    }

    private func applyTools() { reloadToolList(); Task { await self.mcp.reload(); await self.syncFromHub(); await self.refreshServices() } }

    /// The user reviewed what the tool runs and allows it.
    public func approveTool(_ id: String) {
        guard let t = toolList.first(where: { $0.id == id }), t.spec != nil else { return }
        updateSettings { $0.toolApprovals[id] = t.hash }
        applyTools()
    }

    public func revokeTool(_ id: String) {
        updateSettings { $0.toolApprovals[id] = nil }
        applyTools()
    }

    public func setToolValues(_ id: String, _ values: [String: JSON]) {
        try? toolStore.update(id, patch: .object(["values": .object(values)]))
        applyTools()
    }

    public func setTool(_ id: String, enabled: Bool) {
        try? toolStore.update(id, patch: .object(["enabled": .bool(enabled)]))
        applyTools()
    }

    /// Remove a tool's folder and its approval; its server stops.
    public func deleteTool(_ id: String) {
        try? toolStore.delete(id)
        updateSettings { $0.toolApprovals[id] = nil }
        applyTools()
    }

    public func restartServer(_ id: String) { Task { await mcp.restart(id); await refreshServices() } }
    public func reloadTools() { applyTools() }

    // MARK: access tokens

    /// Make a token. The secret is returned once; only its hash is stored.
    public func createToken(name: String, tier: Tier, expires: Date? = nil) -> String? {
        let n = name.trimmingCharacters(in: .whitespaces)
        guard !n.isEmpty, !settings.tokens.contains(where: { $0.name == n }) else { return nil }
        let (secret, token) = APIToken.make(name: n, tier: tier, expires: expires)
        updateSettings { $0.tokens.append(token) }
        return secret
    }

    public func revokeToken(_ name: String) { updateSettings { $0.tokens.removeAll { $0.name == name } } }
}

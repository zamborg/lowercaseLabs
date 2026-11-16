import Foundation

extension ParlorEngine {
    // MARK: modules

    /// The authoring guide and the Python helper, rewritten when this app version has newer ones.
    func seedModuleFiles() {
        let fm = FileManager.default
        try? fm.createDirectory(at: home.modulesDir, withIntermediateDirectories: true)
        for (name, text) in [("CLAUDE.md", Seeds.moduleGuide), ("parlor_module.py", Seeds.moduleHelper)] {
            let url = home.modulesDir.appendingPathComponent(name)
            let current = try? String(contentsOf: url, encoding: .utf8)
            if current == nil || !(current!.contains(Seeds.moduleMarker)) { try? text.write(to: url, atomically: true, encoding: .utf8) }
        }
    }

    public func reloadModules() {
        let found = Modules.scan(home: home, approvals: settings.moduleApprovals)
        if found != modules { modules = found }
    }

    /// Launch configs of approved modules, with a per-process token at the tier the manifest asks for.
    func moduleConfigs() -> [String: MCPServerConfig] {
        var out: [String: MCPServerConfig] = [:]
        for m in modules where m.approved && m.manifest != nil {
            var token: String?
            if m.manifest!.tier != .none, endpoint.isRunning {
                if moduleTokens[m.name] == nil { moduleTokens[m.name] = endpoint.issueEphemeral(name: "module:\(m.name)", tier: m.manifest!.tier) }
                token = moduleTokens[m.name]
            }
            let url = endpoint.isRunning ? "http://127.0.0.1:\(endpoint.port)/mcp" : nil
            if let c = Modules.config(for: m, parlorURL: url, token: token) { out[m.name] = c }
        }
        return out
    }

    /// The user reviewed the manifest and allows this module to run.
    public func approveModule(_ name: String) {
        guard let m = modules.first(where: { $0.name == name }), m.manifest != nil else { return }
        updateSettings { $0.moduleApprovals[name] = m.hash }
        reloadModules()
        Task { await self.mcp.reload(); await self.refreshServices() }
    }

    public func revokeModule(_ name: String) {
        updateSettings { $0.moduleApprovals[name] = nil }
        reloadModules()
        Task { await self.mcp.reload(); await self.refreshServices() }
    }

    public func setModuleValues(_ name: String, _ values: [String: JSON]) {
        guard let m = modules.first(where: { $0.name == name }) else { return }
        Modules.saveValues(m, values)
        reloadModules()
        Task { await self.mcp.reload(); await self.refreshServices() }
    }

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

    // MARK: tools

    public func restartServer(_ id: String) { Task { await mcp.restart(id); await refreshServices() } }
    public func reloadTools() { Task { await mcp.reload(); self.loadConfig() } }

    public func saveMCPConfig(_ file: MCPConfigFile) throws {
        try file.save(home.mcpConfig)
        configuredServers = file.mcpServers
        reloadTools()
    }

    public func loadConfig() { configuredServers = MCPConfigFile.load(home.mcpConfig).mcpServers }

    /// Turn a server off or on without losing its settings (`"disabled": true` in mcp.json).
    public func setServer(_ id: String, enabled: Bool) {
        var f = MCPConfigFile.load(home.mcpConfig)
        guard var cfg = f.mcpServers[id] else { return }
        cfg.disabled = enabled ? nil : true
        f.mcpServers[id] = cfg
        try? saveMCPConfig(f)
    }

    /// Remove a server from mcp.json and stop it.
    public func deleteServer(_ id: String) {
        var f = MCPConfigFile.load(home.mcpConfig)
        f.mcpServers[id] = nil
        try? saveMCPConfig(f)
    }
}

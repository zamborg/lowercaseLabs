import SwiftUI
import ParlorCore

/// Services, your data and machine settings in one place.
struct SystemView: View {
    @Environment(ParlorEngine.self) private var engine

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                servicesSection
                providersSection
                modulesSection
                accessSection
                dataSection
                ConsoleSection()
            }
            .padding(24)
            .frame(maxWidth: 900, alignment: .leading)
        }
        .navigationTitle("System")
        .task {
            while !Task.isCancelled { await engine.refreshServices(); try? await Task.sleep(nanoseconds: 2_000_000_000) }
        }
        .accessibilityIdentifier("system-view")
    }

    // MARK: services

    private var servicesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Services").font(.headline)
            Text("Everything the app keeps running for you. Servers set to on demand start when an agent first uses them and stop after their idle time; change that in mcp.json with \"lifecycle\" and \"idleMinutes\".")
                .font(.caption).foregroundStyle(.secondary)
            ForEach(engine.services) { s in serviceCard(s) }
            if engine.services.isEmpty { ProgressView().controlSize(.small) }
        }
    }

    private func serviceCard(_ s: ServiceInfo) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Circle().fill(color(s.state)).frame(width: 9, height: 9)
                Text(s.title).font(.headline)
                Text(s.kind).font(.caption).padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule())
                Text(s.state.label).font(.caption).foregroundStyle(s.state == .stopped ? Color.secondary : color(s.state))
                if let l = s.lifecycle, l == "onDemand" { Text("on demand").font(.caption).foregroundStyle(.secondary) }
                Spacer()
                if let m = s.memoryBytes { Text(ByteCountFormatter.string(fromByteCount: Int64(m), countStyle: .memory)).font(.caption.monospacedDigit()).foregroundStyle(.secondary).help("Memory this service is using") }
                if let q = s.queueDepth, q > 0 { Text("\(q) waiting").font(.caption).foregroundStyle(.orange) }
                if let t = s.lastUsed { Text(t, style: .relative).font(.caption).foregroundStyle(.secondary).help("Last used") }
                controls(s)
            }
            Text(s.detail).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            if case .failed(let msg) = s.state { Text(msg).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            if s.restarts > 0 { Text("restarted \(s.restarts)× after a crash").font(.caption).foregroundStyle(.orange) }
        }
        .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("service-\(s.id)")
    }

    @ViewBuilder private func controls(_ s: ServiceInfo) -> some View {
        if s.canStart { Button("Start") { engine.startService(s.id) }.controlSize(.small).accessibilityIdentifier("start-\(s.id)") }
        if s.canStop { Button(s.id == "llm" ? "Unload" : "Stop") { engine.stopService(s.id) }.controlSize(.small).accessibilityIdentifier("stop-\(s.id)") }
        if s.canRestart { Button("Restart") { engine.restartService(s.id) }.controlSize(.small).accessibilityIdentifier("restart-\(s.id)") }
        if let log = s.logURL {
            Button { NSWorkspace.shared.open(log) } label: { Image(systemName: "doc.text") }
                .buttonStyle(.borderless).help("Open log").accessibilityIdentifier("log-\(s.id)")
                .disabled(!FileManager.default.fileExists(atPath: log.path))
        }
    }

    private func color(_ s: ServiceState) -> Color {
        switch s {
        case .running: .green
        case .idle: .mint
        case .starting: .orange
        case .stopped: .gray
        case .failed: .red
        }
    }

    // MARK: providers

    @State private var providerResults: [String: String] = [:]

    private var providersSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model providers").font(.headline)
            Text("Where language models come from. MLX runs inside the app. Others are OpenAI-compatible servers (llama.cpp, LM Studio, Ollama, a hosted API), listed under \"providers\" in settings.json. An agent uses one with model: provider/name.")
                .font(.caption).foregroundStyle(.secondary)
            providerRow(id: "mlx", detail: "built in · \(engine.library.installed(kind: .language).count) models installed", remote: false, enabled: true, toggle: nil)
            ForEach(engine.settings.providers) { c in
                providerRow(id: c.id, detail: c.problem ?? c.baseURL + (c.service != nil ? " · starts on demand" : ""), remote: c.locality == .remote,
                            enabled: c.enabled, toggle: { engine.setProviderEnabled(c.id, $0) })
            }
        }
    }

    private func providerRow(id: String, detail: String, remote: Bool, enabled: Bool, toggle: ((Bool) -> Void)?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text(id).font(.headline)
                if remote { Label("remote", systemImage: "network").font(.caption).padding(.horizontal, 7).padding(.vertical, 1).background(.orange.opacity(0.2), in: Capsule()).help("Prompts sent to this provider leave your Mac") }
                else { Text("local").font(.caption).padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule()) }
                Spacer()
                if id != "mlx", enabled {
                    Button("Test") { Task { providerResults[id] = "Testing…"; providerResults[id] = await engine.testProvider(id) } }
                        .controlSize(.small).accessibilityIdentifier("test-provider-\(id)")
                }
                if let toggle {
                    Toggle("", isOn: Binding(get: { enabled }, set: toggle)).labelsHidden().toggleStyle(.switch).controlSize(.small)
                        .accessibilityIdentifier("provider-enabled-\(id)")
                }
            }
            Text(detail).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            if let r = providerResults[id] { Text(r).font(.caption).foregroundStyle(r.hasPrefix("Reachable") ? Color.green : Color.secondary).textSelection(.enabled) }
        }
        .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .contain).accessibilityIdentifier("provider-\(id)")
    }

    // MARK: modules

    @State private var editing: String?

    private var modulesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Modules").font(.headline)
            Text("Small tools in your parlor folder's modules/ directory, written by you or a model (see modules/CLAUDE.md). A module runs only after you approve it, and again if its manifest changes.")
                .font(.caption).foregroundStyle(.secondary)
            if engine.modules.isEmpty { Text("None yet.").font(.caption).foregroundStyle(.secondary) }
            ForEach(engine.modules) { m in moduleCard(m) }
        }
    }

    private func moduleCard(_ m: ModuleInfo) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: m.approved ? "checkmark.shield" : "exclamationmark.shield").foregroundStyle(m.approved ? Color.green : .orange)
                Text(m.name).font(.headline)
                Text(m.approved ? "approved" : (m.error == nil ? "needs approval" : "invalid")).font(.caption).foregroundStyle(.secondary)
                Spacer()
                if m.manifest != nil, !m.fields.isEmpty, m.approved {
                    Button("Settings") { editing = editing == m.name ? nil : m.name }.controlSize(.small)
                }
                if m.approved { Button("Revoke") { engine.revokeModule(m.name) }.controlSize(.small).accessibilityIdentifier("revoke-\(m.name)") }
                else if m.manifest != nil { Button("Approve") { engine.approveModule(m.name) }.controlSize(.small).buttonStyle(.borderedProminent).accessibilityIdentifier("approve-\(m.name)") }
                Button { NSWorkspace.shared.activateFileViewerSelecting([m.dir]) } label: { Image(systemName: "folder") }.buttonStyle(.borderless).help("Show in Finder")
            }
            if let e = m.error { Text(e).font(.caption).foregroundStyle(.red) }
            if let man = m.manifest {
                if let d = man.description { Text(d).font(.callout) }
                Text("Runs: \(([man.command] + (man.args ?? [])).joined(separator: " "))").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
                HStack(spacing: 10) {
                    Label(man.needs?.network == true ? "Network" : "No network", systemImage: "network").font(.caption).foregroundStyle(.secondary)
                    Label("Files: " + (man.needs?.files ?? "none"), systemImage: "doc").font(.caption).foregroundStyle(.secondary)
                    Label("App control: " + man.tier.rawValue, systemImage: "key").font(.caption).foregroundStyle(man.tier == .none ? Color.secondary : .orange)
                }
                Text("Modules are not sandboxed yet: these are what the module says it needs. Read its code before approving.").font(.caption2).foregroundStyle(.secondary)
            }
            if editing == m.name { ModuleSettingsForm(module: m) }
        }
        .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("module-\(m.name)")
    }

    // MARK: access

    @State private var showNewToken = false
    @State private var newSecret: String?
    @State private var newName = ""
    @State private var newTier: Tier = .act

    private var accessSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Access").font(.headline)
            Text("The Parlor server lets Claude Code or a script on this Mac drive the app: read and change agents, talk in rooms, start services. It only listens on 127.0.0.1 and needs a token. A token's tier limits what it may do; deleting anything always asks you first.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Toggle("Parlor server", isOn: Binding(get: { engine.settings.parlorServer.enabled }, set: { v in engine.updateSettings { $0.parlorServer.enabled = v } }))
                    .accessibilityIdentifier("server-toggle")
                Text("http://127.0.0.1:\(engine.settings.parlorServer.port)/mcp").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
                Spacer()
                Button("New Token…") { newName = ""; newTier = .act; newSecret = nil; showNewToken = true }.controlSize(.small).accessibilityIdentifier("new-token")
            }
            ForEach(engine.settings.tokens) { t in
                HStack {
                    Image(systemName: "key.fill").foregroundStyle(.secondary)
                    Text(t.name).fontWeight(.medium)
                    Text(t.tier.rawValue).font(.caption).padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule())
                    Text("created \(t.created.formatted(date: .abbreviated, time: .omitted))").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Revoke", role: .destructive) { engine.revokeToken(t.name) }.controlSize(.small)
                }
                .padding(.horizontal, 12).padding(.vertical, 6).background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
            }
        }
        .sheet(isPresented: $showNewToken) { tokenSheet }
    }

    private var tokenSheet: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(newSecret == nil ? "New token" : "Copy your token now").font(.title3.bold())
            if let secret = newSecret {
                Text("It is shown once. Parlor keeps only a hash of it.").font(.caption).foregroundStyle(.secondary)
                Text(secret).font(.system(.callout, design: .monospaced)).textSelection(.enabled).padding(10).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                Text("Add it to Claude Code:").font(.caption).foregroundStyle(.secondary)
                let cmd = "claude mcp add --transport http parlor http://127.0.0.1:\(engine.settings.parlorServer.port)/mcp --header \"Authorization: Bearer \(secret)\""
                Text(cmd).font(.system(.caption, design: .monospaced)).textSelection(.enabled).padding(10).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                HStack {
                    Button("Copy Token") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(secret, forType: .string) }
                    Button("Copy Command") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(cmd, forType: .string) }
                    Spacer()
                    Button("Done") { showNewToken = false }.keyboardShortcut(.defaultAction)
                }
            } else {
                TextField("Name (e.g. claude-code)", text: $newName).textFieldStyle(.roundedBorder).accessibilityIdentifier("token-name")
                Picker("Tier", selection: $newTier) {
                    Text("Read: look only").tag(Tier.read)
                    Text("Act: talk in rooms, start services").tag(Tier.act)
                    Text("Configure: edit agents and settings").tag(Tier.configure)
                }
                .accessibilityIdentifier("token-tier")
                HStack {
                    Button("Cancel") { showNewToken = false }
                    Spacer()
                    Button("Create") { newSecret = engine.createToken(name: newName, tier: newTier) }
                        .keyboardShortcut(.defaultAction).disabled(newName.trimmingCharacters(in: .whitespaces).isEmpty)
                        .accessibilityIdentifier("create-token")
                }
            }
        }
        .padding(20).frame(width: 520)
    }

    // MARK: your data

    private var dataSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Your data").font(.headline)
            Text("All of it is plain files. Open any of these in Finder, or point Claude Code at the folder.")
                .font(.caption).foregroundStyle(.secondary)
            let h = engine.home
            let items: [(String, String, URL)] = [
                ("Parlor folder", "folder", h.root),
                ("Agents", "person.2", h.agentsDir),
                ("Chats", "bubble.left.and.bubble.right", h.chatsDir),
                ("Models", "cpu", h.modelsDir),
                ("Images", "photo.on.rectangle", h.mediaDir),
                ("Logs", "doc.text", h.logsDir),
                ("mcp.json", "wrench.and.screwdriver", h.mcpConfig),
                ("settings.json", "gearshape", h.settingsFile),
            ]
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 170), spacing: 8)], alignment: .leading, spacing: 8) {
                ForEach(items, id: \.0) { title, symbol, url in
                    Button { reveal(url) } label: {
                        Label(title, systemImage: symbol).frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .controlSize(.large)
                    .help(url.path)
                    .accessibilityIdentifier("data-\(title)")
                }
            }
            Text(h.root.path).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
        }
    }

    private func reveal(_ url: URL) {
        let fm = FileManager.default
        try? fm.createDirectory(at: url.hasDirectoryPath || !url.lastPathComponent.contains(".") ? url : url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fm.fileExists(atPath: url.path) { NSWorkspace.shared.activateFileViewerSelecting([url]) }
        else { NSWorkspace.shared.open(url.deletingLastPathComponent()) }
    }
}


/// A form for a module's JSON-Schema settings.
struct ModuleSettingsForm: View {
    @Environment(ParlorEngine.self) private var engine
    let module: ModuleInfo
    @State private var values: [String: JSON] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(module.fields) { f in
                HStack {
                    Text(f.title).frame(width: 220, alignment: .leading)
                    switch f.type {
                    case "boolean":
                        Toggle("", isOn: Binding(get: { values[f.key]?.boolValue ?? false }, set: { values[f.key] = .bool($0) })).labelsHidden()
                    case "integer":
                        TextField("", value: Binding(get: { values[f.key]?.intValue ?? 0 }, set: { values[f.key] = .int($0) }), format: .number).frame(width: 90)
                    case "number":
                        TextField("", value: Binding(get: { values[f.key]?.doubleValue ?? 0 }, set: { values[f.key] = .double($0) }), format: .number).frame(width: 90)
                    default:
                        TextField("", text: Binding(get: { values[f.key]?.stringValue ?? "" }, set: { values[f.key] = .string($0) })).textFieldStyle(.roundedBorder)
                    }
                }
            }
            Button("Save and Restart") { engine.setModuleValues(module.name, values) }.controlSize(.small)
        }
        .onAppear { values = module.values }
    }
}

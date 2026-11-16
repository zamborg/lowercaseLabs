import SwiftUI
import ParlorCore

/// Services, your data and machine settings in one place.
struct SystemView: View {
    @Environment(ParlorEngine.self) private var engine

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                servicesSection
                ToolsSection()
                accessSection
                dataSection
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
                Toggle("Parlor server", isOn: Binding(get: { engine.settings.gateway.enabled }, set: { v in engine.updateSettings { $0.gateway.enabled = v } }))
                    .accessibilityIdentifier("server-toggle")
                Text("http://127.0.0.1:\(engine.settings.gateway.port)/mcp").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
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
                let cmd = "claude mcp add --transport http parlor http://127.0.0.1:\(engine.settings.gateway.port)/mcp --header \"Authorization: Bearer \(secret)\""
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
                ("Tools", "wrench.and.screwdriver", h.toolsDir),
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

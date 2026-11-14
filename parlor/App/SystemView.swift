import SwiftUI
import ParlorCore

/// Services, your data and machine settings in one place.
struct SystemView: View {
    @Environment(ParlorEngine.self) private var engine

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                servicesSection
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

import SwiftUI
import ParlorCore

/// Tool servers: folders in tools/ (tool.json plus a program) and the built-in servers. A panel inside System.
/// Each tool runs only after you approve it, and again if what it runs changes.
struct ToolsSection: View {
    @Environment(ParlorEngine.self) private var engine
    @State private var editing: String?
    @State private var pendingDelete: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Tools").font(.headline)
                Text("\(engine.tools.count)").font(.caption).foregroundStyle(.secondary).monospacedDigit()
                Spacer()
                Button { NSWorkspace.shared.open(engine.home.toolsDir) } label: { Label("Open tools folder", systemImage: "folder") }.controlSize(.small)
            }
            Text("Each tool is a folder in tools/ with a tool.json (see tools/CLAUDE.md), written by you or a model. It runs only after you approve it, and again if its command changes. Tools are not sandboxed: read the code before approving.")
                .font(.caption).foregroundStyle(.secondary)
            ForEach(engine.servers.filter { $0.kind == "built in" }) { s in builtInCard(s) }
            ForEach(engine.toolList) { t in toolCard(t) }
            if engine.servers.isEmpty && engine.toolList.isEmpty { ProgressView().controlSize(.small) }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("tools-section")
        .confirmationDialog("Remove \(pendingDelete ?? "")?", isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } })) {
            Button("Remove Tool", role: .destructive) { if let id = pendingDelete { Task { _ = await runVerb(engine, "tools.delete", ["id": .string(id)]) } }; pendingDelete = nil }
        } message: { Text("Its folder is deleted and it stops running. Agents that listed its tools simply lose them.") }
    }

    private func status(_ id: String) -> ServerStatus? { engine.servers.first { $0.id == id } }

    private func builtInCard(_ s: ServerStatus) -> some View {
        HStack(spacing: 8) {
            Circle().fill(color(s.state)).frame(width: 9, height: 9)
            Text(s.id).font(.headline)
            Text("built in").font(.caption).padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule())
            Spacer()
            Text("\(s.toolCount) \(s.toolCount == 1 ? "tool" : "tools")").foregroundStyle(.secondary).font(.caption)
        }
        .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityIdentifier("server-\(s.id)")
    }

    private func toolCard(_ t: ToolInfo) -> some View {
        let s = status(t.id)
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                if let s { Circle().fill(color(s.state)).frame(width: 9, height: 9) }
                else { Image(systemName: t.approved ? "checkmark.shield" : "exclamationmark.shield").foregroundStyle(t.approved ? Color.green : .orange) }
                Text(t.id).font(.headline)
                Text(label(t, s)).font(.caption).foregroundStyle(.secondary)
                Spacer()
                if let s, case .ready = s.state { Text("\(s.toolCount) \(s.toolCount == 1 ? "tool" : "tools")").foregroundStyle(.secondary).font(.caption) }
                if let spec = t.spec {
                    if t.approved {
                        if !t.fields.isEmpty { Button("Settings") { editing = editing == t.id ? nil : t.id }.controlSize(.small) }
                        Button("Restart") { Task { _ = await runVerb(engine, "services.restart", ["id": .string("tool:" + t.id)]) } }.controlSize(.small)
                        Toggle("Enabled", isOn: Binding(get: { spec.enabled }, set: { v in Task { _ = await runVerb(engine, "tools.enable", ["id": .string(t.id), "enabled": .bool(v)]) } }))
                            .toggleStyle(.switch).labelsHidden().help(spec.enabled ? "Turn off" : "Turn on")
                        Button("Revoke") { Task { _ = await runVerb(engine, "tools.revoke", ["id": .string(t.id)]) } }.controlSize(.small).accessibilityIdentifier("revoke-\(t.id)")
                    } else {
                        Button("Approve") { Task { _ = await runVerb(engine, "tools.approve", ["id": .string(t.id)]) } }.controlSize(.small).buttonStyle(.borderedProminent).accessibilityIdentifier("approve-\(t.id)")
                    }
                }
                Button { NSWorkspace.shared.activateFileViewerSelecting([t.dir]) } label: { Image(systemName: "folder") }.buttonStyle(.borderless).help("Show in Finder")
                Button(role: .destructive) { pendingDelete = t.id } label: { Image(systemName: "trash") }
                    .buttonStyle(.borderless).help("Delete this tool").accessibilityIdentifier("delete-server-\(t.id)")
            }
            if let e = t.error { Text(e).font(.caption).foregroundStyle(.red) }
            if let s, case .failed(let msg) = s.state { Text(msg).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            if let spec = t.spec {
                if !spec.description.isEmpty { Text(spec.description).font(.callout) }
                let run = spec.transport == "http" ? (spec.url ?? "") : ([spec.command ?? ""] + spec.args).joined(separator: " ")
                Text("Runs: \(run)").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
                Label("App control: " + spec.tier, systemImage: "key").font(.caption).foregroundStyle(spec.tier == "none" ? Color.secondary : .orange)
            }
            if let s, !s.toolNames.isEmpty {
                Text(s.toolNames.joined(separator: "  ·  ")).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
            }
            if editing == t.id { ToolSettingsForm(tool: t) }
        }
        .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("tool-\(t.id)")
    }

    private func label(_ t: ToolInfo, _ s: ServerStatus?) -> String {
        if t.spec == nil { return "invalid" }
        if !t.approved { return "needs approval" }
        if t.spec?.enabled == false { return "off" }
        guard let s else { return "stopped" }
        switch s.state { case .ready: return "ready"; case .connecting: return "starting"; case .failed: return "failed"; case .stopped: return "on demand" }
    }

    private func color(_ s: ServerStatus.State) -> Color {
        switch s { case .ready: .green; case .connecting: .orange; case .failed: .red; case .stopped: .gray }
    }
}

/// A form for a tool's JSON-Schema settings.
struct ToolSettingsForm: View {
    @Environment(ParlorEngine.self) private var engine
    let tool: ToolInfo
    @State private var values: [String: JSON] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(tool.fields) { f in
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
            Button("Save and Restart") { Task { _ = await runVerb(engine, "tools.values", ["id": .string(tool.id), "values": .object(values)]) } }.controlSize(.small)
        }
        .onAppear { values = tool.effectiveValues }
    }
}

import SwiftUI
import ParlorCore

/// MCP servers and the tools they give agents. A panel inside System.
struct ToolsSection: View {
    @Environment(ParlorEngine.self) private var engine
    @State private var showJSON = false
    @State private var showAdd = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Tools").font(.headline)
                Text("\(engine.tools.count)").font(.caption).foregroundStyle(.secondary).monospacedDigit()
                Spacer()
                Button("Edit mcp.json") { showJSON = true }.controlSize(.small)
                Button { showAdd = true } label: { Label("Add Server", systemImage: "plus") }.controlSize(.small)
                    .accessibilityIdentifier("add-server")
            }
            Text("Servers mounted here give every agent new tools. A fresh chat turn picks up the current list; open chats get a note when it changes.")
                .font(.caption).foregroundStyle(.secondary)
            ForEach(engine.servers.filter { engine.configuredServers[$0.id] != nil || $0.kind == "built in" }) { s in serverCard(s) }
            ForEach(disabledIDs, id: \.self) { disabledCard($0) }
            if engine.servers.isEmpty && engine.configuredServers.isEmpty { ProgressView().controlSize(.small) }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("tools-section")
        .confirmationDialog("Remove \(pendingDelete ?? "")?", isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } })) {
            Button("Remove Server", role: .destructive) { if let id = pendingDelete { engine.deleteServer(id) }; pendingDelete = nil }
        } message: { Text("It is removed from mcp.json and stops running. Agents that listed its tools simply lose them.") }
        .sheet(isPresented: $showJSON) { MCPJSONEditor() }
        .sheet(isPresented: $showAdd) { AddServerSheet() }
    }

    @State private var pendingDelete: String?

    private var disabledIDs: [String] {
        engine.configuredServers.filter { $0.value.disabled == true }.map(\.key).sorted()
    }

    private func disabledCard(_ id: String) -> some View {
        HStack(spacing: 8) {
            Circle().strokeBorder(.secondary, lineWidth: 1).frame(width: 9, height: 9)
            Text(id).font(.headline).foregroundStyle(.secondary)
            Text("disabled").font(.caption).padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule())
            Spacer()
            Toggle("Enabled", isOn: Binding(get: { false }, set: { _ in engine.setServer(id, enabled: true) }))
                .toggleStyle(.switch).labelsHidden().help("Turn on")
            deleteButton(id)
        }
        .padding(14).background(.background.secondary.opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
        .accessibilityIdentifier("server-\(id)-disabled")
    }

    private func deleteButton(_ id: String) -> some View {
        Button(role: .destructive) { pendingDelete = id } label: { Image(systemName: "trash") }
            .buttonStyle(.borderless).help("Remove from mcp.json")
            .accessibilityIdentifier("delete-server-\(id)")
    }

    private func serverCard(_ s: ServerStatus) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Circle().fill(color(s.state)).frame(width: 9, height: 9)
                Text(s.id).font(.headline)
                Text(s.kind).font(.caption).padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule())
                Text(s.detail).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).lineLimit(1)
                Spacer()
                switch s.state {
                case .connecting: ProgressView().controlSize(.small)
                case .ready: Text("\(s.toolCount) \(s.toolCount == 1 ? "tool" : "tools")").foregroundStyle(.secondary).font(.caption)
                case .failed: Text("failed").foregroundStyle(.red).font(.caption)
                case .stopped: Text("stopped").foregroundStyle(.secondary).font(.caption)
                }
                if s.kind != "built in" {
                    Button("Restart") { engine.restartServer(s.id) }
                    Toggle("Enabled", isOn: Binding(get: { true }, set: { _ in engine.setServer(s.id, enabled: false) }))
                        .toggleStyle(.switch).labelsHidden().help("Turn off")
                    deleteButton(s.id)
                }
            }
            if case .failed(let msg) = s.state { Text(msg).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            if !s.toolNames.isEmpty {
                Text(s.toolNames.joined(separator: "  ·  ")).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
            }
        }
        .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityIdentifier("server-\(s.id)")
    }

    private func color(_ s: ServerStatus.State) -> Color {
        switch s { case .ready: .green; case .connecting: .orange; case .failed: .red; case .stopped: .gray }
    }
}

struct MCPJSONEditor: View {
    @Environment(ParlorEngine.self) private var engine
    @Environment(\.dismiss) private var dismiss
    @State private var text = ""
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("mcp.json").font(.headline)
            Text("Same format as Claude Desktop. Paste recipes straight in.").font(.caption).foregroundStyle(.secondary)
            TextEditor(text: $text).font(.system(.body, design: .monospaced)).frame(minWidth: 560, minHeight: 320)
                .padding(6).background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
            if let error { Text(error).foregroundStyle(.red).font(.caption) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save") {
                    do {
                        let f = try JSONDecoder().decode(MCPConfigFile.self, from: Data(text.utf8))
                        try engine.saveMCPConfig(f); dismiss()
                    } catch { self.error = "Not valid mcp.json: \(error.localizedDescription)" }
                }.buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .onAppear { text = (try? String(contentsOf: engine.home.mcpConfig, encoding: .utf8)) ?? "{ \"mcpServers\": {} }" }
    }
}

struct AddServerSheet: View {
    @Environment(ParlorEngine.self) private var engine
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var commandLine = ""
    @State private var url = ""
    @State private var remote = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add MCP server").font(.headline)
            Picker("Type", selection: $remote) { Text("Local command").tag(false); Text("HTTP URL").tag(true) }.pickerStyle(.segmented)
            TextField("Name (letters and digits, e.g. calendar)", text: $name)
            if remote { TextField("https://…/mcp", text: $url) }
            else { TextField("Command, e.g. uv run --project mcp/zass zass-mcp", text: $commandLine) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Add", action: add)
                .buttonStyle(.borderedProminent)
                .disabled(name.isEmpty || (remote ? url.isEmpty : commandLine.isEmpty))
            }
        }
        .textFieldStyle(.roundedBorder).padding(20).frame(width: 520)
    }

    private func add() {
        var f = MCPConfigFile.load(engine.home.mcpConfig)
        let id = Agent.slug(name).replacingOccurrences(of: "-", with: "")
        if remote { f.mcpServers[id] = MCPServerConfig(url: url) }
        else {
            let parts = commandLine.split(separator: " ").map(String.init)
            guard let cmd = parts.first else { return }
            f.mcpServers[id] = MCPServerConfig(command: cmd, args: Array(parts.dropFirst()))
        }
        try? engine.saveMCPConfig(f)
        dismiss()
    }
}

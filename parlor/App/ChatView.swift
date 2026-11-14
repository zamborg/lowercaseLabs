import SwiftUI
import ParlorCore

/// Chat or edit page for one agent.
struct AgentPage: View {
    @Environment(ParlorEngine.self) private var engine
    let agentID: String
    @Binding var route: Route?
    @State private var mode: Mode = ProcessInfo.processInfo.environment["PARLOR_MODE"] == "edit" ? .edit : .chat

    enum Mode: String, CaseIterable, Identifiable { case chat = "Chat", edit = "Agent"; var id: String { rawValue } }

    var body: some View {
        Group {
            if let agent = engine.agent(agentID) {
                switch mode {
                case .chat: ChatView(agent: agent, route: $route, mode: $mode)
                case .edit: AgentEditorView(agentID: agentID)
                }
            }
        }
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker("Mode", selection: $mode) {
                    ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(width: 160)
                .accessibilityIdentifier("mode-picker")
            }
        }
        .navigationTitle(engine.agent(agentID)?.name ?? "")
    }
}

struct ChatView: View {
    @Environment(ParlorEngine.self) private var engine
    let agent: Agent
    @Binding var route: Route?
    @Binding var mode: AgentPage.Mode

    @State private var draft = ""
    @State private var selecting = false
    @State private var selected: Set<UUID> = []
    @State private var confirmReset = false
    @State private var showTools = false
    @FocusState private var composerFocused: Bool

    private var room: Room { engine.room(for: agent.id) }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            messageList
            if selecting { pruneBar } else { composer }
        }
        .toolbar {
            ToolbarItemGroup {
                Menu {
                    Button(selecting ? "Done Selecting" : "Select Messages…") { selecting.toggle(); selected = [] }
                    Button("Compact Now") { room.compactNow(agent: agent) }
                        .disabled(room.isBusy || room.messages.count < 8)
                    Divider()
                    Button("Clear Chat…", role: .destructive) { confirmReset = true }
                } label: { Label("Prune", systemImage: "scissors") }
                .help("Prune this chat")
                .accessibilityIdentifier("prune-menu")
            }
        }
        .confirmationDialog("Clear the whole chat with \(agent.name)?", isPresented: $confirmReset) {
            Button("Clear Chat", role: .destructive) { room.reset(); selecting = false }
        } message: { Text("This deletes every message and the summary. It cannot be undone.") }
    }

    // MARK: header

    private var header: some View {
        HStack(spacing: 10) {
            AvatarView(agent: agent, size: 30)
            VStack(alignment: .leading, spacing: 1) {
                Text(agent.name).font(.headline)
                if !agent.summary.isEmpty { Text(agent.summary).font(.caption).foregroundStyle(.secondary).lineLimit(1) }
            }
            Spacer()
            modelChip
            Button { showTools.toggle() } label: {
                Label({ let n = engine.tools.filter { agent.allows($0.qualified) }.count; return "\(n) \(n == 1 ? "tool" : "tools")" }(), systemImage: "wrench.and.screwdriver")
                    .font(.caption)
            }
            .buttonStyle(.bordered).controlSize(.small)
            .popover(isPresented: $showTools) { toolsPopover }
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
    }

    private var modelChip: some View {
        let installed = engine.library.installedDirectory(for: agent.model) != nil
        return Button { route = .models } label: {
            HStack(spacing: 5) {
                Circle().fill(installed ? (engine.loadedModel == agent.model ? Color.green : Color.secondary) : Color.orange).frame(width: 7, height: 7)
                Text(shortModelName(agent.model)).font(.caption)
                if !installed { Text("not downloaded").font(.caption).foregroundStyle(.orange) }
            }
        }
        .buttonStyle(.bordered).controlSize(.small)
        .help(installed ? agent.model : "Download this model in Models")
    }

    private var toolsPopover: some View {
        let list = engine.tools.filter { agent.allows($0.qualified) }
        return VStack(alignment: .leading, spacing: 6) {
            Text("Tools for \(agent.name)").font(.headline)
            if list.isEmpty { Text("None enabled. Turn tools on in the Agent tab.").foregroundStyle(.secondary) }
            ForEach(list) { t in
                VStack(alignment: .leading, spacing: 0) {
                    Text(t.qualified).font(.system(.callout, design: .monospaced))
                    Text(t.description).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }
            }
        }.padding(14).frame(width: 340)
    }

    // MARK: messages

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if room.messages.isEmpty && !room.isBusy { emptyState }
                    ForEach(room.messages) { m in
                        MessageRow(message: m, agent: agent, room: room, selecting: selecting, selected: selected.contains(m.id)) {
                            if selected.contains(m.id) { selected.remove(m.id) } else { selected.insert(m.id) }
                        }
                        .id(m.id)
                    }
                    if room.isBusy { liveRow.id("live") }
                    if let e = room.errorText {
                        Label(e, systemImage: "exclamationmark.triangle.fill")
                            .font(.callout).foregroundStyle(.orange)
                            .padding(10).background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
                            .textSelection(.enabled)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(.horizontal, 20).padding(.vertical, 14)
            }
            .defaultScrollAnchor(.bottom)
            .onChange(of: room.messages.count) { withAnimation { proxy.scrollTo("bottom") } }
            .onChange(of: room.streaming) { proxy.scrollTo("bottom") }
            .onChange(of: room.isBusy) { proxy.scrollTo("bottom") }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            AvatarView(agent: agent, size: 56)
            Text("This is the start of your chat with \(agent.name).").foregroundStyle(.secondary)
            if engine.library.installedDirectory(for: agent.model) == nil {
                Text("Download \(shortModelName(agent.model)) to begin.").font(.callout)
                Button("Open Models") { route = .models }.buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity).padding(.top, 60)
    }

    private var liveRow: some View {
        HStack(alignment: .top, spacing: 8) {
            AvatarView(agent: agent, size: 26)
            VStack(alignment: .leading, spacing: 4) {
                if room.streaming.isEmpty {
                    HStack(spacing: 8) { TypingDots(); if let a = room.activity { Text(a).font(.caption).foregroundStyle(.secondary) } }
                        .padding(.horizontal, 12).padding(.vertical, 9)
                        .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
                } else {
                    Text(markdown(room.streaming)).textSelection(.enabled)
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
                    if let a = room.activity { Text(a).font(.caption).foregroundStyle(.secondary).padding(.leading, 4) }
                }
            }
            Spacer(minLength: 60)
        }
        .accessibilityIdentifier("live-row")
    }

    // MARK: composer

    private var composer: some View {
        VStack(spacing: 6) {
            HStack(alignment: .bottom, spacing: 8) {
                TextField("Message \(agent.name)…", text: $draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...8)
                    .focused($composerFocused)
                    .onSubmit(send)
                    .padding(.horizontal, 12).padding(.vertical, 9)
                    .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(.separator))
                    .accessibilityIdentifier("composer")
                if room.isBusy {
                    Button { room.stop() } label: { Image(systemName: "stop.fill") }
                        .buttonStyle(.borderedProminent).tint(.red).controlSize(.large)
                        .help("Stop").accessibilityIdentifier("stop-button")
                } else {
                    Button(action: send) { Image(systemName: "arrow.up") }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        .keyboardShortcut(.return, modifiers: .command)
                        .help("Send (↩)").accessibilityIdentifier("send-button")
                }
            }
            HStack {
                Text(room.summaryText == nil ? "Return to send, Shift-Return for a new line" : "Earlier messages are summarized")
                Spacer()
                if let s = room.lastStats { Text("\(Int(s.tokensPerSecond)) tok/s · ").monospacedDigit() }
                Text("context \(tokenString(room.contextUsed)) / \(tokenString(agent.contextTokens))").monospacedDigit()
            }
            .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 20).padding(.vertical, 12)
        .onAppear { composerFocused = true }
    }

    private func send() {
        let text = draft
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !room.isBusy else { return }
        draft = ""
        engine.send(text, to: agent.id)
    }

    // MARK: prune

    private var pruneBar: some View {
        HStack(spacing: 10) {
            Text(selected.isEmpty ? "Select messages to prune" : "\(selected.count) selected").foregroundStyle(.secondary)
            Spacer()
            Button("Delete Selected") { room.delete(ids: selected); selected = [] }
                .disabled(selected.isEmpty).accessibilityIdentifier("delete-selected")
            Button("Cut Everything After") {
                if let id = selected.first { room.cut(after: id); selected = [] }
            }.disabled(selected.count != 1).accessibilityIdentifier("cut-after")
            Button("Compact Now") { room.compactNow(agent: agent) }.disabled(room.isBusy)
            Button("Done") { selecting = false; selected = [] }.buttonStyle(.borderedProminent)
        }
        .padding(.horizontal, 20).padding(.vertical, 12)
        .background(.bar)
    }
}

func markdown(_ s: String) -> AttributedString {
    let opts = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
    return (try? AttributedString(markdown: s, options: opts)) ?? AttributedString(s)
}

struct TypingDots: View {
    @State private var phase = false
    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<3) { i in
                Circle().fill(.secondary).frame(width: 5, height: 5)
                    .opacity(phase ? 1 : 0.3)
                    .animation(.easeInOut(duration: 0.6).repeatForever().delay(Double(i) * 0.15), value: phase)
            }
        }.onAppear { phase = true }
    }
}

import SwiftUI
import ParlorCore

/// Edits one agent file. Changes save to `agents/<id>.md` and external edits flow back in.
struct AgentEditorView: View {
    @Environment(ParlorEngine.self) private var engine
    let agentID: String
    @State private var draft: Agent = Agent(id: "", name: "")
    @State private var loaded = false
    @State private var lastSaved: Agent?
    @State private var saveTask: Task<Void, Never>?
    @State private var memoryText = ""
    @State private var memoryLoaded = ""
    @State private var memoryTask: Task<Void, Never>?

    var body: some View {
        HStack(spacing: 0) {
            form.frame(maxWidth: .infinity)
            Divider()
            filePane.frame(width: 290)
        }
        .onAppear {
            if !loaded, let a = engine.agent(agentID) { draft = a; lastSaved = a; loaded = true }
            memoryText = engine.memory.read(agentID); memoryLoaded = memoryText
        }
        .onChange(of: draft) { _, new in scheduleSave(new) }
        .onChange(of: memoryText) { _, new in scheduleMemorySave(new) }
        .onDisappear { flushMemory(); commitRename() }
        .onChange(of: engine.agent(agentID)) { _, external in
            // File changed on disk by something other than us: take it.
            if let external, external != draft, external != lastSaved { draft = external; lastSaved = external }
        }
    }

    private func scheduleSave(_ a: Agent) {
        guard loaded, a != lastSaved else { return }
        saveTask?.cancel()
        saveTask = Task {
            try? await Task.sleep(nanoseconds: 350_000_000)
            if Task.isCancelled { return }
            lastSaved = a
            engine.save(a)
        }
    }

    private func scheduleMemorySave(_ text: String) {
        guard text != memoryLoaded else { return }
        memoryTask?.cancel()
        memoryTask = Task {
            try? await Task.sleep(nanoseconds: 600_000_000)
            if Task.isCancelled { return }
            writeMemory(text)
        }
    }
    private func flushMemory() { memoryTask?.cancel(); if memoryText != memoryLoaded { writeMemory(memoryText) } }
    private func writeMemory(_ text: String) {
        guard text.count <= MemoryStore.maxChars else { return }
        try? engine.memory.write(agentID, text)
        memoryLoaded = text
        engine.noteChange("Edited memory of \(agentID)")
    }

    /// Flush the pending edit, then let the file name and chat folder follow the display name.
    private func commitRename() {
        guard loaded else { return }
        saveTask?.cancel()
        if draft != lastSaved { lastSaved = draft; engine.save(draft) }
        engine.renameToMatchName(agentID)
    }

    // MARK: form

    private var form: some View {
        Form {
            Section {
                HStack(spacing: 14) {
                    AvatarView(agent: draft, size: 54)
                    VStack(spacing: 8) {
                        TextField("Name", text: $draft.name).accessibilityIdentifier("agent-name")
                            .onSubmit { commitRename() }
                        TextField("Short description", text: $draft.summary)
                    }
                }
                Picker("Kind", selection: $draft.kind) {
                    Text("Assistant").tag(Agent.Kind.assistant)
                    Text("Character").tag(Agent.Kind.character)
                }
                TextField("Sidebar group", text: $draft.group, prompt: Text(draft.defaultGroup))
                    .accessibilityIdentifier("agent-group")
                HStack {
                    Text("Avatar color")
                    Spacer()
                    ForEach(0..<Palette.colors.count, id: \.self) { i in
                        Circle().fill(Palette.color(i)).frame(width: 20, height: 20)
                            .overlay(Circle().stroke(Color.primary, lineWidth: draft.color == i ? 2 : 0).padding(-3))
                            .onTapGesture { draft.color = i }
                    }
                }
            }
            Section("Persona") {
                TextEditor(text: $draft.persona)
                    .font(.system(.body)).frame(minHeight: 150)
                    .scrollContentBackground(.hidden)
                    .padding(6).background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
                    .accessibilityIdentifier("agent-persona")
                Text("This becomes the system prompt. Write it the way you would brief a person.").font(.caption).foregroundStyle(.secondary)
            }
            Section("Models") {
                Picker("Language model", selection: $draft.model) {
                    ForEach(languageChoices, id: \.self) { Text(label(for: $0)).tag($0) }
                }
                Picker("Image model", selection: Binding(get: { draft.imageModel ?? "" }, set: { draft.imageModel = $0.isEmpty ? nil : $0 })) {
                    Text("None").tag("")
                    ForEach(imageChoices, id: \.self) { Text(label(for: $0)).tag($0) }
                }
                if let remote = remoteProvider(for: draft.model) {
                    Label("\(remote) is a remote provider: this agent's messages leave your Mac.", systemImage: "network").foregroundStyle(.orange).font(.caption)
                } else if let problem = engine.modelProblem(for: draft) {
                    Label(problem, systemImage: "exclamationmark.triangle").foregroundStyle(.orange).font(.caption)
                }
            }
            Section("Harness") {
                TextField("Command (leave empty for the built-in loop)", text: Binding(get: { draft.harness ?? "" }, set: { draft.harness = $0.isEmpty ? nil : $0 }))
                    .textFieldStyle(.roundedBorder).font(.system(.body, design: .monospaced)).accessibilityIdentifier("agent-harness")
                Text("Runs this agent's turns in your own program instead of the built-in loop. It talks to the models through parlor's OpenAI-compatible endpoint and its tool calls still go through the tools and permission tier below. See harnesses/CLAUDE.md in your parlor folder.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Tools this agent may use") { toolsSection }
            Section("Control of the app") {
                Picker("Permission tier", selection: Binding(get: { draft.tier }, set: { draft.tier = $0 })) {
                    Text("None").tag(Tier.none)
                    Text("Read").tag(Tier.read)
                    Text("Act").tag(Tier.act)
                    Text("Configure").tag(Tier.configure)
                }
                .pickerStyle(.segmented).accessibilityIdentifier("agent-tier")
                Text("Lets this agent use the app's own parlor.* tools, if its tools above include them: Read looks at agents, chats and settings; Act talks in rooms and runs services; Configure edits agents and settings. Deleting anything always asks you first.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("What it remembers") {
                TextEditor(text: $memoryText)
                    .font(.system(.body)).frame(minHeight: 90)
                    .scrollContentBackground(.hidden)
                    .padding(6).background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
                    .accessibilityIdentifier("agent-memory")
                HStack {
                    Text("memory/\(agentID).md · \(memoryText.count) of \(MemoryStore.maxChars)").font(.caption.monospacedDigit()).foregroundStyle(memoryText.count > MemoryStore.maxChars ? .red : .secondary)
                    Spacer()
                }
                Text("Notes this agent keeps across chats. They go into its system prompt every turn. The agent can add to them itself with memory.append; you can edit them here.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Limits for this agent") { overridesSection }
            Section("Memory and behavior") {
                LabeledContent("Context budget") {
                    HStack {
                        Slider(value: Binding(get: { Double(draft.contextTokens) }, set: { draft.contextTokens = Int($0 / 1024) * 1024 }), in: 2048...65536, step: 1024)
                        Text(tokenString(draft.contextTokens)).monospacedDigit().frame(width: 44, alignment: .trailing)
                    }
                }
                LabeledContent("Compact above") {
                    HStack {
                        Slider(value: $draft.compactAt, in: 0.3...0.95, step: 0.05)
                        Text("\(Int(draft.compactAt * 100))%").monospacedDigit().frame(width: 44, alignment: .trailing)
                    }
                }
                LabeledContent("Temperature") {
                    HStack {
                        Slider(value: $draft.temperature, in: 0...1.5, step: 0.05)
                        Text(String(format: "%.2f", draft.temperature)).monospacedDigit().frame(width: 44, alignment: .trailing)
                    }
                }
                Toggle("Let the model think before answering", isOn: $draft.thinking)
                    .help("Slower, sometimes better at multi-step tool use.")
            }
        }
        .formStyle(.grouped)
    }

    private var languageChoices: [String] {
        var set = engine.library.installed(kind: .language).map(\.repo) + engine.providerModelChoices
        if !set.contains(draft.model) { set.append(draft.model) }
        return set
    }
    private func remoteProvider(for model: String) -> String? {
        guard let (p, _) = engine.router.route(model), p.config.locality == .remote else { return nil }
        return p.config.id
    }
    private var imageChoices: [String] {
        var set = engine.library.installed(kind: .image).map(\.repo)
        if let m = draft.imageModel, !m.isEmpty, !set.contains(m) { set.append(m) }
        return set
    }
    private func label(for repo: String) -> String {
        let name = shortModelName(repo)
        return engine.library.installedDirectory(for: repo) == nil ? "\(name) (not downloaded)" : name
    }

    // MARK: limits (settings scopes)

    private var modelMeta: ModelMetadata? { engine.library.installedDirectory(for: draft.model).flatMap { ModelMetadata.load(dir: $0) } }

    @ViewBuilder private var overridesSection: some View {
        let m = engine.settings
        let model = modelMeta
        OverrideRow(title: "Top-p", value: $draft.overrides.topP, format: "%.2f",
                    inherited: model?.topP ?? m.llm.topP, source: model?.topP != nil ? "model" : "machine", id: "top_p")
        OverrideRow(title: "Repetition penalty", value: $draft.overrides.repetitionPenalty, format: "%.2f",
                    inherited: m.llm.repetitionPenalty ?? 1.0, source: "machine", id: "repetition_penalty")
        OverrideIntRow(title: "Reply limit (tokens, 0 = none)", value: $draft.overrides.maxReplyTokens,
                       inherited: m.llm.maxReplyTokens ?? 0, source: "machine", id: "max_reply_tokens")
        OverrideIntRow(title: "Max tool steps per turn", value: $draft.overrides.maxSteps, inherited: m.loop.maxSteps, source: "machine", id: "max_steps")
        OverrideIntRow(title: "Tool result limit (characters)", value: $draft.overrides.toolResultChars, inherited: m.loop.toolResultChars, source: "machine", id: "tool_result_chars")
        OverrideIntRow(title: "Tool timeout (seconds)", value: $draft.overrides.toolTimeoutSeconds, inherited: m.loop.toolTimeoutSeconds, source: "machine", id: "tool_timeout")
        OverrideIntRow(title: "Keep verbatim (messages)", value: $draft.overrides.keepRecent, inherited: m.loop.keepRecent, source: "machine", id: "keep_recent")
        OverrideIntRow(title: "Summary length (tokens)", value: $draft.overrides.summaryTokens, inherited: m.loop.summaryTokens, source: "machine", id: "summary_tokens")
        if let cap = model?.maxContext, cap > 0, draft.contextTokens > cap {
            Label("Context is clamped to this model's maximum of \(tokenString(cap)).", systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
        }
        Text("Each limit follows the model, else the machine (System → settings.json), until you override it here. Overridden values apply to this agent only. Memory knobs such as the attention window stay machine-wide.")
            .font(.caption).foregroundStyle(.secondary)
    }

    // MARK: tools

    @ViewBuilder private var toolsSection: some View {
        let groups = Dictionary(grouping: engine.tools, by: \.server).sorted { $0.key < $1.key }
        if groups.isEmpty {
            Text("No tools mounted yet. Add servers in System.").foregroundStyle(.secondary)
        } else {
            Toggle("All tools", isOn: Binding(
                get: { draft.tools.contains("*") },
                set: { draft.tools = $0 ? ["*"] : [] }))
            ForEach(groups, id: \.key) { server, list in
                DisclosureGroup {
                    ForEach(list) { t in
                        Toggle(isOn: Binding(get: { draft.allows(t.qualified) }, set: { setTool(t, $0) })) {
                            VStack(alignment: .leading, spacing: 0) {
                                Text(t.name).font(.system(.body, design: .monospaced))
                                Text(t.description).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                            }
                        }
                    }
                } label: {
                    Toggle(isOn: Binding(
                        get: { list.allSatisfy { draft.allows($0.qualified) } },
                        set: { on in for t in list { setTool(t, on, collapse: false) }; normalize() })) {
                        Text(server).fontWeight(.medium)
                    }
                }
            }
        }
    }

    private func setTool(_ t: ToolDef, _ on: Bool, collapse: Bool = true) {
        var explicit = Set<String>()
        if draft.tools.contains("*") { explicit = Set(engine.tools.map(\.qualified)) }
        else {
            for tool in engine.tools where draft.allows(tool.qualified) { explicit.insert(tool.qualified) }
            // keep patterns for tools not mounted right now
            for p in draft.tools where !p.contains("*") { explicit.insert(p) }
        }
        if on { explicit.insert(t.qualified) } else { explicit.remove(t.qualified) }
        draft.tools = explicit.sorted()
        if collapse { normalize() }
    }

    /// Collapse fully selected servers into `server.*`, everything into `*`.
    private func normalize() {
        let all = engine.tools
        let on = Set(all.filter { draft.allows($0.qualified) }.map(\.qualified))
        if !all.isEmpty, on.count == all.count { draft.tools = ["*"]; return }
        var out: [String] = []
        for (server, list) in Dictionary(grouping: all, by: \.server) {
            let mine = list.filter { on.contains($0.qualified) }
            if mine.count == list.count && !list.isEmpty { out.append("\(server).*") }
            else { out.append(contentsOf: mine.map(\.qualified)) }
        }
        draft.tools = out.sorted()
    }

    // MARK: file pane

    private var filePane: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("agents/\(agentID).md").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            ScrollView {
                Text(draft.fileText()).font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
            HStack {
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([engine.home.agentsDir.appendingPathComponent("\(agentID).md")])
                }
                Button("Duplicate") { engine.duplicate(agentID) }
            }
            Text("This file is the agent. Editing it elsewhere, including from Claude Code, updates this page live.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(14)
    }
}


/// One overridable number: shows what it inherits and from where, with Override / Inherit.
private struct OverrideRow: View {
    let title: String
    @Binding var value: Double?
    let format: String
    let inherited: Double
    let source: String
    let id: String

    var body: some View {
        LabeledContent(title) {
            HStack {
                if let v = value {
                    TextField("", value: Binding(get: { v }, set: { value = $0 }), format: .number).frame(width: 80).multilineTextAlignment(.trailing)
                    Text("agent").font(.caption).foregroundStyle(.tint)
                    Button("Inherit") { value = nil }.controlSize(.small).accessibilityIdentifier("inherit-\(id)")
                } else {
                    Text(String(format: format, inherited)).monospacedDigit()
                    Text(source).font(.caption).foregroundStyle(.secondary)
                    Button("Override") { value = inherited }.controlSize(.small).accessibilityIdentifier("override-\(id)")
                }
            }
        }
    }
}

private struct OverrideIntRow: View {
    let title: String
    @Binding var value: Int?
    let inherited: Int
    let source: String
    let id: String

    var body: some View {
        LabeledContent(title) {
            HStack {
                if let v = value {
                    TextField("", value: Binding(get: { v }, set: { value = max(0, $0) }), format: .number).frame(width: 80).multilineTextAlignment(.trailing)
                    Text("agent").font(.caption).foregroundStyle(.tint)
                    Button("Inherit") { value = nil }.controlSize(.small).accessibilityIdentifier("inherit-\(id)")
                } else {
                    Text("\(inherited)").monospacedDigit()
                    Text(source).font(.caption).foregroundStyle(.secondary)
                    Button("Override") { value = inherited }.controlSize(.small).accessibilityIdentifier("override-\(id)")
                }
            }
        }
    }
}

import SwiftUI
import ParlorCore

/// Edits one agent. Changes save to `agents/<id>/` and external edits flow back in.
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
            Section("Harness") {
                Picker("Harness", selection: Binding(get: { draft.hasInlineHarness ? "__custom" : draft.harnessID }, set: { v in
                    if v == "__custom" { draft.harness = draft.hasInlineHarness ? draft.harness : "python3 ~/parlor/harnesses/your_harness.py" }
                    else { draft.harness = v == "chat" ? nil : v }
                })) {
                    ForEach(engine.harnessList, id: \.id) { Text($0.title).tag($0.id) }
                    Text("Custom command…").tag("__custom")
                }
                .accessibilityIdentifier("agent-harness-picker")
                if draft.hasInlineHarness {
                    TextField("Command", text: Binding(get: { draft.harness ?? "" }, set: { draft.harness = $0.isEmpty ? nil : $0 }))
                        .textFieldStyle(.roundedBorder).font(.system(.body, design: .monospaced)).accessibilityIdentifier("agent-harness")
                }
                Text(manifest?.description ?? "Runs this agent's turns in your own program. Its tool calls still go through the tools and permission tier below. See harnesses/CLAUDE.md in your parlor folder.")
                    .font(.caption).foregroundStyle(.secondary)
                if let problem = engine.harnessProblem(for: draft) {
                    Label(problem, systemImage: "exclamationmark.triangle").foregroundStyle(.orange).font(.caption)
                }
            }
            Section("Hosts") {
                Picker("Chat host", selection: $draft.model) {
                    ForEach(chatChoices, id: \.self) { Text(hostLabel($0)).tag($0) }
                }
                .accessibilityIdentifier("agent-chat-host")
                if showsImage {
                    Picker("Image host", selection: Binding(get: { draft.imageModel ?? "" }, set: { draft.imageModel = $0.isEmpty ? nil : $0 })) {
                        Text("None").tag("")
                        ForEach(imageChoices, id: \.self) { Text(hostLabel($0)).tag($0) }
                    }
                    .accessibilityIdentifier("agent-image-host")
                }
                if let remote = remoteProvider(for: draft.model) {
                    Label("\(remote) is a remote provider: this agent's messages leave your Mac.", systemImage: "network").foregroundStyle(.orange).font(.caption)
                } else if let problem = engine.modelProblem(for: draft) {
                    Label(problem, systemImage: "exclamationmark.triangle").foregroundStyle(.orange).font(.caption)
                }
            }
            if let m = manifest, m.config != nil {
                Section("\(m.title) settings") {
                    SchemaForm(schema: m.config, values: $draft.config)
                    ForEach(m.actions.filter { $0.placement == "settings" && m.available($0, hosts: draft.hosts) }) { act in
                        Button(act.title) { engine.room(for: agentID).perform(action: act, agent: draft) }
                            .help(act.hint ?? "").accessibilityIdentifier("action-\(act.id)")
                    }
                }
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
                    Text("agents/\(agentID)/memory.md · \(memoryText.count) of \(MemoryStore.maxChars)").font(.caption.monospacedDigit()).foregroundStyle(memoryText.count > MemoryStore.maxChars ? .red : .secondary)
                    Spacer()
                }
                Text("Notes this agent keeps across chats. They go into its system prompt every turn. The agent can add to them itself with memory.append; you can edit them here.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    private var manifest: HarnessManifest? { draft.hasInlineHarness ? nil : engine.harnessList.first { $0.id == draft.harnessID } }
    private var showsImage: Bool { manifest.map { $0.needs["image"] != nil } ?? true }

    private var chatChoices: [String] {
        var set = engine.catalog.hosts(capability: "chat").map(\.id)
        if !draft.model.isEmpty, !set.contains(draft.model) { set.append(draft.model) }
        return set
    }
    private var imageChoices: [String] {
        var set = engine.catalog.hosts(capability: "image").map(\.id)
        if let m = draft.imageModel, !m.isEmpty, !set.contains(m) { set.append(m) }
        return set
    }
    private func remoteProvider(for model: String) -> String? {
        guard let (p, _) = engine.router.route(model), p.config.locality == .remote else { return nil }
        return p.config.id
    }
    private func hostLabel(_ id: String) -> String {
        let cat = engine.catalog
        guard let h = cat.host(id) else { return id.isEmpty ? "None" : "\(shortModelName(id)) (no such host)" }
        let st = cat.status(host: h.id)
        return st.usable ? h.id : "\(h.id) (\(st.label))"
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

    private var recordText: String {
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return (try? String(data: enc.encode(AgentRecord(draft)), encoding: .utf8)) ?? ""
    }

    private var filePane: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("agents/\(agentID)/agent.json").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            ScrollView {
                Text(recordText).font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
            HStack {
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([engine.home.agentsDir.appendingPathComponent(agentID)])
                }
                Button("Duplicate") { engine.duplicate(agentID) }
            }
            Text("This file is the agent. Editing it elsewhere, including from Claude Code, updates this page live.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(14)
    }
}



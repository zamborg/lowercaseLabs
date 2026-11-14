import SwiftUI
import ParlorCore

struct ContentView: View {
    @Environment(ParlorEngine.self) private var engine
    @Binding var route: Route?

    var body: some View {
        NavigationSplitView {
            SidebarView(route: $route)
                .navigationSplitViewColumnWidth(min: 220, ideal: 250, max: 320)
        } detail: {
            switch route {
            case .agent(let id):
                if engine.agent(id) != nil { AgentPage(agentID: id, route: $route).id(id) }
                else { EmptyDetail(route: $route) }
            case .room(let id):
                if engine.conversations.contains(where: { $0.id == id }) { RoomView(roomID: id).id(id) }
                else { EmptyDetail(route: $route) }
            case .models: ModelsView()
            case .tools: ToolsView()
            case .system: SystemView()
            case nil: EmptyDetail(route: $route)
            }
        }
    }
}

struct EmptyDetail: View {
    @Environment(ParlorEngine.self) private var engine
    @Binding var route: Route?
    var body: some View {
        ContentUnavailableView {
            Label("No agent selected", systemImage: "person.crop.circle.badge.plus")
        } description: {
            Text("Create an agent to start a chat.")
        } actions: {
            Button("New Assistant") { let a = engine.create(kind: .assistant); route = .agent(a.id) }
            Button("New Character") { let a = engine.create(kind: .character); route = .agent(a.id) }
        }
    }
}

struct SidebarView: View {
    @Environment(ParlorEngine.self) private var engine
    @Binding var route: Route?
    @State private var memory: (used: Int64, total: Int64) = (0, SystemMemory.totalBytes)
    @State private var showNewRoom = false
    private let timer = Timer.publish(every: 3, on: .main, in: .common).autoconnect()

    var body: some View {
        List(selection: $route) {
            Section("Assistants") {
                ForEach(engine.agents.filter { $0.kind == .assistant }) { row($0) }
            }
            Section("Characters") {
                ForEach(engine.agents.filter { $0.kind == .character }) { row($0) }
            }
            if !engine.conversations.isEmpty {
                Section("Rooms") {
                    ForEach(engine.conversations) { c in
                        HStack(spacing: 9) {
                            Image(systemName: "person.2.fill").frame(width: 26).foregroundStyle(.secondary)
                            VStack(alignment: .leading, spacing: 0) {
                                Text(c.title).fontWeight(.medium)
                                Text(c.config.agentIDs.compactMap { engine.agent($0)?.name }.joined(separator: ", "))
                                    .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                            }
                            Spacer(minLength: 0)
                            if engine.room(for: c.id).isBusy { ProgressView().controlSize(.small) }
                        }
                        .tag(Route.room(c.id))
                        .contextMenu {
                            Button("Reveal Folder in Finder") { NSWorkspace.shared.activateFileViewerSelecting([engine.home.chatDir(for: c.id)]) }
                            Divider()
                            Button("Delete Room", role: .destructive) { engine.deleteRoom(c.id); if route == .room(c.id) { route = nil } }
                        }
                    }
                }
            }
            Section("Library") {
                Label("Models", systemImage: "cpu").tag(Route.models)
                Label {
                    HStack { Text("Tools"); Spacer(); Text("\(engine.tools.count)").foregroundStyle(.secondary).monospacedDigit() }
                } icon: { Image(systemName: "wrench.and.screwdriver") }.tag(Route.tools)
                Label("System", systemImage: "gauge.with.dots.needle.67percent").tag(Route.system)
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .bottom) { memoryGauge }
        .toolbar {
            ToolbarItem {
                Menu {
                    Button("New Assistant") { let a = engine.create(kind: .assistant); route = .agent(a.id) }
                    Button("New Character") { let a = engine.create(kind: .character); route = .agent(a.id) }
                    Divider()
                    Button("New Room…") { showNewRoom = true }.disabled(engine.agents.count < 2)
                } label: { Label("New Agent", systemImage: "plus") }
                .help("New agent")
            }
        }
        .sheet(isPresented: $showNewRoom) { NewRoomSheet(route: $route) }
        .onReceive(timer) { _ in
            memory = (SystemMemory.usedBytes(), SystemMemory.totalBytes)
            Task { await engine.refreshLoaded() }
        }
        .onAppear { memory = (SystemMemory.usedBytes(), SystemMemory.totalBytes) }
    }

    private func row(_ a: Agent) -> some View {
        HStack(spacing: 9) {
            AvatarView(agent: a, size: 26)
            VStack(alignment: .leading, spacing: 0) {
                Text(a.name).fontWeight(.medium)
                if !a.summary.isEmpty { Text(a.summary).font(.caption).foregroundStyle(.secondary).lineLimit(1) }
            }
            Spacer(minLength: 0)
            if engine.room(for: a.id).isBusy { ProgressView().controlSize(.small) }
        }
        .tag(Route.agent(a.id))
        .contextMenu {
            Button("Duplicate") { engine.duplicate(a.id) }
            Button("Reveal File in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([engine.home.agentsDir.appendingPathComponent("\(a.id).md")])
            }
            Divider()
            Button("Delete Agent and Chat", role: .destructive) { engine.delete(a.id) }
        }
    }

    private var memoryGauge: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Memory").font(.caption.weight(.medium))
                Spacer()
                Text("\(memoryString(memory.used)) of \(memoryString(memory.total))").font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            ProgressView(value: Double(memory.used), total: Double(max(memory.total, 1))).controlSize(.small)
            HStack(spacing: 6) {
                Image(systemName: "circle.fill").font(.system(size: 6)).foregroundStyle(engine.loadedModel == nil ? Color.secondary : Color.green)
                Text(engine.loadedModel.map { "LLM · \(shortModelName($0))" } ?? "No model loaded").lineLimit(1)
            }.font(.caption2).foregroundStyle(.secondary)
            HStack(spacing: 6) {
                Image(systemName: "circle.fill").font(.system(size: 6)).foregroundStyle(engine.imageWorkerState == "idle" ? Color.secondary : Color.orange)
                Text("Image · \(engine.imageWorkerState)").lineLimit(1)
            }.font(.caption2).foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .padding(8)
    }
}

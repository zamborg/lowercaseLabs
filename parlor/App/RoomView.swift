import SwiftUI
import ParlorCore

/// A conversation with several agents: one log, one scheduler, every line attributed.
struct RoomView: View {
    @Environment(ParlorEngine.self) private var engine
    let roomID: String
    @State private var draft = ""
    @State private var confirmReset = false
    @FocusState private var composerFocused: Bool

    private var room: Room { engine.room(for: roomID) }
    private var info: ConversationInfo? { engine.conversations.first { $0.id == roomID } }
    private var agents: [Agent] { room.participantIDs.compactMap { engine.agent($0) } }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            messageList
            composer
        }
        .navigationTitle(info?.title ?? "Room")
        .toolbar {
            ToolbarItemGroup {
                Menu {
                    Button("Compact Now") { if let a = agents.first { room.compactNow(agent: a) } }
                        .disabled(room.isBusy || room.messages.count < 8)
                    Button("Reveal Log in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([engine.home.chatDir(for: roomID).appendingPathComponent("log.jsonl")])
                    }
                    Divider()
                    Button("Clear Room…", role: .destructive) { confirmReset = true }
                } label: { Image(systemName: "ellipsis.circle") }
            }
        }
        .confirmationDialog("Clear the whole room?", isPresented: $confirmReset) {
            Button("Clear Room", role: .destructive) { room.reset() }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 10) {
                HStack(spacing: -8) {
                    ForEach(agents) { AvatarView(agent: $0, size: 30).overlay(Circle().stroke(.background, lineWidth: 2)) }
                }
                VStack(alignment: .leading, spacing: 0) {
                    Text(info?.title ?? "Room").font(.headline)
                    Text(agents.map(\.name).joined(separator: ", ") + " · " + (info?.config.scheduler.label ?? ""))
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
            }
            if Set(agents.map(\.model)).count > 1 {
                Label("These agents use different models, so a model may reload between turns.", systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
        }
        .padding(.horizontal, 20).padding(.vertical, 12)
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if room.messages.isEmpty && !room.isBusy {
                        Text(emptyText).foregroundStyle(.secondary).frame(maxWidth: .infinity).padding(.top, 60)
                    }
                    ForEach(room.messages) { m in
                        MessageRow(message: m, agent: speaker(of: m), room: room, showName: true, selecting: false, selected: false) {}
                    }
                    if room.isBusy { liveRow.id("live") }
                    if let e = room.errorText {
                        Label(e, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red).font(.callout)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(20)
            }
            .onChange(of: room.messages.count) { withAnimation { proxy.scrollTo("bottom") } }
            .onChange(of: room.streaming) { proxy.scrollTo("bottom") }
        }
    }

    private var emptyText: String {
        switch info?.config.scheduler.kind {
        case "autonomous": return "Press Run to let them talk."
        case "roundRobin": return "Say something, and each agent answers in turn. Or press Run."
        case "mention": return "Name an agent to hand them the floor."
        default: return "Say something to the room."
        }
    }

    private func speaker(of m: ChatMessage) -> Agent {
        let id = m.speaker ?? room.participantIDs.first ?? ""
        return engine.agent(id) ?? agents.first ?? Agent(id: id, name: id)
    }

    private var liveRow: some View {
        let a = room.speakingID.flatMap { engine.agent($0) } ?? agents.first
        return HStack(alignment: .top, spacing: 8) {
            if let a { AvatarView(agent: a, size: 26) }
            VStack(alignment: .leading, spacing: 2) {
                if let a { Text(a.name).font(.caption.weight(.semibold)).foregroundStyle(Palette.color(a.color)).padding(.leading, 4) }
                if room.streaming.isEmpty {
                    HStack(spacing: 8) { TypingDots(); if let t = room.activity { Text(t).font(.caption).foregroundStyle(.secondary) } }
                        .padding(.horizontal, 12).padding(.vertical, 9)
                        .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
                } else {
                    Text(markdown(room.streaming)).textSelection(.enabled)
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
                }
            }
            Spacer(minLength: 80)
        }
        .accessibilityIdentifier("live-row")
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Message the room…", text: $draft, axis: .vertical)
                .textFieldStyle(.plain).lineLimit(1...8).focused($composerFocused).onSubmit(send)
                .padding(.horizontal, 12).padding(.vertical, 9)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(.separator))
                .accessibilityIdentifier("room-composer")
            if room.isBusy {
                Button { room.stop() } label: { Image(systemName: "stop.fill") }
                    .buttonStyle(.borderedProminent).tint(.red).controlSize(.large).help("Stop")
            } else {
                if info?.config.scheduler.kind != "userDriven" {
                    Button { room.run() } label: { Label("Run", systemImage: "play.fill") }
                        .controlSize(.large).help("Let the agents continue without a message")
                        .accessibilityIdentifier("room-run")
                }
                Button(action: send) { Image(systemName: "arrow.up") }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                    .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityIdentifier("room-send")
            }
        }
        .padding(.horizontal, 20).padding(.vertical, 12)
        .onAppear { composerFocused = true }
    }

    private func send() {
        let text = draft
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !room.isBusy else { return }
        draft = ""
        engine.send(text, to: roomID)
    }
}

struct NewRoomSheet: View {
    @Environment(ParlorEngine.self) private var engine
    @Environment(\.dismiss) private var dismiss
    @Binding var route: Route?
    @State private var title = ""
    @State private var chosen: Set<String> = []
    @State private var kind = "roundRobin"
    @State private var turns = 6

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("New Room").font(.title2.weight(.semibold))
            TextField("Name (optional)", text: $title).textFieldStyle(.roundedBorder)
            Text("Who is in it").font(.headline)
            VStack(alignment: .leading, spacing: 6) {
                ForEach(engine.agents) { a in
                    Toggle(isOn: Binding(get: { chosen.contains(a.id) }, set: { if $0 { chosen.insert(a.id) } else { chosen.remove(a.id) } })) {
                        HStack(spacing: 8) { AvatarView(agent: a, size: 22); Text(a.name); Text(shortModelName(a.model)).font(.caption).foregroundStyle(.secondary) }
                    }
                }
            }
            Text("Who speaks next").font(.headline)
            Picker("", selection: $kind) {
                Text("Round robin").tag("roundRobin")
                Text("By mention").tag("mention")
                Text("Autonomous").tag("autonomous")
                Text("You drive").tag("userDriven")
            }.labelsHidden().pickerStyle(.segmented)
            Text(help).font(.caption).foregroundStyle(.secondary)
            if kind != "userDriven" { Stepper("Up to \(turns) turns per run", value: $turns, in: 1...30) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Create") {
                    var spec = SchedulerSpec(kind: kind)
                    if kind == "autonomous" { spec.stepBudget = turns } else if kind != "userDriven" { spec.maxTurns = turns }
                    let ordered = engine.agents.map(\.id).filter { chosen.contains($0) }
                    let names = ordered.compactMap { engine.agent($0)?.name }
                    let id = engine.createRoom(title: title.isEmpty ? names.joined(separator: " & ") : title, agents: ordered, scheduler: spec)
                    route = .room(id)
                    dismiss()
                }
                .buttonStyle(.borderedProminent).disabled(chosen.count < 2)
                .accessibilityIdentifier("create-room")
            }
        }
        .padding(22).frame(width: 440)
    }

    private var help: String {
        switch kind {
        case "roundRobin": return "Each agent answers in turn after you speak, or when you press Run."
        case "mention": return "The agent you name answers; if they name someone, that agent answers next."
        case "autonomous": return "Press Run and the agents talk to each other for up to the turn limit."
        default: return "Only the first agent replies to each message."
        }
    }
}

import SwiftUI
import ParlorCore

struct MessageRow: View {
    @Environment(ParlorEngine.self) private var engine
    let message: ChatMessage
    let agent: Agent
    let room: Room
    var showName = false
    let selecting: Bool
    let selected: Bool
    let toggle: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            if selecting && message.role != .marker {
                Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(selected ? Color.accentColor : .secondary)
                    .font(.title3).padding(.top, 4)
            }
            content
        }
        .contentShape(Rectangle())
        .onTapGesture { if selecting { toggle() } }
        .background(selected ? Color.accentColor.opacity(0.08) : .clear, in: RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder private var content: some View {
        switch message.role {
        case .user:
            HStack { Spacer(minLength: 80)
                Text(markdown(message.text)).selectable(!selecting)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 14))
                    .foregroundStyle(.white)
            }
        case .assistant:
            if message.text.isEmpty { EmptyView() }
            else {
                HStack(alignment: .top, spacing: 8) {
                    AvatarView(agent: agent, size: 26)
                    VStack(alignment: .leading, spacing: 2) {
                        if showName { Text(agent.name).font(.caption.weight(.semibold)).foregroundStyle(Palette.color(agent.color)).padding(.leading, 4) }
                        Text(markdown(message.text)).selectable(!selecting)
                            .padding(.horizontal, 12).padding(.vertical, 8)
                            .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
                    }
                    Spacer(minLength: 80)
                }
            }
        case .tool:
            HStack(alignment: .top, spacing: 8) {
                Color.clear.frame(width: 26, height: 1)
                ToolResultView(message: message, call: callRecord, home: engine.home)
                Spacer(minLength: 80)
            }
        case .marker:
            HStack(spacing: 10) {
                VStack { Divider() }
                Text(message.text).font(.caption).foregroundStyle(.secondary)
                VStack { Divider() }
            }.padding(.vertical, 2)
        }
    }

    private var callRecord: ToolCallRecord? {
        guard let id = message.toolCallID else { return nil }
        for m in room.messages { if let c = m.toolCalls?.first(where: { $0.id == id }) { return c } }
        return nil
    }
}

struct ToolResultView: View {
    let message: ChatMessage
    let call: ToolCallRecord?
    let home: ParlorHome
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button { expanded.toggle() } label: {
                HStack(spacing: 6) {
                    Image(systemName: message.isError == true ? "xmark.circle.fill" : "checkmark.circle.fill")
                        .foregroundStyle(message.isError == true ? Color.red : Color.green)
                    Text((call?.qualified ?? message.toolName?.replacingOccurrences(of: "__", with: ".")) ?? "tool")
                        .font(.system(.caption, design: .monospaced).weight(.semibold))
                    Text(argsSummary).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).lineLimit(1)
                    if let s = message.seconds { Text(String(format: "%.1fs", s)).font(.caption).foregroundStyle(.secondary).monospacedDigit() }
                    Image(systemName: expanded ? "chevron.up" : "chevron.down").font(.caption2).foregroundStyle(.secondary)
                }
                .padding(.horizontal, 10).padding(.vertical, 5)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("tool-chip")
            if expanded {
                VStack(alignment: .leading, spacing: 4) {
                    if let a = call?.arguments.text { Text("args  \(a)") }
                    Text(message.text.isEmpty ? "(no output)" : String(message.text.prefix(1500)))
                }
                .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
                .padding(8).frame(maxWidth: 460, alignment: .leading)
                .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
            }
            ForEach(message.images ?? [], id: \.self) { rel in
                ImageCard(path: rel, home: home, prompt: call?.arguments.objectValue?["prompt"]?.stringValue)
            }
        }
    }

    private var argsSummary: String {
        guard let o = call?.arguments.objectValue else { return "" }
        return o.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value.stringValue ?? $0.value.text)" }.joined(separator: " ")
    }
}

struct ImageCard: View {
    let path: String
    let home: ParlorHome
    let prompt: String?
    @State private var showPrompt = false

    var body: some View {
        let url = home.root.appendingPathComponent(path)
        VStack(alignment: .leading, spacing: 0) {
            if let img = NSImage(contentsOf: url) {
                Image(nsImage: img).resizable().scaledToFit().frame(maxWidth: 380)
                    .accessibilityLabel("Generated image")
                    .accessibilityIdentifier("generated-image")
            } else {
                Label("Image missing", systemImage: "photo").padding(30)
            }
            HStack {
                Text(url.lastPathComponent).font(.caption2).foregroundStyle(.secondary)
                Spacer()
                if prompt != nil { Button("Prompt") { showPrompt.toggle() }.buttonStyle(.link).font(.caption2) }
                Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([url]) }.buttonStyle(.link).font(.caption2)
            }
            .padding(.horizontal, 10).padding(.vertical, 5).background(.quaternary.opacity(0.4))
            if showPrompt, let p = prompt {
                Text(p).font(.caption).textSelection(.enabled).padding(10).frame(maxWidth: 380, alignment: .leading)
            }
        }
        .frame(maxWidth: 380)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(.separator))
        .contextMenu {
            Button("Copy Image") {
                if let img = NSImage(contentsOf: url) { NSPasteboard.general.clearContents(); NSPasteboard.general.writeObjects([img]) }
            }
            Button("Save As…") {
                let p = NSSavePanel(); p.nameFieldStringValue = url.lastPathComponent
                if p.runModal() == .OK, let dest = p.url { try? FileManager.default.copyItem(at: url, to: dest) }
            }
        }
    }
}

extension View {
    /// Text selection swallows clicks, so it is off while the user is picking messages to prune.
    @ViewBuilder func selectable(_ on: Bool) -> some View {
        if on { textSelection(.enabled) } else { textSelection(.disabled) }
    }
}

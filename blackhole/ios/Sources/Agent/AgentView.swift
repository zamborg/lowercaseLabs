import SwiftUI

struct AgentView: View {
    @State private var messages: [AgentChatMessage] = []
    @State private var draft = ""
    @State private var isSending = false
    @State private var alertMessage: String?
    @FocusState private var isComposerFocused: Bool

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        if messages.isEmpty {
                            AgentEmptyState(onSend: sendPreset)
                                .padding(.top, 24)
                        }

                        ForEach(messages) { message in
                            AgentMessageRow(message: message)
                                .id(message.id)
                        }

                        if isSending {
                            AgentTypingRow()
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                    .padding(.bottom, 160)
                }
                .scrollDismissesKeyboard(.interactively)
                .simultaneousGesture(TapGesture().onEnded {
                    isComposerFocused = false
                })
                .onChange(of: messages.count) { _, _ in
                    guard let last = messages.last else { return }
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
            .navigationTitle("Agent")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(Color.black, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { isComposerFocused = false }
                }
            }
        }
        .background(Color.black.ignoresSafeArea())
        .safeAreaInset(edge: .bottom, spacing: 0) {
            AgentComposer(text: $draft, isSending: isSending, isFocused: $isComposerFocused) {
                Task { await sendDraft() }
            }
        }
        .preferredColorScheme(.dark)
        .alert("Agent Error", isPresented: Binding(
            get: { alertMessage != nil },
            set: { if !$0 { alertMessage = nil } }
        )) {
            Button("OK") { alertMessage = nil }
        } message: {
            Text(alertMessage ?? "")
        }
    }

    private func sendPreset(_ text: String) {
        draft = text
        Task { await sendDraft() }
    }

    private func sendDraft() async {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isSending else { return }

        draft = ""
        isSending = true
        isComposerFocused = false
        messages.append(AgentChatMessage(role: .user, text: text))

        do {
            let response = try await APIClient.shared.askAgent(message: text)
            messages.append(AgentChatMessage(response: response))
        } catch {
            alertMessage = error.localizedDescription
        }

        isSending = false
    }
}

private struct AgentChatMessage: Identifiable, Hashable {
    enum Role: Hashable {
        case user
        case assistant
    }

    let id = UUID()
    let role: Role
    let text: String
    let toolCalls: [AgentResponse.ToolCall]
    let createdItems: [Item]
    let updatedItems: [Item]

    init(role: Role, text: String) {
        self.role = role
        self.text = text
        toolCalls = []
        createdItems = []
        updatedItems = []
    }

    init(response: AgentResponse) {
        role = .assistant
        text = response.response
        toolCalls = response.toolCalls
        createdItems = response.itemsCreated
        updatedItems = response.itemsUpdated
    }
}

private struct AgentEmptyState: View {
    let onSend: (String) -> Void

    private let presets = [
        "Give me a daily brief",
        "Find stale todos",
        "What should I focus on today?"
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Agent", systemImage: "sparkles")
                .font(.system(.title3, design: .rounded, weight: .semibold))
                .foregroundStyle(.white)

            VStack(alignment: .leading, spacing: 8) {
                ForEach(presets, id: \.self) { preset in
                    Button {
                        onSend(preset)
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: "arrow.up.right")
                                .font(.system(size: 12, weight: .semibold))
                            Text(preset)
                                .font(.system(.subheadline, design: .rounded, weight: .medium))
                            Spacer()
                        }
                        .foregroundStyle(.white.opacity(0.82))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(Color.white.opacity(0.07))
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct AgentMessageRow: View {
    let message: AgentChatMessage

    var body: some View {
        HStack(alignment: .top) {
            if message.role == .user { Spacer(minLength: 36) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 8) {
                Text(message.text)
                    .font(.system(.body, design: .rounded))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .background(message.role == .user ? Color.white.opacity(0.16) : Color.white.opacity(0.07))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .textSelection(.enabled)

                if message.role == .assistant {
                    AgentToolCallStrip(toolCalls: message.toolCalls)
                    AgentChangedItemsSection(title: "Created", items: message.createdItems)
                    AgentChangedItemsSection(title: "Updated", items: message.updatedItems)
                }
            }
            .frame(maxWidth: 320, alignment: message.role == .user ? .trailing : .leading)

            if message.role == .assistant { Spacer(minLength: 36) }
        }
    }
}

private struct AgentToolCallStrip: View {
    let toolCalls: [AgentResponse.ToolCall]

    var body: some View {
        if !toolCalls.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(Array(toolCalls.enumerated()), id: \.offset) { _, call in
                        Label(call.name, systemImage: call.status == "success" ? "checkmark.circle" : "exclamationmark.circle")
                            .font(.system(.caption2, design: .rounded, weight: .semibold))
                            .foregroundStyle(call.status == "success" ? .green.opacity(0.8) : .red.opacity(0.8))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 5)
                            .background(Color.white.opacity(0.06))
                            .clipShape(Capsule())
                    }
                }
            }
        }
    }
}

private struct AgentChangedItemsSection: View {
    let title: String
    let items: [Item]

    var body: some View {
        if !items.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text(title.uppercased())
                    .font(.system(.caption2, design: .rounded, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.38))

                ForEach(items) { item in
                    HStack(spacing: 8) {
                        Image(systemName: item.type.systemImage)
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.yellow.opacity(0.8))
                            .frame(width: 18)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(item.title)
                                .font(.system(.caption, design: .rounded, weight: .semibold))
                                .foregroundStyle(.white.opacity(0.86))
                                .lineLimit(1)
                            Text(item.type.displayName)
                                .font(.system(.caption2, design: .rounded))
                                .foregroundStyle(.white.opacity(0.42))
                        }
                        Spacer()
                    }
                    .padding(10)
                    .background(Color.white.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
            }
        }
    }
}

private struct AgentTypingRow: View {
    var body: some View {
        HStack {
            HStack(spacing: 6) {
                ProgressView()
                    .scaleEffect(0.72)
                    .tint(.white.opacity(0.65))
                Text("Thinking")
                    .font(.system(.caption, design: .rounded, weight: .medium))
                    .foregroundStyle(.white.opacity(0.55))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Color.white.opacity(0.07))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            Spacer()
        }
    }
}

private struct AgentComposer: View {
    @Binding var text: String
    let isSending: Bool
    @FocusState.Binding var isFocused: Bool
    let onSend: () -> Void

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField("Message", text: $text, axis: .vertical)
                .font(.system(.body, design: .rounded))
                .foregroundStyle(.white)
                .lineLimit(1...5)
                .textFieldStyle(.plain)
                .tint(.white)
                .focused($isFocused)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(Color.white.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .onSubmit(onSend)

            Button(action: onSend) {
                Image(systemName: "paperplane.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.black)
                    .frame(width: 38, height: 38)
                    .background(.white)
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .disabled(isSending || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .opacity(isSending || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.35 : 1)
        }
        .padding(12)
        .background(.black.opacity(0.96))
    }
}

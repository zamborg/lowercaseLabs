import SwiftUI

struct IngestView: View {
    @StateObject private var viewModel = IngestViewModel()

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 14) {
                        enginePicker
                        transcriptCard
                        if viewModel.canSubmit { submitButton }
                    }
                    .padding(16)
                    .padding(.bottom, 100)
                }
            }
            .navigationTitle("blackhole")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.black, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .safeAreaInset(edge: .bottom) { holdBar }
        }
        .preferredColorScheme(.dark)
        .alert("Error", isPresented: Binding(
            get: { viewModel.alertMessage != nil },
            set: { if !$0 { viewModel.alertMessage = nil } }
        )) {
            Button("OK") { viewModel.alertMessage = nil }
        } message: {
            Text(viewModel.alertMessage ?? "")
        }
        .overlay(alignment: .top) {
            if let item = viewModel.lastPostedItem {
                savedToast(item: item)
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .padding(.top, 8)
                    .onAppear {
                        Task {
                            try? await Task.sleep(nanoseconds: 2_500_000_000)
                            withAnimation { viewModel.lastPostedItem = nil }
                        }
                    }
            }
        }
        .animation(.spring(response: 0.3), value: viewModel.lastPostedItem?.id)
        .animation(.easeInOut(duration: 0.2), value: viewModel.canSubmit)
    }

    private var enginePicker: some View {
        Picker("Engine", selection: Binding(
            get: { viewModel.selectedEngineKind },
            set: { viewModel.selectEngine($0) }
        )) {
            ForEach(DictationEngineKind.allCases) { kind in
                Text(kind.title).tag(kind)
            }
        }
        .pickerStyle(.segmented)
        .disabled(viewModel.areControlsLocked)
    }

    private var transcriptCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label(viewModel.status.title, systemImage: viewModel.status.symbolName)
                    .font(.system(.subheadline, design: .rounded, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.6))
                Spacer()
                if viewModel.isBusy {
                    ProgressView().controlSize(.small).tint(.white)
                }
                if !viewModel.transcriptText.isEmpty && viewModel.status == .idle {
                    Button("Clear") { viewModel.clearTranscript() }
                        .font(.system(.caption, design: .rounded, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.4))
                }
            }

            if viewModel.status == .idle && !viewModel.transcriptText.isEmpty {
                TextEditor(text: $viewModel.transcriptText)
                    .font(.system(.body, design: .serif))
                    .foregroundStyle(.white)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 160)
                    .tint(.white)
            } else {
                Text(viewModel.transcriptText.isEmpty
                    ? viewModel.status.detail
                    : viewModel.transcriptText)
                    .font(.system(.body, design: .serif))
                    .foregroundStyle(viewModel.transcriptText.isEmpty ? .white.opacity(0.35) : .white)
                    .frame(maxWidth: .infinity, minHeight: 160, alignment: .topLeading)
                    .multilineTextAlignment(.leading)
            }
        }
        .padding(18)
        .background(Color.white.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(.white.opacity(0.1), lineWidth: 1))
    }

    private var submitButton: some View {
        Button { viewModel.submitTranscript() } label: {
            HStack(spacing: 8) {
                Image(systemName: "arrow.up.circle.fill")
                Text("Send to Blackhole")
                    .font(.system(.headline, design: .rounded, weight: .bold))
            }
            .foregroundStyle(.black)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(.white)
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
        .disabled(viewModel.isBusy)
    }

    private var holdBar: some View {
        VStack(spacing: 8) {
            Text(viewModel.status.holdInstruction)
                .font(.system(.caption, design: .rounded, weight: .medium))
                .foregroundStyle(.white.opacity(0.45))
            HoldToDictateButton(
                isActive: viewModel.isListening,
                isDisabled: viewModel.isBusy,
                onPressStart: { viewModel.pressDidStart() },
                onPressEnd: { viewModel.pressDidEnd() }
            )
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.black.opacity(0.95))
    }

    private func savedToast(item: Item) -> some View {
        HStack(spacing: 8) {
            Image(systemName: item.type == .todo ? "checkmark.circle.fill" : "note.text")
                .foregroundStyle(item.type == .todo ? .yellow : .white)
            Text(item.title)
                .font(.system(.subheadline, design: .rounded, weight: .semibold))
                .foregroundStyle(.white)
                .lineLimit(1)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color(white: 0.15))
        .clipShape(Capsule())
        .overlay(Capsule().stroke(.white.opacity(0.15), lineWidth: 1))
    }
}

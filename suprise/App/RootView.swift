import SwiftUI

struct RootView: View {
    @StateObject private var viewModel = DictationViewModel()
    @State private var selectedRecord: DictationRecord?

    var body: some View {
        NavigationStack {
            ZStack {
                LinearGradient(
                    colors: [
                        Color(red: 0.05, green: 0.06, blue: 0.08),
                        Color(red: 0.08, green: 0.09, blue: 0.12),
                        Color(red: 0.04, green: 0.05, blue: 0.06),
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        headerCard
                        transcriptCard
                        auditSection
                    }
                    .padding(.horizontal, 18)
                    .padding(.top, 18)
                    .padding(.bottom, 148)
                }
            }
            .navigationTitle("suprise")
            .safeAreaInset(edge: .bottom) {
                holdBar
            }
        }
        .preferredColorScheme(.dark)
        .task {
            viewModel.reloadSavedDictations()
        }
        .alert("Dictation", isPresented: Binding(
            get: { viewModel.alertMessage != nil },
            set: { if !$0 { viewModel.alertMessage = nil } }
        )) {
            Button("OK", role: .cancel) {
                viewModel.alertMessage = nil
            }
        } message: {
            Text(viewModel.alertMessage ?? "")
        }
        .sheet(item: $selectedRecord) { record in
            DictationRecordDetailView(record: record)
        }
    }

    private var headerCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Live Dictation")
                .font(.system(.title2, design: .rounded, weight: .bold))
                .foregroundStyle(.white)

            Text(viewModel.statusDetail)
                .font(.system(.subheadline, design: .rounded, weight: .medium))
                .foregroundStyle(Color.white.opacity(0.68))

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

            Text(viewModel.selectedEngineKind.caption)
                .font(.system(.footnote, design: .rounded, weight: .medium))
                .foregroundStyle(Color.white.opacity(0.55))
        }
        .padding(18)
        .background(Color.white.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .stroke(Color.white.opacity(0.12), lineWidth: 1)
        )
    }

    private var transcriptCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label(viewModel.statusTitle, systemImage: viewModel.statusSymbolName)
                    .font(.system(.headline, design: .rounded, weight: .semibold))
                    .foregroundStyle(.white)

                Spacer()

                if viewModel.isBusy {
                    ProgressView()
                        .controlSize(.small)
                        .tint(.white)
                }
            }

            Text(viewModel.transcriptText.isEmpty ? "Press and hold to dictate. Your text will appear here while you speak." : viewModel.transcriptText)
                .font(.system(.title3, design: .serif, weight: .regular))
                .foregroundStyle(viewModel.transcriptText.isEmpty ? Color.white.opacity(0.42) : .white)
                .frame(maxWidth: .infinity, minHeight: 210, alignment: .topLeading)
                .multilineTextAlignment(.leading)
        }
        .padding(20)
        .background(
            LinearGradient(
                colors: [
                    Color(red: 0.11, green: 0.12, blue: 0.15),
                    Color(red: 0.08, green: 0.09, blue: 0.11),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .stroke(Color.white.opacity(0.1), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.3), radius: 24, x: 0, y: 14)
    }

    private var auditSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Saved Dictations")
                .font(.system(.headline, design: .rounded, weight: .semibold))
                .foregroundStyle(.white)

            if viewModel.savedRecords.isEmpty {
                Text("Nothing saved yet.")
                    .font(.system(.subheadline, design: .rounded, weight: .medium))
                    .foregroundStyle(Color.white.opacity(0.58))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .background(Color.white.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            } else {
                ForEach(viewModel.savedRecords) { record in
                    Button {
                        selectedRecord = record
                    } label: {
                        DictationRecordRow(record: record)
                    }
                    .buttonStyle(.plain)
                    .contextMenu {
                        Button(role: .destructive) {
                            viewModel.delete(record)
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                }
            }
        }
    }

    private var holdBar: some View {
        VStack(spacing: 10) {
            Text(viewModel.holdInstruction)
                .font(.system(.footnote, design: .rounded, weight: .semibold))
                .foregroundStyle(Color.white.opacity(0.66))

            HoldToDictateButton(
                isActive: viewModel.isListening,
                isDisabled: viewModel.isBusy,
                onPressStart: {
                    viewModel.pressDidStart()
                },
                onPressEnd: {
                    viewModel.pressDidEnd()
                }
            )
        }
        .padding(.horizontal, 18)
        .padding(.top, 12)
        .padding(.bottom, 10)
        .background(Color.black.opacity(0.36))
    }
}

private struct DictationRecordRow: View {
    let record: DictationRecord

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(record.timestampLabel)
                    .font(.system(.subheadline, design: .rounded, weight: .semibold))
                    .foregroundStyle(.white)

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(Color.white.opacity(0.35))
            }

            Text(record.previewText)
                .font(.system(.body, design: .serif, weight: .regular))
                .foregroundStyle(Color.white.opacity(0.66))
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .background(Color.white.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(Color.white.opacity(0.1), lineWidth: 1)
        )
    }
}

private struct DictationRecordDetailView: View {
    let record: DictationRecord
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                Text(record.text)
                    .font(.system(.title3, design: .serif, weight: .regular))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(20)
            }
            .background(
                LinearGradient(
                    colors: [
                        Color(red: 0.05, green: 0.06, blue: 0.08),
                        Color(red: 0.08, green: 0.09, blue: 0.12),
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .navigationTitle(record.timestampLabel)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

#Preview {
    RootView()
}

import SwiftUI

struct CaptureView: View {
    @Bindable var model: CaptureViewModel

    var body: some View {
        NavigationStack {
            ZStack {
                AppTheme.background
                    .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 20) {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Capture")
                                .font(.system(size: 34, weight: .black, design: .rounded))
                            Text(model.statusMessage)
                                .font(.headline)
                                .foregroundStyle(AppTheme.ink.opacity(0.7))
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)

                        modePicker

                        VStack(spacing: 14) {
                            Text(timeLabel(model.elapsedSeconds))
                                .font(.system(size: 44, weight: .bold, design: .rounded))
                                .foregroundStyle(AppTheme.ink)

                            Button {
                                Task { await model.toggleRecording() }
                            } label: {
                                ZStack {
                                    Circle()
                                        .fill(
                                            RadialGradient(
                                                colors: [AppTheme.accent, AppTheme.accentStrong],
                                                center: .center,
                                                startRadius: 4,
                                                endRadius: 140
                                            )
                                        )
                                        .frame(width: 190, height: 190)
                                        .shadow(color: AppTheme.accentStrong.opacity(0.35), radius: 24, x: 0, y: 12)

                                    Circle()
                                        .stroke(Color.white.opacity(0.45), lineWidth: 2)
                                        .frame(width: 166, height: 166)

                                    Image(systemName: model.isRecording ? "stop.fill" : "mic.fill")
                                        .font(.system(size: 50, weight: .bold))
                                        .foregroundStyle(.white)
                                }
                            }
                            .buttonStyle(.plain)

                            Text(model.routeSummary)
                                .font(.headline)
                                .foregroundStyle(AppTheme.ink.opacity(0.8))
                        }
                        .frostedPanel()

                        if !model.transcriptPreview.isEmpty {
                            VStack(alignment: .leading, spacing: 12) {
                                Label("Transcript Preview", systemImage: "text.quote")
                                    .font(.headline)
                                Text(model.transcriptPreview)
                                    .foregroundStyle(AppTheme.ink.opacity(0.85))
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .frostedPanel()
                        }

                        if let errorMessage = model.errorMessage {
                            Text(errorMessage)
                                .font(.footnote)
                                .foregroundStyle(.red)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .frostedPanel()
                        }
                    }
                    .padding(20)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .sheet(
                isPresented: Binding(
                    get: { model.pendingConfirmation != nil },
                    set: { if !$0 { model.pendingConfirmation = nil } }
                )
            ) {
                confirmationSheet
                    .presentationDetents([.fraction(0.34)])
            }
            .onAppear {
                model.syncPendingMode()
            }
        }
    }

    private var modePicker: some View {
        HStack(spacing: 10) {
            ForEach(CaptureMode.allCases, id: \.self) { mode in
                Button {
                    model.selectedMode = mode
                } label: {
                    Text(mode.rawValue.capitalized)
                        .font(.subheadline.weight(.semibold))
                        .padding(.vertical, 10)
                        .padding(.horizontal, 14)
                        .foregroundStyle(model.selectedMode == mode ? .white : AppTheme.ink)
                        .background(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(model.selectedMode == mode ? AppTheme.accentStrong : AppTheme.mist)
                        )
                }
                .buttonStyle(.plain)
            }
        }
        .frostedPanel()
    }

    private var confirmationSheet: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Confirm Destination")
                .font(.title3.bold())

            Text("The route confidence was low. Pick the right destination for this capture.")
                .foregroundStyle(.secondary)

            Button("Create Note") {
                Task { await model.confirmCreateNote() }
            }
            .buttonStyle(.borderedProminent)

            Button("Create Todo") {
                Task { await model.confirmCreateTodo() }
            }
            .buttonStyle(.borderedProminent)

            Button("Search Notes") {
                model.confirmSearch()
            }
            .buttonStyle(.bordered)
        }
        .padding(24)
    }

    private func timeLabel(_ seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds))
        return String(format: "%02d:%02d", total / 60, total % 60)
    }
}


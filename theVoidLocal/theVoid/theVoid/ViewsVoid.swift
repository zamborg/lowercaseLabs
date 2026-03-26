import SwiftUI
import UIKit

// MARK: - Void experience

struct VoidExperienceView: View {
    @EnvironmentObject private var model: AppModel
    @StateObject private var recorder = RecorderEngine()
    @State private var pendingSubmission: PendingSubmission?
    @State private var showDecisionSheet = false
    @State private var showWelcomeOverlay = false
    @State private var showModelDownloadPrompt = false
    @State private var hasPresentedModelDownloadPrompt = false
    @State private var autoWelcomePendingAck = false
    @State private var isRecordingLocked = false

    private struct PendingSubmission {
        let url: URL
        let durationSeconds: Int
    }

    var body: some View {
        NavigationStack {
            GeometryReader { geometry in
                let availableHeight = geometry.size.height
                let isCompactHeight = availableHeight < 760
                let isVeryCompactHeight = availableHeight < 700
                let stackSpacing: CGFloat = isVeryCompactHeight ? 14 : (isCompactHeight ? 18 : 26)
                let signalSize: CGFloat = isVeryCompactHeight ? 156 : (isCompactHeight ? 172 : 190)
                let touchAreaMinHeight: CGFloat = isVeryCompactHeight ? 180 : (isCompactHeight ? 200 : 240)
                let topPadding: CGFloat = isVeryCompactHeight ? 16 : (isCompactHeight ? 20 : 26)
                let bottomPadding = max(6, geometry.safeAreaInsets.bottom + 4)

                ZStack {
                    LinearGradient(
                        colors: [Color.black, Color(red: 0.05, green: 0.06, blue: 0.1)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                    .ignoresSafeArea()

                    VStack(spacing: stackSpacing) {
                        Text(model.submissionState.title)
                            .font(.headline)
                            .foregroundStyle(.white.opacity(0.78))
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                            .frame(maxWidth: .infinity)
                            .multilineTextAlignment(.center)

                        if !model.liquidModelPrepared {
                            VStack(spacing: 6) {
                                Label("Insights model not downloaded", systemImage: "exclamationmark.triangle.fill")
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(Color.orange.opacity(0.95))
                                    .lineLimit(1)
                                    .minimumScaleFactor(0.85)
                                Text("On-device insights are off until the AI model is downloaded. Use the download prompt or the Settings tile > On-Device AI.")
                                    .font(.footnote)
                                    .foregroundStyle(.white.opacity(0.74))
                                    .multilineTextAlignment(.center)
                                    .lineLimit(isVeryCompactHeight ? 3 : 4)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .layoutPriority(2)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.horizontal, 12)
                        }

                        Text(formatDuration(recorder.elapsed))
                            .font(.system(size: isVeryCompactHeight ? 40 : 46, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.white)

                        PulsingSignalDot(
                            amplitude: recorder.amplitude,
                            isRecording: recorder.isRecording,
                            isProcessing: model.submissionState == .transcribing
                        )
                        .frame(width: signalSize, height: signalSize)

                        Text(recordingInstructionTitle)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.white.opacity(0.9))
                            .multilineTextAlignment(.center)

                        Text(recordingInstructionDetail)
                            .font(.footnote)
                            .foregroundStyle(.white.opacity(0.68))
                            .lineLimit(2)
                            .multilineTextAlignment(.center)

                        if model.submissionState == .transcribing {
                            Text("Processing transcription and tags on this device…")
                                .font(.footnote)
                                .foregroundStyle(.white.opacity(0.75))
                                .multilineTextAlignment(.center)
                                .lineLimit(2)
                                .fixedSize(horizontal: false, vertical: true)
                        }

                        GeometryReader { padGeometry in
                            let dynamicTouchAreaHeight = max(touchAreaMinHeight, padGeometry.size.height)
                            touchAndHoldPad
                                .frame(height: dynamicTouchAreaHeight)
                                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
                        }
                        .frame(maxHeight: .infinity, alignment: .bottom)
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, topPadding)
                    .padding(.bottom, bottomPadding)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .blur(radius: showWelcomeOverlay ? 6 : 0)
                    .allowsHitTesting(!showWelcomeOverlay)

                    if showWelcomeOverlay {
                        LinearGradient(
                            colors: [
                                Color.black.opacity(0.72),
                                Color.black.opacity(0.58),
                            ],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                            .ignoresSafeArea()

                        WelcomeOverlayCard {
                            dismissWelcomeOverlay()
                        }
                        .padding(.horizontal, 22)
                        .transition(.opacity.combined(with: .scale(scale: 0.96)))
                    }
                }
            }
            .navigationTitle("The Void")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        autoWelcomePendingAck = false
                        showWelcomeOverlay = true
                    } label: {
                        Image(systemName: "questionmark.circle")
                    }
                }
            }
            .sheet(isPresented: $showDecisionSheet, onDismiss: { pendingSubmission = nil }) {
                RecordingDecisionSheet(
                    onShare: { submitPending(shareToSocial: true) },
                    onLocalOnly: { submitPending(shareToSocial: false) }
                )
                .presentationDetents([.fraction(0.32)])
                .presentationDragIndicator(.visible)
            }
            .alert("Download On-Device AI Model?", isPresented: $showModelDownloadPrompt) {
                Button("Not Now", role: .cancel) {}
                Button("Download Model") {
                    model.prepareLiquidModelIfNeeded()
                }
            } message: {
                Text("Download Liquid on this device to enable mood tags and dots. This is a one-time setup.")
            }
            .onAppear {
                recorder.onWarning = { _ in
                    UINotificationFeedbackGenerator().notificationOccurred(.warning)
                }
                recorder.onAutoStop = { url, duration in
                    pendingSubmission = PendingSubmission(url: url, durationSeconds: duration)
                    model.submissionState = .uploading
                    showDecisionSheet = true
                    isRecordingLocked = false
                }
                model.reloadDrafts()
                maybePresentWelcomeOverlayIfNeeded()
                maybePresentModelDownloadPromptIfNeeded()
            }
            .onChange(of: model.userID) { _, _ in
                hasPresentedModelDownloadPrompt = false
            }
            .onChange(of: model.liquidModelPrepared) { _, prepared in
                if prepared {
                    showModelDownloadPrompt = false
                } else {
                    maybePresentModelDownloadPromptIfNeeded()
                }
            }
            .animation(.easeInOut(duration: 0.2), value: showWelcomeOverlay)
        }
    }

    private func startRecording() {
        Task { @MainActor in
            if recorder.recordPermissionStatus() == .denied {
                model.errorMessage = "Microphone permission is denied. Enable it in iOS Settings > Privacy & Security > Microphone."
                return
            }

            let granted = await recorder.requestPermission()
            guard granted else {
                model.errorMessage = "Microphone access is required"
                return
            }
            do {
                let draftURL = model.makeDraftURL()
                try recorder.startRecording(at: draftURL)
                model.submissionState = .recording
            } catch {
                model.errorMessage = "Failed to start recording: \(error.localizedDescription)\nIf using a Simulator, ensure an audio input is available."
            }
        }
    }

    private func finalizeRecordingForChoice() {
        guard let recordedURL = recorder.stopRecording() else {
            return
        }
        isRecordingLocked = false
        let duration = max(1, Int(recorder.elapsed))
        pendingSubmission = PendingSubmission(url: recordedURL, durationSeconds: duration)
        model.submissionState = .uploading
        showDecisionSheet = true
    }

    private func submitPending(shareToSocial: Bool) {
        guard let pendingSubmission else {
            return
        }
        let payload = pendingSubmission
        self.pendingSubmission = nil
        showDecisionSheet = false
        Task {
            await model.submitDraft(
                url: payload.url,
                durationSeconds: payload.durationSeconds,
                shareToSocial: shareToSocial
            )
        }
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds))
        let mins = total / 60
        let secs = total % 60
        return String(format: "%02d:%02d", mins, secs)
    }

    private var welcomeSeenKey: String {
        let scopedID = model.userID.isEmpty ? "anonymous" : model.userID
        return "thevoid.welcome.overlay.seen.\(scopedID)"
    }

    private func maybePresentWelcomeOverlayIfNeeded() {
        guard !model.userID.isEmpty else {
            return
        }
        if UserDefaults.standard.bool(forKey: welcomeSeenKey) {
            return
        }
        autoWelcomePendingAck = true
        showWelcomeOverlay = true
    }

    private func dismissWelcomeOverlay() {
        if autoWelcomePendingAck {
            UserDefaults.standard.set(true, forKey: welcomeSeenKey)
            autoWelcomePendingAck = false
        }
        showWelcomeOverlay = false
        maybePresentModelDownloadPromptIfNeeded()
    }

    private func maybePresentModelDownloadPromptIfNeeded() {
        guard !showWelcomeOverlay else {
            return
        }
        guard !model.userID.isEmpty else {
            return
        }
        guard !model.liquidModelPrepared else {
            return
        }
        guard !model.isPreparingLiquidModel else {
            return
        }
        guard !hasPresentedModelDownloadPrompt else {
            return
        }
        hasPresentedModelDownloadPrompt = true
        showModelDownloadPrompt = true
    }

    private var recordingInstructionTitle: String {
        if recorder.isRecording {
            return isRecordingLocked
                ? "Recording locked. Touch the pad to stop"
                : "Release on the pad to stop. Release outside to lock"
        }
        return "Press and hold to record"
    }

    private var recordingInstructionDetail: String {
        if recorder.isRecording {
            return isRecordingLocked
                ? "Recording continues until you touch the pad again."
                : "Slide outside and lift to keep recording hands-free."
        }
        return "Max 5:00."
    }

    private var touchAndHoldPad: some View {
        RoundedRectangle(cornerRadius: 22)
            .fill(
                LinearGradient(
                    colors: [
                        Color.white.opacity(recorder.isRecording ? 0.22 : 0.11),
                        Color.white.opacity(recorder.isRecording ? 0.14 : 0.06),
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .frame(maxWidth: .infinity)
            .overlay {
                VStack(spacing: 10) {
                    Image(systemName: recorder.isRecording ? (isRecordingLocked ? "lock.fill" : "waveform.circle.fill") : "hand.tap.fill")
                        .font(.system(size: 38, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.88))
                    Text(
                        recorder.isRecording
                            ? (isRecordingLocked ? "Recording Locked" : "Recording…")
                            : "Touch and hold here"
                    )
                        .font(.headline)
                        .foregroundStyle(.white.opacity(0.92))
                        .lineLimit(1)
                        .minimumScaleFactor(0.85)
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: 22)
                    .stroke(Color.white.opacity(recorder.isRecording ? 0.5 : 0.2), lineWidth: 1.2)
            )
            .overlay {
                PressAndHoldCaptureView(
                    onPressStart: {
                        guard !showDecisionSheet else { return }
                        if recorder.isRecording {
                            if isRecordingLocked {
                                finalizeRecordingForChoice()
                            }
                            return
                        }
                        isRecordingLocked = false
                        startRecording()
                    },
                    onPressEnd: { endedInside in
                        guard recorder.isRecording else { return }
                        guard !isRecordingLocked else { return }
                        if endedInside {
                            finalizeRecordingForChoice()
                        } else {
                            isRecordingLocked = true
                        }
                    }
                )
            }
            .contentShape(RoundedRectangle(cornerRadius: 22))
    }

}

private struct PressAndHoldCaptureView: UIViewRepresentable {
    let onPressStart: () -> Void
    let onPressEnd: (Bool) -> Void

    func makeUIView(context: Context) -> PressAndHoldControl {
        let control = PressAndHoldControl()
        control.onPressStart = onPressStart
        control.onPressEnd = onPressEnd
        return control
    }

    func updateUIView(_ uiView: PressAndHoldControl, context _: Context) {
        uiView.onPressStart = onPressStart
        uiView.onPressEnd = onPressEnd
    }
}

private final class PressAndHoldControl: UIControl {
    var onPressStart: (() -> Void)?
    var onPressEnd: ((Bool) -> Void)?
    private var isPressing = false

    override init(frame: CGRect) {
        super.init(frame: frame)
        isExclusiveTouch = true
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func beginTracking(_ touch: UITouch, with event: UIEvent?) -> Bool {
        _ = touch
        _ = event
        guard !isPressing else { return true }
        isPressing = true
        onPressStart?()
        return true
    }

    override func continueTracking(_ touch: UITouch, with event: UIEvent?) -> Bool {
        _ = touch
        _ = event
        return true
    }

    override func endTracking(_ touch: UITouch?, with event: UIEvent?) {
        _ = event
        let endedInside: Bool
        if let touch {
            endedInside = bounds.contains(touch.location(in: self))
        } else {
            endedInside = false
        }
        finishPressIfNeeded(endedInside: endedInside)
    }

    override func cancelTracking(with event: UIEvent?) {
        _ = event
        finishPressIfNeeded(endedInside: false)
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesCancelled(touches, with: event)
        finishPressIfNeeded(endedInside: false)
    }

    private func finishPressIfNeeded(endedInside: Bool) {
        guard isPressing else { return }
        isPressing = false
        onPressEnd?(endedInside)
    }
}

struct PulsingSignalDot: View {
    let amplitude: CGFloat
    let isRecording: Bool
    let isProcessing: Bool

    var body: some View {
        let normalized = max(0.08, min(1.0, amplitude))
        let pulseDriver: CGFloat = isRecording ? normalized : (isProcessing ? 0.58 : 0.16)
        let pulseScale = 0.86 + (pulseDriver * 0.54)
        let haloColor = isProcessing ? Color.orange : Color.teal
        let coreTop = isProcessing ? Color(red: 1.0, green: 0.78, blue: 0.32) : Color.cyan
        let coreBottom = isProcessing ? Color(red: 0.98, green: 0.56, blue: 0.18) : Color.teal
        let isActive = isRecording || isProcessing

        ZStack {
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            haloColor.opacity(isActive ? 0.24 : 0.10),
                            Color.clear,
                        ],
                        center: .center,
                        startRadius: 12,
                        endRadius: 102
                    )
                )
                .scaleEffect(isActive ? pulseScale * 1.36 : 1.0)

            Circle()
                .fill(
                    LinearGradient(
                        colors: [
                            coreTop.opacity(isActive ? 0.92 : 0.72),
                            coreBottom.opacity(isActive ? 0.74 : 0.52),
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: 92, height: 92)
                .scaleEffect(isActive ? pulseScale : 0.88)
                .shadow(color: haloColor.opacity(0.45), radius: isActive ? 14 : 7, x: 0, y: 1)
                .overlay(
                    Circle()
                        .stroke(Color.white.opacity(0.28), lineWidth: 1)
                )
        }
        .animation(.easeInOut(duration: 0.12), value: normalized)
        .animation(.easeInOut(duration: 0.18), value: isRecording)
        .animation(.easeInOut(duration: 0.22), value: isProcessing)
    }
}

struct RecordingDecisionSheet: View {
    let onShare: () -> Void
    let onLocalOnly: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            Text("Save This Reflection")
                .font(.headline)
            HStack(spacing: 14) {
                decisionButton(
                    icon: "person.2.circle.fill",
                    title: "Share Dot",
                    subtitle: "Post to friends",
                    tint: .teal,
                    action: onShare
                )
                decisionButton(
                    icon: "iphone.gen3.radiowaves.left.and.right",
                    title: "Local Only",
                    subtitle: "Keep it private",
                    tint: .indigo,
                    action: onLocalOnly
                )
            }
            Text("Shared entries publish the color wheel dot to all friends.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(20)
    }

    private func decisionButton(
        icon: String,
        title: String,
        subtitle: String,
        tint: Color,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundStyle(tint)
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(Color.secondary.opacity(0.10), in: RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }
}

struct WelcomeOverlayCard: View {
    let onDismiss: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("WELCOME")
                        .font(.caption.weight(.semibold))
                        .tracking(1.4)
                        .foregroundStyle(Color.white.opacity(0.58))
                    Text("theVoid")
                        .font(.system(size: 40, weight: .heavy, design: .rounded))
                        .foregroundStyle(.white)
                }
                Spacer()
                Button(action: onDismiss) {
                    Image(systemName: "xmark")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(.white.opacity(0.82))
                        .frame(width: 30, height: 30)
                        .background(Color.white.opacity(0.08), in: Circle())
                        .overlay(
                            Circle()
                                .stroke(Color.white.opacity(0.18), lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)
            }

            Text("theVoid is a place for you to share your thoughts freely. Your thoughts become dots, a reflection of your mood in that moment.")
                .font(.body.weight(.medium))
                .foregroundStyle(Color.white.opacity(0.93))

            Text("Everything is transcribed and analyzed on your device. You choose if you want to share your *dots* with friends.")
                .font(.body)
                .foregroundStyle(Color.white.opacity(0.86))

            HStack(spacing: 8) {
                WelcomeBadge(icon: "lock.shield.fill", label: "On-device")
                WelcomeBadge(icon: "circle.grid.2x2.fill", label: "Dots, not transcripts")
                WelcomeBadge(icon: "sparkles", label: "Check in anytime")
            }

            Text("Check in as much as you'd like, and have a fun time in theVoid.")
                .font(.callout)
                .foregroundStyle(Color.white.opacity(0.86))

            Button {
                onDismiss()
            } label: {
                HStack {
                    Spacer()
                    Text("Enter theVoid")
                        .font(.headline.weight(.semibold))
                    Spacer()
                }
                .padding(.vertical, 12)
                .background(
                    LinearGradient(
                        colors: [
                            Color.cyan.opacity(0.92),
                            Color.teal.opacity(0.82),
                        ],
                        startPoint: .leading,
                        endPoint: .trailing
                    ),
                    in: RoundedRectangle(cornerRadius: 12)
                )
                .foregroundStyle(.black.opacity(0.78))
            }
            .buttonStyle(.plain)
        }
        .padding(22)
        .background(
            ZStack {
                RoundedRectangle(cornerRadius: 24)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(red: 0.05, green: 0.08, blue: 0.14),
                                Color(red: 0.08, green: 0.12, blue: 0.19),
                                Color(red: 0.04, green: 0.09, blue: 0.13),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )

                Circle()
                    .fill(Color.teal.opacity(0.26))
                    .frame(width: 180, height: 180)
                    .offset(x: 78, y: -76)
                    .blur(radius: 32)

                Circle()
                    .fill(Color.cyan.opacity(0.18))
                    .frame(width: 130, height: 130)
                    .offset(x: -84, y: 82)
                    .blur(radius: 26)
            }
        )
        .overlay(
            RoundedRectangle(cornerRadius: 24)
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.5), radius: 24, x: 0, y: 14)
    }
}

private struct WelcomeBadge: View {
    let icon: String
    let label: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.caption.weight(.bold))
            Text(label)
                .font(.caption.weight(.semibold))
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .foregroundStyle(Color.white.opacity(0.86))
        .background(Color.white.opacity(0.10), in: Capsule())
        .overlay(
            Capsule()
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        )
    }
}

struct AudioReactiveBars: View {
    let amplitude: CGFloat

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let count = 36
            let spacing: CGFloat = 4
            let barWidth = (width - (CGFloat(count - 1) * spacing)) / CGFloat(count)

            HStack(spacing: spacing) {
                ForEach(0..<count, id: \.self) { index in
                    let phase = CGFloat(index) / CGFloat(count)
                    let wave = sin((phase + amplitude) * .pi * 4)
                    let strength = max(0.12, amplitude + wave * 0.28)
                    Capsule()
                        .fill(Color.teal.opacity(0.85))
                        .frame(width: barWidth, height: max(18, strength * 170))
                }
            }
            .frame(maxHeight: .infinity, alignment: .center)
        }
    }
}

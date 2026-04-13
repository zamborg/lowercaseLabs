import SwiftUI
import UIKit

struct HoldToDictateButton: View {
    let isActive: Bool
    let isDisabled: Bool
    let onPressStart: () -> Void
    let onPressEnd: () -> Void

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: isDisabled
                            ? [Color.gray.opacity(0.3), Color.gray.opacity(0.2)]
                            : isActive
                                ? [Color(white: 0.18), Color(white: 0.12)]
                                : [Color(white: 0.14), Color(white: 0.1)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )

            HStack(spacing: 10) {
                Image(systemName: isActive ? "waveform.circle.fill" : "mic.fill")
                    .font(.system(size: 20, weight: .semibold))
                Text(isActive ? "Recording…" : (isDisabled ? "Preparing…" : "Hold to Dictate"))
                    .font(.system(.headline, design: .rounded, weight: .bold))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 24)

            PressAndHoldCaptureView(
                isDisabled: isDisabled,
                onPressStart: onPressStart,
                onPressEnd: onPressEnd
            )
        }
        .frame(maxWidth: .infinity)
        .frame(height: 64)
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .stroke(Color.white.opacity(isActive ? 0.3 : 0.12), lineWidth: 1)
        )
    }
}

// Internal so SearchView can reuse
struct PressAndHoldCaptureView: UIViewRepresentable {
    let isDisabled: Bool
    let onPressStart: () -> Void
    let onPressEnd: () -> Void

    func makeUIView(context _: Context) -> PressAndHoldControl {
        let control = PressAndHoldControl()
        control.isEnabled = !isDisabled
        control.onPressStart = onPressStart
        control.onPressEnd = onPressEnd
        return control
    }

    func updateUIView(_ uiView: PressAndHoldControl, context _: Context) {
        uiView.isEnabled = !isDisabled
        uiView.onPressStart = onPressStart
        uiView.onPressEnd = onPressEnd
    }
}

final class PressAndHoldControl: UIControl {
    var onPressStart: (() -> Void)?
    var onPressEnd: (() -> Void)?
    private var isPressing = false

    override init(frame: CGRect) {
        super.init(frame: frame)
        isExclusiveTouch = true
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError() }

    override func beginTracking(_ touch: UITouch, with event: UIEvent?) -> Bool {
        _ = touch; _ = event
        guard isEnabled, !isPressing else { return false }
        isPressing = true
        onPressStart?()
        return true
    }

    override func endTracking(_ touch: UITouch?, with event: UIEvent?) {
        _ = touch; _ = event
        finishPressIfNeeded()
    }

    override func cancelTracking(with event: UIEvent?) {
        _ = event
        finishPressIfNeeded()
    }

    private func finishPressIfNeeded() {
        guard isPressing else { return }
        isPressing = false
        onPressEnd?()
    }
}

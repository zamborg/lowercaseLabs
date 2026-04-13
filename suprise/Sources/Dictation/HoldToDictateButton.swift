import SwiftUI
import UIKit

struct HoldToDictateButton: View {
    let isActive: Bool
    let isDisabled: Bool
    let onPressStart: () -> Void
    let onPressEnd: () -> Void

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: isDisabled
                            ? [Color.gray.opacity(0.35), Color.gray.opacity(0.25)]
                            : isActive
                                ? [Color(red: 0.13, green: 0.18, blue: 0.16), Color(red: 0.09, green: 0.13, blue: 0.12)]
                                : [Color(red: 0.17, green: 0.19, blue: 0.18), Color(red: 0.12, green: 0.14, blue: 0.13)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )

            HStack(spacing: 12) {
                Image(systemName: isActive ? "waveform.circle.fill" : "mic.fill")
                    .font(.system(size: 24, weight: .semibold))

                Text(isActive ? "Recording…" : (isDisabled ? "Preparing Model" : "Hold To Dictate"))
                    .font(.system(.headline, design: .rounded, weight: .bold))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 24)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 78)
        .overlay {
            PressAndHoldCaptureView(
                isDisabled: isDisabled,
                onPressStart: onPressStart,
                onPressEnd: onPressEnd
            )
        }
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(Color.white.opacity(isActive ? 0.34 : 0.18), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(isActive ? 0.22 : 0.14), radius: 18, x: 0, y: 10)
    }
}

private struct PressAndHoldCaptureView: UIViewRepresentable {
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

private final class PressAndHoldControl: UIControl {
    var onPressStart: (() -> Void)?
    var onPressEnd: (() -> Void)?

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

        guard isEnabled, !isPressing else {
            return false
        }

        isPressing = true
        onPressStart?()
        return true
    }

    override func endTracking(_ touch: UITouch?, with event: UIEvent?) {
        _ = touch
        _ = event
        finishPressIfNeeded()
    }

    override func cancelTracking(with event: UIEvent?) {
        _ = event
        finishPressIfNeeded()
    }

    private func finishPressIfNeeded() {
        guard isPressing else {
            return
        }

        isPressing = false
        onPressEnd?()
    }
}

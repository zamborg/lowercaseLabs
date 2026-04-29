import SwiftUI

enum AppTheme {
    static let background = LinearGradient(
        colors: [
            Color(red: 0.96, green: 0.92, blue: 0.84),
            Color(red: 0.84, green: 0.76, blue: 0.62),
            Color(red: 0.35, green: 0.19, blue: 0.15),
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let panel = Color.white.opacity(0.72)
    static let panelBorder = Color.white.opacity(0.35)
    static let ink = Color(red: 0.16, green: 0.11, blue: 0.08)
    static let accent = Color(red: 0.98, green: 0.46, blue: 0.21)
    static let accentStrong = Color(red: 0.84, green: 0.27, blue: 0.12)
    static let mist = Color.white.opacity(0.18)
}

struct FrostedPanelModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(18)
            .background(AppTheme.panel, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .stroke(AppTheme.panelBorder, lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.08), radius: 18, x: 0, y: 8)
    }
}

extension View {
    func frostedPanel() -> some View {
        modifier(FrostedPanelModifier())
    }
}


import SwiftUI
import ParlorCore

enum Palette {
    static let colors: [Color] = [
        Color(red: 0.12, green: 0.48, blue: 0.43), Color(red: 0.54, green: 0.35, blue: 0.62),
        Color(red: 0.75, green: 0.41, blue: 0.23), Color(red: 0.25, green: 0.44, blue: 0.71),
        Color(red: 0.48, green: 0.50, blue: 0.16), Color(red: 0.70, green: 0.26, blue: 0.30),
    ]
    static func color(_ i: Int) -> Color { colors[((i % colors.count) + colors.count) % colors.count] }
}

struct AvatarView: View {
    let agent: Agent
    var size: CGFloat = 28
    var body: some View {
        Circle().fill(Palette.color(agent.color).gradient)
            .frame(width: size, height: size)
            .overlay(Text(agent.initial).font(.system(size: size * 0.46, weight: .semibold, design: .rounded)).foregroundStyle(.white))
            .accessibilityHidden(true)
    }
}

func byteString(_ b: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: b, countStyle: .file)
}

func memoryString(_ b: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: b, countStyle: .memory)
}

func shortModelName(_ repo: String) -> String {
    repo.split(separator: "/").last.map(String.init) ?? repo
}

func tokenString(_ n: Int) -> String {
    n >= 1000 ? String(format: "%.1fk", Double(n) / 1000) : "\(n)"
}

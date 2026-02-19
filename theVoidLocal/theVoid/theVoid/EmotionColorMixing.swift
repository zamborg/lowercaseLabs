import SwiftUI
import UIKit

struct EmotionColorSlice: Identifiable, Hashable {
    let id: Int
    let tag: String
    let color: Color
    let weight: Double
}

enum EmotionColorMixer {
    static func slices(for tags: [String], maxSegments: Int = 8) -> [EmotionColorSlice] {
        let canonical = canonicalized(tags: tags)
        guard !canonical.isEmpty else {
            return []
        }

        let total = Double(canonical.count)
        return canonical.enumerated().map { index, tag in
            EmotionColorSlice(
                id: index,
                tag: tag,
                color: EmotionPalette.color(forTag: tag),
                weight: 1.0 / total
            )
        }
    }

    static func mixedColor(for tags: [String], fallbackHex: String? = nil) -> Color {
        let slices = slices(for: tags)
        guard !slices.isEmpty else {
            if let fallbackHex {
                return Color(hex: fallbackHex)
            }
            return Color(hex: "#6C6F74")
        }
        return blendedColor(from: slices)
    }

    static func mixedHex(for tags: [String], fallbackHex: String? = nil) -> String {
        let uiColor = UIColor(mixedColor(for: tags, fallbackHex: fallbackHex))
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        if uiColor.getRed(&red, green: &green, blue: &blue, alpha: &alpha) {
            let r = Int((red * 255.0).rounded())
            let g = Int((green * 255.0).rounded())
            let b = Int((blue * 255.0).rounded())
            return String(format: "#%02X%02X%02X", r, g, b)
        }
        return fallbackHex ?? "#6C6F74"
    }

    private static func canonicalized(tags: [String], maxSegments: Int = 8) -> [String] {
        var ordered: [String] = []
        for tag in tags {
            let canonical = EmotionTaxonomy.canonicalTag(for: tag)
            if !ordered.contains(canonical) {
                ordered.append(canonical)
            }
            if ordered.count >= maxSegments {
                break
            }
        }
        return ordered
    }

    private static func blendedColor(from slices: [EmotionColorSlice]) -> Color {
        let components = slices.map { (slice: EmotionColorSlice) -> (r: Double, g: Double, b: Double, weight: Double) in
            let ui = UIColor(slice.color)
            var red: CGFloat = 0
            var green: CGFloat = 0
            var blue: CGFloat = 0
            var alpha: CGFloat = 0
            let success = ui.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
            if success {
                return (Double(red), Double(green), Double(blue), slice.weight)
            }
            return (0.45, 0.45, 0.45, slice.weight)
        }

        let red = components.reduce(0.0) { $0 + ($1.r * $1.weight) }
        let green = components.reduce(0.0) { $0 + ($1.g * $1.weight) }
        let blue = components.reduce(0.0) { $0 + ($1.b * $1.weight) }
        return Color(red: red, green: green, blue: blue)
    }
}

struct EmotionMixedCircle: View {
    let tags: [String]
    var fallbackHex: String? = nil
    var diameter: CGFloat = 42
    var borderColor: Color = .white
    var borderOpacity: Double = 0.34

    var body: some View {
        let slices = EmotionColorMixer.slices(for: tags)

        ZStack {
            if slices.isEmpty {
                Circle()
                    .fill(EmotionColorMixer.mixedColor(for: tags, fallbackHex: fallbackHex))
            } else if slices.count == 1 {
                Circle()
                    .fill(slices[0].color)
            } else {
                ForEach(slices) { slice in
                    let start = startAngle(for: slice.id, total: slices.count)
                    let end = endAngle(for: slice.id, total: slices.count)
                    PieSliceShape(startAngle: start, endAngle: end)
                        .fill(slice.color)
                }
            }
        }
        .frame(width: diameter, height: diameter)
        .overlay(
            Circle()
                .stroke(borderColor.opacity(borderOpacity), lineWidth: 1)
        )
        .clipShape(Circle())
    }

    private func startAngle(for index: Int, total: Int) -> Angle {
        .degrees((Double(index) / Double(total)) * 360.0 - 90.0)
    }

    private func endAngle(for index: Int, total: Int) -> Angle {
        .degrees((Double(index + 1) / Double(total)) * 360.0 - 90.0)
    }
}

private struct PieSliceShape: Shape {
    let startAngle: Angle
    let endAngle: Angle

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let radius = min(rect.width, rect.height) / 2.0

        path.move(to: center)
        path.addArc(
            center: center,
            radius: radius,
            startAngle: startAngle,
            endAngle: endAngle,
            clockwise: false
        )
        path.closeSubpath()
        return path
    }
}

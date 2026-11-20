import SwiftUI

/// How We Feel quadrants. The only tag values that exist. Deterministic, hardcoded, never free text.
enum Quadrant: String, Codable, CaseIterable, Sendable {
    case highPleasant = "high_pleasant"
    case lowPleasant = "low_pleasant"
    case highUnpleasant = "high_unpleasant"
    case lowUnpleasant = "low_unpleasant"

    var color: Color {
        switch self {
        case .highPleasant: .yellow
        case .lowPleasant: .green
        case .highUnpleasant: .red
        case .lowUnpleasant: .blue
        }
    }

    var label: String {
        switch self {
        case .highPleasant: "high energy · pleasant"
        case .lowPleasant: "low energy · pleasant"
        case .highUnpleasant: "high energy · unpleasant"
        case .lowUnpleasant: "low energy · unpleasant"
        }
    }
}

import Foundation
import ParlorCore

/// Deterministic end-to-end flow check using a scripted model (no weights needed).
enum Flow {
    @MainActor static func run(_ engine: ParlorEngine) async -> Int32 { out("flow runs in the test target"); return 0 }
}

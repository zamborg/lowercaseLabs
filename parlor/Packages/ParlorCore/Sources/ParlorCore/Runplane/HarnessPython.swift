import Foundation

/// The Python that ships with parlor: the harness helper and the two harnesses built on it. Written to
/// `~/parlor/harnesses/` with the managed marker, so an edited copy is never overwritten.
enum HarnessPython {
    static let marker = "parlor-harnesses-v3"

    /// The shipped Python lives in `Resources/harnesses/`, one file each.
    static func load(_ file: String) -> String {
        guard let url = Bundle.module.url(forResource: (file as NSString).deletingPathExtension, withExtension: "py", subdirectory: "harnesses"),
              let text = try? String(contentsOf: url, encoding: .utf8) else { fatalError("ParlorCore is missing its resource \(file)") }
        return text
    }

    static let header = "# \(marker): managed by parlor. Copy this file to make your own; edits here are overwritten.\n"

    // MARK: parlor_harness.py

    static let helper = header + load("parlor_harness.py")

    // MARK: chat.py

    static let chat = header + load("chat.py")

    // MARK: character.py

    static let character = header + load("character.py")
}

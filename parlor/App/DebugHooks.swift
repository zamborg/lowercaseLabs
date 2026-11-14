import Foundation
import ParlorCore

/// Every `PARLOR_*` launch-environment switch the app reads, in one place. They let UI tests and screenshots
/// drive the app without scripting. None of them changes how a normal launch behaves.
///
/// - `PARLOR_HOME` (core) the parlor folder; `PARLOR_MODELS` (core) the model folder.
/// - `PARLOR_DEMO=1` scripted stand-in model, no weights needed.
/// - `PARLOR_ROUTE=agent:<id>|room:<id>|models|system` opens a screen directly (`tools` is an alias of `system`).
/// - `PARLOR_MODE=edit` opens the agent page on its editor.
/// - `PARLOR_AUTODOWNLOAD=repo[@image],…` (debug builds) starts those downloads.
/// - `PARLOR_AUTOSEND` / `PARLOR_AUTOSEND_AGENT` (debug builds) sends a first message.
enum DebugHooks {
    private static let env = ProcessInfo.processInfo.environment

    static var demo: Bool { env["PARLOR_DEMO"] == "1" }
    static var editMode: Bool { env["PARLOR_MODE"] == "edit" }

    static var route: Route? {
        switch env["PARLOR_ROUTE"] ?? "" {
        case "models": return .models
        case "system", "tools": return .system
        case let r where r.hasPrefix("agent:"): return .agent(String(r.dropFirst(6)))
        case let r where r.hasPrefix("room:"): return .room(String(r.dropFirst(5)))
        default: return nil
        }
    }

    /// Debug builds only: downloads and a first message requested by the environment.
    @MainActor static func applyStartupActions(_ engine: ParlorEngine) {
        #if DEBUG
        if let dl = env["PARLOR_AUTODOWNLOAD"] {
            for item in dl.split(separator: ",") {
                let parts = item.split(separator: "@")
                engine.library.download(String(parts[0]), kind: parts.count > 1 && parts[1] == "image" ? .image : .language)
            }
        }
        if let msg = env["PARLOR_AUTOSEND"], let id = env["PARLOR_AUTOSEND_AGENT"] ?? engine.agents.first?.id {
            engine.send(msg, to: id)
        }
        #endif
    }
}

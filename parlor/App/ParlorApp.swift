import SwiftUI
import ParlorCore

@main
struct ParlorApp: App {
    @State private var engine: ParlorEngine = {
        // PARLOR_DEMO=1 swaps in a scripted model so UI tests run with no weights.
        let demo = ProcessInfo.processInfo.environment["PARLOR_DEMO"] == "1"
        return ParlorEngine(runtime: demo ? DemoRuntime() : nil)
    }()
    @State private var route: Route? = nil

    var body: some Scene {
        WindowGroup {
            ContentView(route: $route)
                .environment(engine)
                .frame(minWidth: 940, minHeight: 600)
                .task {
                    await engine.start()
                    #if DEBUG
                    // Debug hooks so screenshots and checks can drive the app without UI scripting.
                    let env = ProcessInfo.processInfo.environment
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
                    if route == nil {
                        // PARLOR_ROUTE=agent:<id>|models|tools opens a screen directly (used for screenshots).
                        switch ProcessInfo.processInfo.environment["PARLOR_ROUTE"] ?? "" {
                        case "models": route = .models
                        case "tools": route = .tools
                        case "system": route = .system
                        case let r where r.hasPrefix("agent:"): route = .agent(String(r.dropFirst(6)))
                        case let r where r.hasPrefix("room:"): route = .room(String(r.dropFirst(5)))
                        default: if let first = engine.agents.first { route = .agent(first.id) }
                        }
                    }
                }
        }
        .defaultSize(width: 1180, height: 760)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Assistant") { let a = engine.create(kind: .assistant); route = .agent(a.id) }
                    .keyboardShortcut("n")
                Button("New Character") { let a = engine.create(kind: .character); route = .agent(a.id) }
                    .keyboardShortcut("n", modifiers: [.command, .shift])
            }
            CommandGroup(after: .appInfo) {
                Button("Reveal Parlor Folder in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([engine.home.root])
                }
            }
        }
    }
}

enum Route: Hashable {
    case agent(String)
    case room(String)
    case models
    case tools
    case system
}

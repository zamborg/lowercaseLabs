import SwiftUI
import ParlorCore

@main
struct ParlorApp: App {
    @State private var engine: ParlorEngine = {
        ParlorEngine(runtime: DebugHooks.demo ? DemoRuntime() : nil)
    }()
    @State private var route: Route? = nil

    var body: some Scene {
        WindowGroup {
            ContentView(route: $route)
                .environment(engine)
                .frame(minWidth: 940, minHeight: 600)
                .task {
                    engine.confirm = { question in
                        let a = NSAlert()
                        a.messageText = "Allow this?"
                        a.informativeText = question
                        a.alertStyle = .warning
                        a.addButton(withTitle: "Cancel")
                        a.addButton(withTitle: "Allow")
                        return a.runModal() == .alertSecondButtonReturn
                    }
                    await engine.start()
                    DebugHooks.applyStartupActions(engine)
                    if route == nil { route = DebugHooks.route ?? engine.agents.first.map { .agent($0.id) } }
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
    case system
}

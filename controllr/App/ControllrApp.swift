import SwiftUI
import SwiftData

@main
struct ControllrApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                SetupView().tabItem { Label("Setup", systemImage: "gearshape") }
                ModulesView().tabItem { Label("Modules", systemImage: "square.grid.2x2") }
                HistoryView().tabItem { Label("History", systemImage: "clock") }
                TryItView().tabItem { Label("Try it", systemImage: "text.cursor") }
            }
        }
        .modelContainer(Store.container)
    }
}

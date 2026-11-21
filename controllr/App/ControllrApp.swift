import SwiftUI
import SwiftData

@main
struct ControllrApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                ThreadsView().tabItem { Label("Threads", systemImage: "bubble.left.and.bubble.right") }
                SavedItemsView().tabItem { Label("Saved", systemImage: "bookmark") }
                ModulesView().tabItem { Label("Modules", systemImage: "square.grid.2x2") }
                HistoryView().tabItem { Label("History", systemImage: "clock") }
                SetupView().tabItem { Label("Setup", systemImage: "gearshape") }
            }
        }
        .modelContainer(Store.container)
    }
}

import SwiftData
import SwiftUI

@MainActor
final class AppModel {
    let container: ModelContainer
    let journal: Journal
    let capture: CaptureModel

    init() {
        container = Self.makeContainer()
        journal = Journal(context: container.mainContext, tagger: Tagger())
        capture = CaptureModel(journal: journal)
        capture.seedIfRequested()
    }

    /// Entries sync through the user's private iCloud database when this build carries iCloud. The store has its
    /// own name so it can never collide with the first build's `Memo` store.
    private static func makeContainer() -> ModelContainer {
        let cloud = ModelConfiguration("Entries", cloudKitDatabase: DotStore.enabled ? .private(DotStore.containerID) : .none)
        if let container = try? ModelContainer(for: Entry.self, configurations: cloud) { return container }
        return try! ModelContainer(for: Entry.self, configurations: ModelConfiguration("Entries", cloudKitDatabase: .none))
    }
}

@main
struct TheVoidApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var app = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView(app: app)
                .preferredColorScheme(.dark)
        }
        .modelContainer(app.container)
    }
}

struct RootView: View {
    let app: AppModel
    @State private var tab = 0

    var body: some View {
        TabView(selection: $tab) {
            Tab("Capture", systemImage: "circle.fill", value: 0) { CaptureView(model: app.capture) }
            Tab("Void", systemImage: "tray.full", value: 1) { VoidView(journal: app.journal) }
            Tab("Friends", systemImage: "person.2", value: 2) { SocialView() }
        }
        .onAppear {
            let args = ProcessInfo.processInfo.arguments
            if let i = args.firstIndex(of: "-tab"), i + 1 < args.count, let t = Int(args[i + 1]) { tab = t }
        }
    }
}

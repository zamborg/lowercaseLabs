import SwiftData
import SwiftUI

@main
struct TheVoidApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    private let container = TheVoidApp.makeContainer()

    var body: some Scene {
        WindowGroup {
            CaptureView()
                .preferredColorScheme(.dark)
        }
        .modelContainer(container)
    }

    /// Memos sync through the user's private iCloud database. Falls back to local-only if iCloud is unavailable.
    private static func makeContainer() -> ModelContainer {
        let cloud = ModelConfiguration(cloudKitDatabase: .private(DotStore.containerID))
        if let container = try? ModelContainer(for: Memo.self, configurations: cloud) { return container }
        return try! ModelContainer(for: Memo.self, configurations: ModelConfiguration(cloudKitDatabase: .none))
    }
}

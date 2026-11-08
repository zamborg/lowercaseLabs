import SwiftData
import SwiftUI

@main
struct TheVoidApp: App {
    var body: some Scene {
        WindowGroup {
            CaptureView()
                .preferredColorScheme(.dark)
                .modelContainer(for: Memo.self)
        }
    }
}

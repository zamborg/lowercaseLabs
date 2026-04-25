import SwiftUI

@main
struct BlackholeApp: App {
    @StateObject private var auth = AuthViewModel()

    init() {
        WhisperWarmupCoordinator.start()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(auth)
        }
    }
}

private enum WhisperWarmupCoordinator {
    private static var didStart = false

    static func start() {
        guard !didStart else { return }
        didStart = true

        Task.detached(priority: .utility) {
            // Start loading WhisperKit as soon as the process launches.
            try? await WhisperKitRuntime.shared.prepare()
        }
    }
}

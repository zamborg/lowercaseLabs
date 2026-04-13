import SwiftUI

struct RootView: View {
    @EnvironmentObject var auth: AuthViewModel

    var body: some View {
        Group {
            if auth.isSignedIn {
                MainTabView()
            } else {
                SignInView()
            }
        }
        .animation(.easeInOut(duration: 0.25), value: auth.isSignedIn)
    }
}

private struct MainTabView: View {
    var body: some View {
        TabView {
            IngestView()
                .tabItem { Label("Void", systemImage: "mic.fill") }
            FeedView()
                .tabItem { Label("Feed", systemImage: "list.bullet") }
            SearchView()
                .tabItem { Label("Search", systemImage: "magnifyingglass") }
        }
        .preferredColorScheme(.dark)
        .tint(.white)
    }
}

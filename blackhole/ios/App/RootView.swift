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
    @State private var selectedTab: MainAppTab = .feed
    @State private var isCapturePresented = false

    var body: some View {
        ZStack {
            selectedContent
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            MainTabBar(selectedTab: $selectedTab) {
                isCapturePresented = true
            }
        }
        .sheet(isPresented: $isCapturePresented) {
            IngestView()
        }
        .preferredColorScheme(.dark)
        .tint(.white)
    }

    @ViewBuilder
    private var selectedContent: some View {
        switch selectedTab {
        case .feed:
            FeedView()
        case .agent:
            AgentView()
        case .epics:
            EpicsView()
        case .notes:
            NotesView()
        case .todos:
            TodosView()
        }
    }
}

private enum MainAppTab: String, CaseIterable, Identifiable {
    case feed
    case agent
    case epics
    case notes
    case todos

    var id: String { rawValue }

    var title: String {
        switch self {
        case .feed: return "Feed"
        case .agent: return "Agent"
        case .epics: return "Epics"
        case .notes: return "Notes"
        case .todos: return "Todos"
        }
    }

    var systemImage: String {
        switch self {
        case .feed: return "list.bullet"
        case .agent: return "sparkles"
        case .epics: return "square.stack.3d.up"
        case .notes: return "note.text"
        case .todos: return "checkmark.circle"
        }
    }
}

private struct MainTabBar: View {
    @Binding var selectedTab: MainAppTab
    let onCapture: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 4) {
            ForEach(MainAppTab.allCases) { tab in
                Button {
                    selectedTab = tab
                } label: {
                    VStack(spacing: 3) {
                        Image(systemName: tab.systemImage)
                            .font(.system(size: 17, weight: .semibold))
                        Text(tab.title)
                            .font(.system(size: 10, weight: .semibold, design: .rounded))
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                    }
                    .foregroundStyle(selectedTab == tab ? .white : .white.opacity(0.46))
                    .frame(maxWidth: .infinity, minHeight: 52)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(tab.title)
            }

            Button(action: onCapture) {
                Image(systemName: "mic.fill")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.black)
                    .frame(width: 42, height: 42)
                    .background(.white)
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Open Void capture")
        }
        .padding(.horizontal, 10)
        .padding(.top, 8)
        .padding(.bottom, 8)
        .background(.black.opacity(0.96))
        .overlay(alignment: .top) {
            Rectangle()
                .fill(.white.opacity(0.08))
                .frame(height: 1)
        }
    }
}

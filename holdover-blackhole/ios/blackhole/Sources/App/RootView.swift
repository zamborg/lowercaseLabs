import Observation
import SwiftUI

struct RootView: View {
    let container: AppContainer

    var body: some View {
        if container.sessionController.isAuthenticated {
            MainShellView(
                sessionController: container.sessionController,
                navigator: container.navigator,
                captureViewModel: container.captureViewModel,
                todosViewModel: container.todosViewModel,
                notesViewModel: container.notesViewModel
            )
        } else {
            SignInView(sessionController: container.sessionController)
        }
    }
}

struct SignInView: View {
    @Bindable var sessionController: SessionController

    var body: some View {
        ZStack {
            AppTheme.background
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 24) {
                Text("blackhole")
                    .font(.system(size: 42, weight: .black, design: .rounded))
                    .foregroundStyle(AppTheme.ink)

                Text("Capture voice notes, todos, and recall in one pass.")
                    .font(.headline)
                    .foregroundStyle(AppTheme.ink.opacity(0.8))

                VStack(spacing: 16) {
                    TextField("API base URL", text: $sessionController.baseURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    TextField("Display name", text: $sessionController.displayName)
                    TextField("Email (optional)", text: $sessionController.email)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                .textFieldStyle(.roundedBorder)

                Text("Default backend: \(AppEnvironment.productionBaseURL)")
                    .font(.footnote)
                    .foregroundStyle(AppTheme.ink.opacity(0.65))

                Button {
                    Task { await sessionController.signInDev() }
                } label: {
                    HStack {
                        Text(sessionController.isSigningIn ? "Signing In..." : "Enter blackhole")
                            .font(.headline)
                        Spacer()
                        Image(systemName: "arrow.right.circle.fill")
                    }
                    .padding()
                    .foregroundStyle(.white)
                    .background(AppTheme.accentStrong, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                }
                .disabled(sessionController.isSigningIn)

                if let errorMessage = sessionController.errorMessage {
                    Text(errorMessage)
                        .font(.footnote)
                        .foregroundStyle(Color.red.opacity(0.85))
                }
            }
            .frostedPanel()
            .padding(24)
        }
    }
}

struct MainShellView: View {
    @Bindable var sessionController: SessionController
    @Bindable var navigator: AppNavigator
    @Bindable var captureViewModel: CaptureViewModel
    @Bindable var todosViewModel: TodosViewModel
    @Bindable var notesViewModel: NotesViewModel

    var body: some View {
        TabView(selection: $navigator.selectedTab) {
            CaptureView(model: captureViewModel)
                .tabItem {
                    Label("Capture", systemImage: "record.circle")
                }
                .tag(AppNavigator.Tab.capture)

            TodosView(model: todosViewModel)
                .tabItem {
                    Label("Todos", systemImage: "checklist")
                }
                .tag(AppNavigator.Tab.todos)

            NotesView(model: notesViewModel, navigator: navigator)
                .tabItem {
                    Label("Notes", systemImage: "note.text")
                }
                .tag(AppNavigator.Tab.notes)
        }
        .task {
            await notesViewModel.refresh()
            await todosViewModel.refresh()
        }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("Sign Out") {
                    sessionController.signOut()
                }
            }
        }
    }
}

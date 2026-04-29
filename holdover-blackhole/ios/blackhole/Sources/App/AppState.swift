import Foundation
import Observation

enum AppEnvironment {
    static let productionBaseURL = "https://blackhole-local.fly.dev"
    private static let legacyLocalBaseURLs: Set<String> = [
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ]

    static func resolvedBaseURL(from storedValue: String?) -> String {
        guard let storedValue else {
            return productionBaseURL
        }

        let trimmed = storedValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return productionBaseURL
        }
        if legacyLocalBaseURLs.contains(trimmed.lowercased()) {
            return productionBaseURL
        }
        return trimmed
    }
}

@MainActor
@Observable
final class SessionController {
    var session: StoredAuthSession?
    var baseURL: String
    var displayName: String
    var email: String
    var isSigningIn = false
    var errorMessage: String?

    private let store: AuthSessionStore
    private let authService: AuthService
    private let apiClient: APIClient

    init(store: AuthSessionStore, authService: AuthService, apiClient: APIClient) {
        self.store = store
        self.authService = authService
        self.apiClient = apiClient
        self.session = store.loadSession()
        self.baseURL = AppEnvironment.resolvedBaseURL(from: store.loadBaseURL())
        self.displayName = "blackhole"
        self.email = ""
        apiClient.setBaseURL(baseURL)
        apiClient.setAccessToken(session?.accessToken)
    }

    var isAuthenticated: Bool {
        session != nil
    }

    func signInDev() async {
        errorMessage = nil
        isSigningIn = true
        baseURL = AppEnvironment.resolvedBaseURL(from: baseURL)
        apiClient.setBaseURL(baseURL)
        store.saveBaseURL(baseURL)
        defer { isSigningIn = false }

        do {
            let signedIn = try await authService.signInDev(displayName: displayName, email: email.isEmpty ? nil : email)
            session = signedIn
            store.saveSession(signedIn)
            apiClient.setAccessToken(signedIn.accessToken)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func signOut() {
        session = nil
        apiClient.setAccessToken(nil)
        store.clearSession()
    }
}

@MainActor
@Observable
final class AppNavigator {
    enum Tab: Hashable {
        case capture
        case todos
        case notes
    }

    var selectedTab: Tab = .capture
    var pendingCaptureMode: CaptureMode?
    var pendingSearchQuery: String?

    func startVoiceTodo() {
        pendingCaptureMode = .todo
        selectedTab = .capture
    }

    func startVoiceSearch() {
        pendingCaptureMode = .search
        selectedTab = .capture
    }

    func consumePendingCaptureMode() -> CaptureMode? {
        defer { pendingCaptureMode = nil }
        return pendingCaptureMode
    }

    func openSearchResults(query: String) {
        pendingSearchQuery = query
        selectedTab = .notes
    }

    func consumePendingSearchQuery() -> String? {
        defer { pendingSearchQuery = nil }
        return pendingSearchQuery
    }
}

@MainActor
final class AppContainer {
    let sessionController: SessionController
    let navigator = AppNavigator()
    let notesViewModel: NotesViewModel
    let todosViewModel: TodosViewModel
    let captureViewModel: CaptureViewModel

    init() {
        let store = UserDefaultsAuthSessionStore()
        let runtime = RuntimeConfiguration(
            baseURLString: AppEnvironment.resolvedBaseURL(from: store.loadBaseURL()),
            accessToken: store.loadSession()?.accessToken
        )
        let apiClient = APIClient(config: runtime)
        let authService = RemoteAuthService(apiClient: apiClient)
        let notesRepository = RemoteNotesRepository(apiClient: apiClient)
        let todosRepository = RemoteTodosRepository(apiClient: apiClient)
        let searchRepository = RemoteSearchRepository(apiClient: apiClient)

        sessionController = SessionController(store: store, authService: authService, apiClient: apiClient)
        notesViewModel = NotesViewModel(notesRepository: notesRepository, searchRepository: searchRepository, navigator: navigator)
        todosViewModel = TodosViewModel(todosRepository: todosRepository, navigator: navigator)
        captureViewModel = CaptureViewModel(
            audioCaptureService: DefaultAudioCaptureService(),
            transcriptionClient: RemoteTranscriptionSessionClient(apiClient: apiClient),
            routerClient: DefaultCaptureRouterClient(),
            notesRepository: notesRepository,
            todosRepository: todosRepository,
            navigator: navigator,
            onNoteCreated: { [notesViewModel] in
                await notesViewModel.refresh()
            },
            onTodoCreated: { [todosViewModel] in
                await todosViewModel.refresh()
            }
        )
    }
}

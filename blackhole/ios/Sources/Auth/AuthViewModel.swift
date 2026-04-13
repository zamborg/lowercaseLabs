import AuthenticationServices
import Foundation
import SwiftUI

@MainActor
final class AuthViewModel: ObservableObject {
    @Published private(set) var isSignedIn = false
    @Published var alertMessage: String?

    private let tokenKey = "blackhole.sessionToken"

    init() {
        if let token = UserDefaults.standard.string(forKey: tokenKey) {
            Task { await APIClient.shared.setSessionToken(token) }
            isSignedIn = true
        }
    }

    func handleAppleSignIn(result: Result<ASAuthorization, Error>) async {
        switch result {
        case .success(let auth):
            guard
                let credential = auth.credential as? ASAuthorizationAppleIDCredential,
                let tokenData = credential.identityToken,
                let tokenString = String(data: tokenData, encoding: .utf8)
            else {
                alertMessage = "Failed to read Apple credential."
                return
            }
            do {
                let sessionToken = try await APIClient.shared.signIn(identityToken: tokenString)
                UserDefaults.standard.set(sessionToken, forKey: tokenKey)
                await APIClient.shared.setSessionToken(sessionToken)
                isSignedIn = true
            } catch {
                alertMessage = error.localizedDescription
            }

        case .failure(let error):
            alertMessage = error.localizedDescription
        }
    }

    func signOut() {
        UserDefaults.standard.removeObject(forKey: tokenKey)
        Task { await APIClient.shared.setSessionToken(nil) }
        isSignedIn = false
    }
}

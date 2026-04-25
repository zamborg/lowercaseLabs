import AuthenticationServices
import SwiftUI

struct SignInView: View {
    @EnvironmentObject var auth: AuthViewModel

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()

                VStack(spacing: 10) {
                    Text("blackhole")
                        .font(.system(.largeTitle, design: .monospaced, weight: .bold))
                        .foregroundStyle(.white)

                    Text("notes, epics & to-dos, captured by voice")
                        .font(.system(.subheadline, design: .rounded))
                        .foregroundStyle(.white.opacity(0.45))
                }

                Spacer()

                VStack(spacing: 16) {
                    SignInWithAppleButton(.signIn) { request in
                        request.requestedScopes = [.email, .fullName]
                    } onCompletion: { result in
                        Task { await auth.handleAppleSignIn(result: result) }
                    }
                    .signInWithAppleButtonStyle(.white)
                    .frame(height: 50)
                    .padding(.horizontal, 40)
                }
                .padding(.bottom, 60)
            }
        }
        .preferredColorScheme(.dark)
        .alert("Sign In Failed", isPresented: Binding(
            get: { auth.alertMessage != nil },
            set: { if !$0 { auth.alertMessage = nil } }
        )) {
            Button("OK") { auth.alertMessage = nil }
        } message: {
            Text(auth.alertMessage ?? "")
        }
    }
}

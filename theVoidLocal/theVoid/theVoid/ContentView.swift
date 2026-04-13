//
//  ContentView.swift
//  theVoid
//
//  Created by zubin aysola on 2/16/26.
//

import AuthenticationServices
import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var model = AppModel()

    var body: some View {
        Group {
            if model.needsOnboarding {
                OnboardingView()
            } else if model.showsLiquidModelPreparationScreen {
                LiquidModelPreparationView()
            } else {
                MainTabView()
            }
        }
        .environmentObject(model)
        .task {
            await model.handleAppDidBecomeActive()
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)) { _ in
            Task {
                await model.handleAppDidBecomeActive()
            }
        }
        .alert("Error", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
        .sheet(isPresented: Binding(
            get: { model.showsSessionReauthenticationPrompt },
            set: { if !$0 { model.dismissSessionReauthenticationPrompt() } }
        )) {
            SessionReauthenticationPrompt()
                .environmentObject(model)
                .presentationDetents([.medium])
                .presentationDragIndicator(.visible)
        }
    }
}

struct MainTabView: View {
    var body: some View {
        TabView {
            VoidExperienceView()
                .tabItem {
                    Label("Void", systemImage: "waveform")
                }

            JournalView()
                .tabItem {
                    Label("Journal", systemImage: "book")
                }

            EmotionAtlasScreen()
                .tabItem {
                    Label("Atlas", systemImage: "sparkles.rectangle.stack")
                }

            SocialView()
                .tabItem {
                    Label("Social", systemImage: "circle.grid.3x3.fill")
                }

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gearshape")
                }
        }
    }
}

struct SessionReauthenticationPrompt: View {
    @EnvironmentObject private var model: AppModel
    @State private var currentAppleNonce: String?

    private var promptMessage: String {
        model.sessionReauthenticationMessage
            ?? "Your backend session expired. Sign in again to restore social features."
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Capsule()
                .fill(Color.secondary.opacity(0.3))
                .frame(width: 40, height: 5)
                .frame(maxWidth: .infinity)

            Text("Reconnect Session")
                .font(.title2.weight(.semibold))

            Text(promptMessage)
                .font(.body)
                .foregroundStyle(.secondary)

            Text("Local entries, transcripts, and audio stay on this device.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            SignInWithAppleButton(.signIn) { request in
                request.requestedScopes = [.fullName]
                let nonce = AppleNonce.random()
                currentAppleNonce = nonce
                request.nonce = AppleNonce.sha256(nonce)
            } onCompletion: { result in
                switch result {
                case .failure(let error):
                    model.errorMessage = error.localizedDescription
                case .success(let auth):
                    guard let credential = auth.credential as? ASAuthorizationAppleIDCredential,
                          let tokenData = credential.identityToken,
                          let token = String(data: tokenData, encoding: .utf8) else {
                        model.errorMessage = "Unable to read Apple identity token"
                        return
                    }

                    let fullName = [credential.fullName?.givenName, credential.fullName?.familyName]
                        .compactMap { $0 }
                        .joined(separator: " ")
                    let nonce = currentAppleNonce
                    currentAppleNonce = nil
                    Task {
                        await model.signIn(
                            identityToken: token,
                            nonce: nonce,
                            suggestedName: fullName.isEmpty ? nil : fullName
                        )
                    }
                }
            }
            .signInWithAppleButtonStyle(.black)
            .frame(height: 48)

            Button("Not Now") {
                model.dismissSessionReauthenticationPrompt()
            }
            .buttonStyle(.bordered)
            .frame(maxWidth: .infinity, alignment: .center)

            Spacer(minLength: 0)
        }
        .padding(24)
    }
}

#Preview {
    ContentView()
}

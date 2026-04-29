import AuthenticationServices
import AVFoundation
import SwiftUI
import UIKit

struct LiquidModelPreparationView: View {
    @EnvironmentObject private var model: AppModel

    private var progress: Double {
        max(0, min(1, model.liquidModelPreparationProgress))
    }

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color.black, Color(red: 0.03, green: 0.08, blue: 0.12)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(spacing: 22) {
                Spacer()

                Text("Preparing On-Device AI")
                    .font(.system(size: 31, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)

                Text("Downloading the Liquid model to this device.")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.74))

                ZStack {
                    Circle()
                        .stroke(Color.white.opacity(0.16), lineWidth: 12)

                    Circle()
                        .trim(from: 0, to: max(progress, 0.01))
                        .stroke(
                            AngularGradient(
                                colors: [
                                    Color.cyan.opacity(0.95),
                                    Color.teal.opacity(0.95),
                                    Color.cyan.opacity(0.95),
                                ],
                                center: .center
                            ),
                            style: StrokeStyle(lineWidth: 12, lineCap: .round)
                        )
                        .rotationEffect(.degrees(-90))
                        .animation(.easeInOut(duration: 0.24), value: progress)

                    Text("\(Int((progress * 100).rounded()))%")
                        .font(.system(size: 22, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                }
                .frame(width: 148, height: 148)

                Text(model.liquidModelPreparationStatus)
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.70))
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 320)

                if let error = model.liquidModelPreparationError {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(.red.opacity(0.9))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 24)

                    Button("Retry Download") {
                        model.retryLiquidModelPreparation()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.teal)
                }

                VStack(spacing: 10) {
                    Text("The insights engine needs this model. You can also download it later from the Settings > On-Device AI.")
                        .font(.footnote)
                        .foregroundStyle(.white.opacity(0.72))
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 340)

                    Button("Cancel") {
                        model.cancelLiquidModelPreparation()
                    }
                    .buttonStyle(.bordered)
                    .tint(.white.opacity(0.88))
                }
                .padding(.top, 6)

                Spacer()
            }
            .padding(24)
        }
    }
}

// MARK: - Onboarding

struct OnboardingView: View {
    @EnvironmentObject private var model: AppModel

    @State private var micGranted = false
    @State private var currentAppleNonce: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Text("theVoid")
                    .font(.system(size: 42, weight: .bold, design: .rounded))
                Text("One intentional check-in. Private by default.")
                    .font(.headline)
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 12) {
                    Text("Identity")
                        .font(.headline)

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
                                  let token = String(data: tokenData, encoding: .utf8)
                            else {
                                model.errorMessage = "Unable to read Apple identity token"
                                return
                            }

                            let fullName = [credential.fullName?.givenName, credential.fullName?.familyName]
                                .compactMap { $0 }
                                .joined(separator: " ")
                            let nonce = currentAppleNonce
                            currentAppleNonce = nil
                            Task {
                                await model.signIn(identityToken: token, nonce: nonce, suggestedName: fullName.isEmpty ? nil : fullName)
                            }
                        }
                    }
                    .signInWithAppleButtonStyle(.white)
                    .frame(height: 48)

                    if !model.anonymousHandle.isEmpty {
                        Text("Signed in as @\(model.anonymousHandle)")
                            .font(.subheadline)
                    }
                    
                    TextField("Display name (optional)", text: $model.displayName)
                        .textFieldStyle(.plain)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(Color.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 10))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color.white.opacity(0.16), lineWidth: 1)
                        )
                    
                }

//                VStack(alignment: .leading, spacing: 12) {
//                    Text("Check-In")
//                        .font(.headline)
//                    DatePicker("Default time", selection: $model.dailyCheckin, displayedComponents: .hourAndMinute)
//                    ReminderScheduleEditor()
//
//                    Button("Save Reminder Schedule") {
//                        Task { await model.configureDailyReminder() }
//                    }
//                    .buttonStyle(.borderedProminent)
//
//                    if let reminderStatus = model.reminderStatus {
//                        Text(reminderStatus)
//                            .font(.footnote)
//                            .foregroundStyle(.secondary)
//                    }
//
//                }

                VStack(alignment: .leading, spacing: 12) {
                    Text("Permissions")
                        .font(.headline)

                    HStack {
                        Label("Microphone", systemImage: micGranted ? "checkmark.circle.fill" : "circle")
                        Spacer()
                        Button(micGranted ? "Granted" : "Allow") {
                            Task {
                                micGranted = await RecorderEngine().requestPermission()
                            }
                        }
                        .disabled(micGranted)
                    }

                    HStack {
                        Label("Health", systemImage: healthIconName)
                        Spacer()
                        Button(healthActionLabel) {
                            Task {
                                await runHealthPermissionAction()
                            }
                        }
                        .disabled(!canRunHealthAction)
                    }

                    Text("Health is used locally to create a readiness score at note time.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Button("Enter the Void") {
                    model.completeOnboarding()
                    Task {
                        await model.saveProfile()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.sessionToken == nil)
            }
            .padding(24)
        }
        .background(Color.black.ignoresSafeArea())
        .foregroundStyle(.white)
        .onAppear {
            micGranted = AVAudioSession.sharedInstance().recordPermission == .granted
            Task {
                await model.refreshHealthAuthorizationState()
            }
        }
    }

    private var healthIconName: String {
        switch model.healthAuthorizationState {
        case .authorizedAll:
            return "checkmark.circle.fill"
        case .authorizedPartial:
            return "checkmark.circle"
        case .notDetermined:
            return "circle"
        case .denied:
            return "xmark.circle.fill"
        case .unavailable:
            return "slash.circle"
        }
    }

    private var healthActionLabel: String {
        switch model.healthAuthorizationState {
        case .authorizedAll:
            return "Granted"
        case .authorizedPartial:
            return "Granted"
        case .notDetermined:
            return "Allow"
        case .denied:
            return "Open Health"
        case .unavailable:
            return "Unavailable"
        }
    }

    private var canRunHealthAction: Bool {
        switch model.healthAuthorizationState {
        case .authorizedAll, .authorizedPartial, .unavailable:
            return false
        case .notDetermined, .denied:
            return true
        }
    }

    private func runHealthPermissionAction() async {
        switch model.healthAuthorizationState {
        case .denied:
            await model.requestHealthAuthorization()
            if model.healthAuthorizationState == .denied {
                await MainActor.run {
                    UIApplication.openHealthAccessManagement()
                }
            }
        case .unavailable, .authorizedAll, .authorizedPartial:
            break
        case .notDetermined:
            await model.requestHealthAuthorization()
        }
    }
}

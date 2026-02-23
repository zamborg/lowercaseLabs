//
//  ContentView.swift
//  theVoid
//
//  Created by zubin aysola on 2/16/26.
//

import SwiftUI

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
        .alert("Error", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
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

#Preview {
    ContentView()
}

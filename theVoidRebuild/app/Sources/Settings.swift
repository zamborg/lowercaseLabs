import SwiftUI

/// The few things the user controls (D18). Stored on this device only, never synced. Every default is already right:
/// the app is complete without ever opening the panel.
enum Settings {
    static let cameraBackOnlyKey = "cameraBackOnly"
    static let showTimingsKey = "showTimings"

    static var cameraBackOnly: Bool { UserDefaults.standard.bool(forKey: cameraBackOnlyKey) }
    /// `-debug` still forces timings on, for launch-argument runs.
    static var showTimings: Bool { UserDefaults.standard.bool(forKey: showTimingsKey) || ProcessInfo.processInfo.arguments.contains("-debug") }
}

struct SettingsView: View {
    let journal: Journal
    @AppStorage(Settings.cameraBackOnlyKey) private var cameraBackOnly = false
    @AppStorage(Settings.showTimingsKey) private var showTimings = false
    @State private var retagging = false
    @State private var retagged = false
    @Environment(\.dismiss) private var dismiss

    private var version: String {
        let info = Bundle.main.infoDictionary
        return "\(info?["CFBundleShortVersionString"] as? String ?? "?") (\(info?["CFBundleVersion"] as? String ?? "?"))"
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Camera") {
                    Toggle("Back camera only", isOn: $cameraBackOnly)
                        .accessibilityIdentifier("backOnlyToggle")
                }
                Section {
                    Toggle("Show timings", isOn: $showTimings)
                        .accessibilityIdentifier("timingsToggle")
                } footer: { Text("Capture and tagging times on the Capture screen.") }
                Section {
                    Button(retagging ? "Re-tagging…" : (retagged ? "Done" : "Re-tag my messages")) {
                        retagging = true; retagged = false
                        Task { await journal.retagModelFields(); retagging = false; retagged = true }
                    }
                    .disabled(retagging)
                    .accessibilityIdentifier("retagButton")
                } footer: { Text("Runs the model again over every title and tag it chose. Anything you edited stays as you left it.") }
                Section("About") {
                    LabeledContent("Version", value: version)
                    LabeledContent("iCloud", value: DotStore.enabled ? "on" : "off in this build")
                    LabeledContent("Dots", value: !DotStore.enabled ? "off" : (DotStore.shared.status.isEmpty ? "ok" : DotStore.shared.status))
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .preferredColorScheme(.dark)
    }
}

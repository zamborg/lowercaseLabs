import SwiftUI

@main
struct ControllrApp: App {
    var body: some Scene { WindowGroup { SpikeView() } }
}

/// Phase 0 only: paste an OpenAI key into the Keychain so DecideNetworkIntent can use it.
struct SpikeView: View {
    @State private var key = ""
    @State private var saved = Keychain.get("openai") != nil

    var body: some View {
        Form {
            Section("OpenAI key (spike)") {
                SecureField("sk-…", text: $key)
                Button("Save to Keychain") {
                    Keychain.set(key, for: "openai")
                    key = ""
                    saved = true
                }
                .disabled(key.isEmpty)
                Text(saved ? "Key stored" : "No key stored").foregroundStyle(.secondary)
            }
        }
    }
}

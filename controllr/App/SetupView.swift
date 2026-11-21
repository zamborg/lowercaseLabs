import SwiftUI
import ControllrCore

struct SetupView: View {
    @State private var provider = Settings.provider
    @State private var openaiModel = Settings.openaiModel
    @State private var anthropicModel = Settings.anthropicModel
    @State private var home = Settings.homeAddress
    @State private var nicknames = Settings.nicknamesText
    @State private var keyDraft = ""
    @State private var keySaved: [String: Bool] = ["openai": Keychain.get("openai") != nil, "anthropic": Keychain.get("anthropic") != nil]
    @State private var models: [String] = []
    @State private var status = ""
    @State private var contacts = Contacts.authorized
    @State private var reminders = Reminders.authorized

    private var model: Binding<String> {
        provider == "anthropic" ? $anthropicModel : $openaiModel
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Provider") {
                    Picker("Provider", selection: $provider) {
                        Text("OpenAI").tag("openai"); Text("Anthropic").tag("anthropic")
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: provider) { _, v in Settings.provider = v; models = []; status = "" }
                }

                Section("API key") {
                    SecureField(keySaved[provider] == true ? "Key stored — paste to replace" : "Paste API key", text: $keyDraft)
                    HStack {
                        PasteButton(payloadType: String.self) { strings in
                            if let s = strings.first { keyDraft = s.trimmingCharacters(in: .whitespacesAndNewlines) }
                        }
                        Spacer()
                        Button("Save to Keychain") {
                            Keychain.set(keyDraft, for: provider); keyDraft = ""; keySaved[provider] = true; status = "Key saved"
                        }.disabled(keyDraft.isEmpty)
                    }
                    Button("Test key & load models") { Task { await testKey() } }
                        .disabled(keySaved[provider] != true)
                    if !status.isEmpty { Text(status).font(.footnote).foregroundStyle(.secondary) }
                }

                Section("Model") {
                    if models.isEmpty {
                        TextField("Model id (or test the key to pick)", text: model)
                            .textInputAutocapitalization(.never).autocorrectionDisabled()
                    } else {
                        Picker("Model", selection: model) {
                            ForEach(models, id: \.self) { Text($0).tag($0) }
                        }
                    }
                }
                .onChange(of: openaiModel) { _, v in Settings.openaiModel = v }
                .onChange(of: anthropicModel) { _, v in Settings.anthropicModel = v }

                Section("About you") {
                    TextField("Home address", text: $home, axis: .vertical).onChange(of: home) { _, v in Settings.homeAddress = v }
                    VStack(alignment: .leading) {
                        Text("Nicknames (one per line: mom=Linda Smith)").font(.footnote).foregroundStyle(.secondary)
                        TextEditor(text: $nicknames).frame(minHeight: 70)
                            .textInputAutocapitalization(.never).autocorrectionDisabled()
                            .onChange(of: nicknames) { _, v in Settings.nicknamesText = v }
                    }
                }

                Section("Permissions") {
                    Button(contacts ? "Contacts: allowed" : "Allow Contacts") { Task { contacts = await Contacts.request() } }.disabled(contacts)
                    Button(reminders ? "Reminders: allowed" : "Allow Reminders") { Task { reminders = await Reminders.request() } }.disabled(reminders)
                }

                Section("Router shortcut") {
                    Text("1. Install `Controllr · Router` in Shortcuts.\n2. Run it once and tap Allow when asked.\n3. Bind it: Settings → Action Button → Shortcut, or Accessibility → Touch → Back Tap, or add it to Control Center.\nSay what you want after the router starts listening.")
                        .font(.footnote)
                }
            }
            .navigationTitle("Setup")
        }
    }

    private func testKey() async {
        guard let key = Keychain.get(provider) else { return }
        let p: LLMProvider = provider == "anthropic" ? AnthropicProvider(apiKey: key, model: "x") : OpenAIProvider(apiKey: key, model: "x")
        status = "Testing…"
        do {
            let all = try await p.listModels()
            // Chat-capable ids only; the user picks, we never guess a default.
            models = all.filter { id in
                provider == "anthropic" ? true
                : (id.hasPrefix("gpt") || id.hasPrefix("o")) && !["audio", "realtime", "tts", "transcribe", "image", "search", "embedding", "instruct", "codex", "moderation"].contains { id.contains($0) }
            }.sorted()
            if !models.contains(model.wrappedValue) { model.wrappedValue = models.first ?? "" }
            status = "Key works — \(models.count) models"
        } catch let e as ProviderError {
            if case .http(let s, _) = e { status = s == 401 ? "Key rejected (401)" : "Provider error (\(s))" } else { status = "Network problem" }
        } catch { status = "Failed" }
    }
}

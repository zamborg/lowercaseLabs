import SwiftUI
import ControllrCore

struct SetupView: View {
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.openURL) private var openURL
    @State private var provider = Settings.provider
    @State private var home = Settings.homeAddress
    @State private var nicknames = Settings.nicknamesText
    @State private var contacts = ModulePermission.contacts.accessLabel
    @State private var reminders = ModulePermission.reminders.accessLabel
    @State private var calendar = ModulePermission.calendar.accessLabel
    @State private var timers = ModulePermission.timers.accessLabel
    @AppStorage(RouterShortcut.preferenceKey) private var routerShortcut: RouterShortcut = .legacy

    var body: some View {
        NavigationStack {
            Form {
                Section("Provider") {
                    Picker("Provider", selection: $provider) {
                        ForEach(CompletionProvider.allCases, id: \.self) { provider in
                            Text(provider.title).tag(provider)
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: provider) { _, provider in Settings.provider = provider }
                }

                ProviderConfiguration(provider: provider).id(provider)

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
                    permissionButton(.calendar, status: calendar) { await PhoneCalendar.request() }
                    permissionButton(.contacts, status: contacts) { await Contacts.request() }
                    permissionButton(.reminders, status: reminders) { await Reminders.request() }
                    if PhoneTimers.available {
                        permissionButton(.timers, status: timers) { await PhoneTimers.request() }
                    }
                    LabeledContent("Messages and other Shortcut actions", value: ModulePermission.shortcuts.accessLabel)
                }

                Section("Agent") {
                    NavigationLink("System Prompt") { SystemPromptView() }
                }

                Section("Diagnostics") {
                    NavigationLink("Preview a request") { TryItView() }
                }

                Section("Router shortcut") {
                    Picker("Voice shortcut", selection: $routerShortcut) {
                        ForEach(RouterShortcut.allCases, id: \.self) { shortcut in
                            Text(shortcut.rawValue).tag(shortcut)
                        }
                    }
                    Text("Install \(routerShortcut.rawValue) and run it once to grant access. You can assign it to the Action Button, Back Tap, or Control Center.")
                        .font(.footnote)
                }
            }
            .navigationTitle("Setup")
            .onChange(of: scenePhase) { _, phase in
                if phase == .active { refreshPermissions() }
            }
        }
    }

    private func permissionButton(_ permission: ModulePermission, status: String,
                                  request: @escaping () async -> Bool) -> some View {
        Button {
            if status == "Not requested" || status == "Write only" {
                Task {
                    _ = await request()
                    refreshPermissions()
                }
            } else if let settings = URL(string: UIApplication.openSettingsURLString) {
                openURL(settings)
            }
        } label: {
            LabeledContent(permission.title, value: status)
        }
        .disabled(status == "Restricted")
    }

    private func refreshPermissions() {
        calendar = ModulePermission.calendar.accessLabel
        contacts = ModulePermission.contacts.accessLabel
        reminders = ModulePermission.reminders.accessLabel
        timers = ModulePermission.timers.accessLabel
    }
}

private struct ProviderConfiguration: View {
    let provider: CompletionProvider
    @State private var model: String
    @State private var reasoningEffort: ReasoningEffort
    @State private var keySaved: Bool
    @State private var keyDraft = ""
    @State private var keyError = false
    @State private var models: [ProviderModel] = []
    @State private var catalogStatus = ""
    @State private var loading = false
    @State private var refresh = 0

    init(provider: CompletionProvider) {
        self.provider = provider
        let model = Settings.model(for: provider)
        _model = State(initialValue: model)
        _reasoningEffort = State(initialValue: Settings.reasoningEffort(for: provider, model: model))
        _keySaved = State(initialValue: Keychain.get(provider.rawValue)?.isEmpty == false)
    }

    private var modelSelection: Binding<String> {
        Binding(get: { model }, set: { value in
            model = value
            Settings.setModel(value, for: provider)
            reasoningEffort = Settings.reasoningEffort(for: provider, model: value)
        })
    }

    private var reasoningOptions: [ReasoningEffort] {
        let id = model.trimmingCharacters(in: .whitespacesAndNewlines)
        return provider.reasoningOptions(for: id, catalog: models.first { $0.id == id })
    }

    private var reasoningSelection: Binding<ReasoningEffort> {
        Binding(get: { reasoningEffort }, set: { effort in
            reasoningEffort = effort
            Settings.setReasoningEffort(effort, for: provider, model: model)
        })
    }

    var body: some View {
        Section("API key") {
            SecureField(keySaved ? "Key stored — paste to replace" : "\(provider.title) API key", text: $keyDraft)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            HStack {
                PasteButton(payloadType: String.self) { strings in
                    if let value = strings.first { keyDraft = value.trimmingCharacters(in: .whitespacesAndNewlines) }
                }
                Spacer()
                Button("Save key", action: saveKey)
                    .disabled(keyDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            if keyError {
                Text("Couldn't save the key.").font(.footnote).foregroundStyle(.red)
            }
        }
        Section("Model") {
            if models.isEmpty {
                TextField("Model ID", text: modelSelection)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
            } else {
                NavigationLink {
                    ModelPicker(models: models, selection: modelSelection)
                } label: {
                    LabeledContent("Choose model", value: models.first { $0.id == model }?.name ?? (model.isEmpty ? "Select" : model))
                }
            }
            if !reasoningOptions.isEmpty || reasoningEffort != .automatic {
                Picker("Reasoning", selection: reasoningSelection) {
                    Text(ReasoningEffort.automatic.title).tag(ReasoningEffort.automatic)
                    ForEach(reasoningOptions, id: \.self) { effort in
                        Text(effort.title).tag(effort)
                    }
                    if reasoningEffort != .automatic && !reasoningOptions.contains(reasoningEffort) {
                        Text("\(reasoningEffort.title) (unavailable)").tag(reasoningEffort)
                    }
                }
            }
            if loading {
                ProgressView("Loading models…")
            } else if keySaved {
                Button("Refresh models") { refresh += 1 }
            }
            if !catalogStatus.isEmpty {
                Text(catalogStatus).font(.footnote).foregroundStyle(.secondary)
            }
        }
        .task(id: refresh) { await loadModels() }
    }

    private func saveKey() {
        let trimmed = keyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard Keychain.set(trimmed, for: provider.rawValue) else { keyError = true; return }
        keyError = false
        keySaved = true
        keyDraft = ""
        models = []
        refresh += 1
    }

    private func loadModels() async {
        guard let key = Keychain.get(provider.rawValue), !key.isEmpty else { return }
        let request = refresh
        loading = true
        catalogStatus = ""
        defer { if request == refresh && !Task.isCancelled { loading = false } }
        do {
            let listed = try await provider.makeProvider(apiKey: key, model: model).listModels()
            guard request == refresh, !Task.isCancelled else { return }
            var seen: Set<String> = []
            models = listed.filter { seen.insert($0.id).inserted }.sorted {
                let comparison = $0.name.localizedStandardCompare($1.name)
                return comparison == .orderedSame ? $0.id < $1.id : comparison == .orderedAscending
            }
            if models.isEmpty { catalogStatus = "No compatible models found for this key." }
        } catch {
            guard request == refresh, !Task.isCancelled else { return }
            switch error {
            case ProviderError.http(let status, _) where status == 401 || status == 403:
                models = []
                catalogStatus = "Key rejected by \(provider.title)."
            default:
                catalogStatus = "Couldn't load models. Try refreshing."
            }
        }
    }
}

private struct ModelPicker: View {
    let models: [ProviderModel]
    @Binding var selection: String
    @State private var query = ""
    @Environment(\.dismiss) private var dismiss

    private var matches: [ProviderModel] {
        let terms = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !terms.isEmpty else { return models }
        return models.filter { $0.name.localizedStandardContains(terms) || $0.id.localizedStandardContains(terms) }
    }

    var body: some View {
        List(matches) { model in
            Button {
                selection = model.id
                dismiss()
            } label: {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(model.name)
                        if model.name != model.id {
                            Text(model.id).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    if model.id == selection { Image(systemName: "checkmark").foregroundStyle(.tint) }
                }
                .foregroundStyle(.primary)
            }
            .buttonStyle(.plain)
        }
        .overlay { if matches.isEmpty { ContentUnavailableView.search(text: query) } }
        .navigationTitle("Models")
        .searchable(text: $query, prompt: "Search models")
    }
}

import ControllrCore
import SwiftUI

struct ModulesView: View {
    @AppStorage("modulePreferences") private var preferencesData = Data()

    var body: some View {
        NavigationStack {
            List(Registry.v1.modules.filter { $0.kind == .native || $0.kind == .shortcut }, id: \.id) { module in
                NavigationLink {
                    ModuleDetail(module: module, preferencesData: $preferencesData)
                } label: {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(module.displayTitle)
                        Text(status(for: module))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Modules")
        }
    }

    private func status(for module: Module) -> String {
        let policy = ModulePolicy(data: preferencesData)
        guard policy.isEnabled(module) else { return "Off" }
        if module.hasConfirmationSetting && policy.requiresConfirmation(module) { return "Asks first" }
        return "Enabled"
    }
}

private struct ModuleDetail: View {
    let module: Module
    @Binding var preferencesData: Data
    @AppStorage("preferredCalendar") private var calendar = ""
    @AppStorage("preferredReminderList") private var reminderList = ""
    @AppStorage("lyftShortcut") private var lyftShortcut = ""

    private var preference: Binding<ModulePreference> {
        Binding {
            ModulePolicy(data: preferencesData).preferences[module.id] ?? ModulePreference()
        } set: { preference in
            var preferences = ModulePolicy(data: preferencesData).preferences
            preferences[module.id] = preference
            if let data = try? JSONEncoder().encode(preferences) { preferencesData = data }
        }
    }

    var body: some View {
        Form {
            Section {
                Toggle("Enabled", isOn: preference.enabled)
                if module.hasConfirmationSetting {
                    Picker("Confirmation", selection: preference.confirmation) {
                        Text("Use default").tag(ConfirmationPreference.useDefault)
                        Text("Ask first").tag(ConfirmationPreference.askFirst)
                        Text("Run directly").tag(ConfirmationPreference.runDirectly)
                    }
                }
            } footer: {
                if module.hasConfirmationSetting {
                    Text(module.requiresConfirmation ? "This action asks first by default." : "This action runs directly by default.")
                }
            }

            if let permissions = module.permissions, !permissions.isEmpty {
                Section("Access") {
                    ForEach(permissions, id: \.rawValue) { permission in
                        LabeledContent(permission.title, value: permission.accessLabel)
                    }
                }
            }

            Section {
                if module.id == "add_calendar_event", PhoneCalendar.authorized {
                    Picker("Default calendar", selection: $calendar) {
                        Text("System default").tag("")
                        ForEach(PhoneCalendar.writableCalendars) { list in Text(list.title).tag(list.id) }
                    }
                }
                if module.id == "add_reminder", Reminders.authorized {
                    Picker("Default list", selection: $reminderList) {
                        Text("System default").tag("")
                        ForEach(Reminders.writableLists) { list in Text(list.title).tag(list.id) }
                    }
                }
                if module.id == "request_lyft" {
                    TextField("Shortcut name", text: $lyftShortcut)
                        .autocorrectionDisabled()
                }
                Text(module.description).foregroundStyle(.secondary)
            }
        }
        .navigationTitle(module.displayTitle)
        .navigationBarTitleDisplayMode(.inline)
    }
}

private extension ModulePolicy {
    init(data: Data) {
        self.init(preferences: (try? JSONDecoder().decode([String: ModulePreference].self, from: data)) ?? [:])
    }
}

private extension Module {
    var displayTitle: String { id.replacingOccurrences(of: "_", with: " ").capitalized }
}

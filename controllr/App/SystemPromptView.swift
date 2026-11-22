import ControllrCore
import SwiftUI

struct SystemPromptView: View {
    @State private var settings = Settings.routerSettings

    var body: some View {
        Form {
            Section {
                NavigationLink {
                    PromptInstructionsEditor(instructions: settings.systemInstructions ?? Prompt.defaultInstructions) { text in
                        Settings.setSystemInstructions(text)
                        settings = Settings.routerSettings
                    }
                } label: {
                    LabeledContent("Edit Instructions", value: settings.systemInstructions == nil ? "Default" : "Custom")
                }
            }

            Section {
                Text(Prompt.system(settings: settings))
                    .font(.footnote.monospaced())
                    .textSelection(.enabled)
            } header: {
                Text("Effective System Prompt")
            } footer: {
                Text("Controllr adds the current time, locale, and configured personal context to each request.")
            }
        }
        .navigationTitle("System Prompt")
        .onAppear { settings = Settings.routerSettings }
    }
}

private struct PromptInstructionsEditor: View {
    @Environment(\.dismiss) private var dismiss
    @State var instructions: String
    let save: (String) -> Void

    var body: some View {
        Form {
            Section {
                TextEditor(text: $instructions)
                    .font(.body.monospaced())
                    .frame(minHeight: 300)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .accessibilityLabel("System instructions")
            } footer: {
                Text("Saved instructions apply to the next request across all providers. A blank edit restores the defaults.")
            }

            Button("Use Default Instructions") { instructions = Prompt.defaultInstructions }
                .disabled(instructions == Prompt.defaultInstructions)
        }
        .navigationTitle("Edit Instructions")
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Save") {
                    save(instructions)
                    dismiss()
                }
            }
        }
    }
}

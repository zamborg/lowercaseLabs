import SwiftUI
import ControllrCore

struct ModulesView: View {
    var body: some View {
        NavigationStack {
            List(Registry.v1.modules.filter { $0.kind != .response }, id: \.id) { m in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(m.id).font(.headline.monospaced())
                        Spacer()
                        if m.requiresConfirmation { Label("confirms", systemImage: "checkmark.shield").font(.caption).foregroundStyle(.orange) }
                        Text(kind(m)).font(.caption).padding(.horizontal, 6).padding(.vertical, 2).background(.quaternary, in: Capsule())
                    }
                    Text(m.description).font(.subheadline)
                    if !m.params.isEmpty {
                        Text(m.params.map { "\($0.name)\($0.required ? "" : "?")" }.joined(separator: ", "))
                            .font(.caption.monospaced()).foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Modules")
        }
    }
    private func kind(_ m: Module) -> String {
        switch m.kind {
        case .shortcut: "router"
        case .native: "app"
        case .builtin: "built-in"
        case .response: "reply"
        }
    }
}

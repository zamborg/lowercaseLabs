import SwiftUI
import ControllrCore

/// Runs the same Engine as the Decide intent, but never executes native modules or shortcut modules.
struct TryItView: View {
    @State private var text = ""
    @State private var output = ""
    @State private var running = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Say it in text") {
                    TextField("e.g. text Sam I'm running late", text: $text, axis: .vertical)
                    Button(running ? "Routing…" : "Route") { Task { await run() } }.disabled(text.isEmpty || running)
                }
                if !output.isEmpty {
                    Section("Decision") {
                        Text(output).font(.footnote.monospaced()).textSelection(.enabled)
                        Button("Copy") { UIPasteboard.general.string = output }
                    }
                }
            }
            .navigationTitle("Try it")
        }
    }

    private func run() async {
        running = true
        let r = await Engine.run(text, source: "try", execute: false)
        output = "module: \(r.toolCall?.name ?? "-")\nlatency: \(r.latencyMs) ms (\(r.provider)/\(r.model))\n\(r.error.map { "error: \($0)\n" } ?? "")\n\(r.decision.json)"
        running = false
    }
}

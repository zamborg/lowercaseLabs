import SwiftUI
import ParlorCore

/// Models, and a tab for every noun: hosts, engines, harnesses, tools, agents.
struct LibraryView: View {
    enum Tab: String, CaseIterable, Identifiable { case models = "Models", hosts = "Hosts", engines = "Engines", harnesses = "Harnesses", tools = "Tools", agents = "Agents"; var id: String { rawValue } }
    @State private var tab: Tab = .models

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $tab) { ForEach(Tab.allCases) { Text($0.rawValue).tag($0) } }
                .labelsHidden().pickerStyle(.segmented).frame(width: 520).padding(.vertical, 10)
                .accessibilityIdentifier("library-tabs")
            Divider()
            switch tab {
            case .models: ModelsView()
            case .hosts: NounLibraryView(noun: "host")
            case .engines: NounLibraryView(noun: "engine")
            case .harnesses: NounLibraryView(noun: "harness")
            case .tools: NounLibraryView(noun: "tool")
            case .agents: NounLibraryView(noun: "agent")
            }
        }
        .navigationTitle("Library")
    }
}

struct AddHostSheet: View {
    @Environment(ParlorEngine.self) private var engine
    @Environment(\.dismiss) private var dismiss
    @State private var id = ""
    @State private var url = "http://127.0.0.1:1234/v1"
    @State private var models = ""
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add a remote host").font(.title3.bold())
            Text("Any OpenAI-compatible server: LM Studio, Ollama, a llama.cpp server or a hosted API.").font(.caption).foregroundStyle(.secondary)
            TextField("Name (for example lmstudio)", text: $id).textFieldStyle(.roundedBorder).accessibilityIdentifier("host-id")
            TextField("URL", text: $url).textFieldStyle(.roundedBorder).accessibilityIdentifier("host-url")
            TextField("Models, comma separated (blank asks the server)", text: $models).textFieldStyle(.roundedBorder)
            if let error { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Add") {
                    Task {
                        var draft: [String: JSON] = ["id": .string(id), "engine": .string("openai-remote"), "url": .string(url)]
                        let ms = models.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                        if !ms.isEmpty { draft["models"] = .strings(ms) }
                        let r = await runVerb(engine, "hosts.create", ["draft": .object(draft)])
                        if r.isError { error = r.text } else { dismiss() }
                    }
                }
                .buttonStyle(.borderedProminent).disabled(id.isEmpty).accessibilityIdentifier("create-host")
            }
        }
        .padding(22).frame(width: 440)
    }
}

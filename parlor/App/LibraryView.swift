import SwiftUI
import ParlorCore

/// Runs a verb the way the UI does, so every button here is a verb anyone can call.
@MainActor
func runVerb(_ engine: ParlorEngine, _ verb: String, _ args: [String: JSON] = [:]) async -> ToolOutput {
    await engine.commands.call(verb, .object(args), caller: .ui)
}

/// Models, Hosts and Engines in one place.
struct LibraryView: View {
    enum Tab: String, CaseIterable, Identifiable { case models = "Models", hosts = "Hosts", engines = "Engines"; var id: String { rawValue } }
    @State private var tab: Tab = .models

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $tab) { ForEach(Tab.allCases) { Text($0.rawValue).tag($0) } }
                .labelsHidden().pickerStyle(.segmented).frame(width: 280).padding(.vertical, 10)
                .accessibilityIdentifier("library-tabs")
            Divider()
            switch tab {
            case .models: ModelsView()
            case .hosts: HostsTab()
            case .engines: EnginesTab()
            }
        }
        .navigationTitle("Library")
    }
}

struct HostsTab: View {
    @Environment(ParlorEngine.self) private var engine
    @State private var results: [String: String] = [:]
    @State private var adding = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Hosts").font(.headline)
                    Spacer()
                    Button("Add Remote Host…") { adding = true }.accessibilityIdentifier("add-host")
                }
                Text("A host is an engine serving a model at the gateway. Installed models get an automatic host; make your own to point at LM Studio, Ollama or a hosted API, or to run a model with different settings.")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(engine.catalog.hosts) { h in row(h) }
            }
            .padding(24).frame(maxWidth: 900, alignment: .leading)
        }
        .sheet(isPresented: $adding) { AddHostSheet() }
    }

    private func row(_ h: HostInfo) -> some View {
        let cat = engine.catalog
        let st = cat.status(host: h.id)
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Circle().fill(st.usable ? Color.green : Color.orange).frame(width: 9, height: 9)
                Text(h.id).font(.headline)
                Text(h.spec.engine).font(.caption).padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule())
                if h.isAuto { Text("auto").font(.caption).padding(.horizontal, 7).padding(.vertical, 1).background(.blue.opacity(0.2), in: Capsule()) }
                Text(st.label).font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Start") { Task { _ = await runVerb(engine, "hosts.start", ["id": .string(h.id)]) } }.controlSize(.small)
                Button("Stop") { Task { _ = await runVerb(engine, "hosts.stop", ["id": .string(h.id)]) } }.controlSize(.small)
                Button("Test") { Task { results[h.id] = "Testing…"; results[h.id] = (await runVerb(engine, "hosts.test", ["id": .string(h.id)])).text } }
                    .controlSize(.small).accessibilityIdentifier("test-host-\(h.id)")
                Button("Duplicate") { Task { _ = await runVerb(engine, "hosts.duplicate", ["id": .string(h.id)]) } }.controlSize(.small)
                if !h.isAuto {
                    Toggle("", isOn: Binding(get: { h.spec.enabled }, set: { v in Task { _ = await runVerb(engine, "hosts.enable", ["id": .string(h.id), "enabled": .bool(v)]) } }))
                        .labelsHidden().toggleStyle(.switch).controlSize(.small)
                    Button(role: .destructive) { Task { _ = await runVerb(engine, "hosts.delete", ["id": .string(h.id)]) } } label: { Image(systemName: "trash") }
                        .controlSize(.small)
                }
            }
            Text([h.spec.model, h.spec.url].compactMap { $0 }.joined(separator: " · ")).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            if let r = results[h.id] { Text(r).font(.caption).foregroundStyle(.secondary).textSelection(.enabled).lineLimit(6) }
        }
        .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .contain).accessibilityIdentifier("host-\(h.id)")
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

struct EnginesTab: View {
    @Environment(ParlorEngine.self) private var engine

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("Engines").font(.headline)
                Text("Backends that serve models. Add one by dropping a folder with an engine.json into engines/ in your parlor folder; engines/CLAUDE.md explains the format.")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(engine.catalog.engines, id: \.id) { e in
                    let cat = engine.catalog
                    let found = e.remote || e.requires?.binary == nil || cat.binary(for: e) != nil
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Circle().fill(found ? Color.green : Color.orange).frame(width: 9, height: 9)
                            Text(e.title).font(.headline)
                            ForEach(e.capabilities, id: \.self) { Text($0).font(.caption).padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule()) }
                            Spacer()
                            Text("\(cat.hosts.filter { $0.spec.engine == e.id }.count) hosts").font(.caption).foregroundStyle(.secondary)
                        }
                        Text("\(e.adapter) · formats: \(e.formats.isEmpty ? "none" : e.formats.joined(separator: ", "))").font(.caption).foregroundStyle(.secondary)
                        if !found, let b = e.requires?.binary {
                            Text("\(b) not found. \(e.requires?.hint ?? "")").font(.caption).foregroundStyle(.orange)
                        }
                    }
                    .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
                    .accessibilityElement(children: .contain).accessibilityIdentifier("engine-\(e.id)")
                }
            }
            .padding(24).frame(maxWidth: 900, alignment: .leading)
        }
    }
}

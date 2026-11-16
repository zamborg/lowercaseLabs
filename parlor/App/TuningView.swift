import SwiftUI
import ParlorCore

/// The superadmin pane: machine knobs on top, hosts on the left, the selected host's engine settings, request
/// defaults, live numbers and benchmark on the right. Every control is a verb.
struct TuningView: View {
    @Environment(ParlorEngine.self) private var engine
    @State private var selected: String?

    private var hosts: [HostInfo] { engine.catalog.hosts }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView { ConsoleSection().padding(24).frame(maxWidth: 900, alignment: .leading) }
                .frame(maxHeight: 280)
            Divider()
            HSplitView {
                hostList.frame(minWidth: 200, idealWidth: 230, maxWidth: 300)
                Group {
                    if let id = selected, let h = engine.catalog.host(id) { HostTuning(host: h).id(id) }
                    else { Text("Pick a host to tune it.").foregroundStyle(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity) }
                }
            }
        }
        .navigationTitle("Tuning")
        .accessibilityIdentifier("tuning-view")
        .onAppear { if selected == nil { selected = hosts.first?.id } }
    }

    private var hostList: some View {
        List(selection: $selected) {
            ForEach([("Text", "chat"), ("Image", "image")], id: \.1) { title, cap in
                let group = engine.catalog.hosts(capability: cap)
                if !group.isEmpty {
                    Section(title) {
                        ForEach(group) { h in
                            HStack(spacing: 6) {
                                Circle().fill(engine.catalog.status(host: h.id).usable ? Color.green : Color.orange).frame(width: 8, height: 8)
                                Text(h.id).lineLimit(1)
                                if h.isAuto { Text("auto").font(.caption2).foregroundStyle(.secondary) }
                            }
                            .tag(h.id)
                            .accessibilityIdentifier("tune-\(h.id)")
                        }
                    }
                }
            }
        }
    }
}

struct HostTuning: View {
    @Environment(ParlorEngine.self) private var engine
    let host: HostInfo
    @State private var settings: [String: JSON] = [:]
    @State private var defaults: [String: JSON] = [:]
    @State private var message: String?
    @State private var benching = false
    @State private var runs: [JSON] = []
    @State private var stats = ""

    private static let defaultsSchema: JSON = {
        (try? JSONDecoder().decode(JSON.self, from: Data("""
        { "type": "object", "properties": {
            "temperature": { "type": "number", "minimum": 0, "maximum": 2 },
            "top_p": { "type": "number", "minimum": 0, "maximum": 1 },
            "min_p": { "type": "number", "minimum": 0, "maximum": 1 },
            "repetition_penalty": { "type": "number", "minimum": 0.5, "maximum": 2 },
            "max_tokens": { "type": "integer", "minimum": 1 } } }
        """.utf8))) ?? .null
    }()

    private var engineManifest: EngineManifest? { engine.catalog.engine(host.spec.engine) }
    private var isChat: Bool { engine.catalog.capabilities(of: host).contains("chat") }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Text(host.id).font(.title3.bold())
                    Text(engineManifest?.title ?? host.spec.engine).foregroundStyle(.secondary)
                    Spacer()
                    Button("Duplicate") { Task { _ = await runVerb(engine, "hosts.duplicate", ["id": .string(host.id)]) } }
                        .help("Copy this host to try other settings on the same model").accessibilityIdentifier("tune-duplicate")
                }
                if host.isAuto {
                    Text("This host is computed from the installed model. Duplicate it to change its settings.").font(.callout).foregroundStyle(.secondary)
                }
                if let schema = engineManifest?.settings, !(schema.objectValue?["properties"]?.objectValue ?? [:]).isEmpty {
                    GroupBox("Engine settings") { Form { SchemaForm(schema: schema, values: $settings) }.formStyle(.grouped).disabled(host.isAuto) }
                }
                if isChat {
                    GroupBox("Request defaults") {
                        VStack(alignment: .leading) {
                            Form { SchemaForm(schema: Self.defaultsSchema, values: $defaults) }.formStyle(.grouped).disabled(host.isAuto)
                            Text("Used when an agent's config is silent. Order: agent config, then these, then the engine's own.").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                if !host.isAuto {
                    HStack {
                        Button("Save") { Task { await save() } }.buttonStyle(.borderedProminent).accessibilityIdentifier("tune-save")
                        if engine.catalog.status(host: host.id) == .running {
                            Button("Restart now") { Task { _ = await runVerb(engine, "hosts.stop", ["id": .string(host.id)]); message = "Stopped. It starts with the new settings on the next request." } }
                        }
                        if let message { Text(message).font(.caption).foregroundStyle(.secondary) }
                    }
                }
                GroupBox("Live") { Text(stats.isEmpty ? "No numbers yet." : stats).font(.system(.caption, design: .monospaced)).frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled) }
                GroupBox("Benchmark") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Button(benching ? "Running…" : "Run benchmark") { Task { await bench() } }.disabled(benching).accessibilityIdentifier("tune-bench")
                            Text(isChat ? "Three short chat turns, so cache reuse shows." : "One fixed prompt at fast and detailed size.").font(.caption).foregroundStyle(.secondary)
                        }
                        benchTable
                    }
                }
            }
            .padding(24)
        }
        .onAppear { load() }
    }

    private var benchTable: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(runs.suffix(4).enumerated()), id: \.offset) { i, r in
                let o = r.objectValue ?? [:]
                HStack(alignment: .firstTextBaseline) {
                    Text(i == runs.suffix(4).count - 1 ? "latest" : "earlier").font(.caption).foregroundStyle(.secondary).frame(width: 52, alignment: .leading)
                    Text(numbers(o)).font(.system(.caption, design: .monospaced))
                    Spacer()
                    Text(summary(o["settings"])).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary).lineLimit(1)
                }
            }
            if runs.count >= 2, let d = delta() { Text(d).font(.caption.bold()).accessibilityIdentifier("bench-delta") }
        }
    }

    private func numbers(_ o: [String: JSON]) -> String {
        if let t = o["tokens_per_second_last"]?.doubleValue { return String(format: "%.1f tok/s · turns %@ s", t, (o["seconds_per_turn"]?.arrayValue ?? []).compactMap { $0.doubleValue.map { String(format: "%.1f", $0) } }.joined(separator: "/")) }
        if let im = o["seconds_per_image"]?.objectValue { return "fast \(im["fast"]?.doubleValue ?? 0) s · detailed \(im["detailed"]?.doubleValue ?? 0) s" }
        return "-"
    }

    private func summary(_ s: JSON?) -> String { (s?.objectValue ?? [:]).sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value.text)" }.joined(separator: " ") }

    private func delta() -> String? {
        let a = runs[runs.count - 2].objectValue ?? [:], b = runs[runs.count - 1].objectValue ?? [:]
        if let x = a["tokens_per_second_last"]?.doubleValue, let y = b["tokens_per_second_last"]?.doubleValue { return String(format: "%.1f → %.1f tok/s (%+.0f%%)", x, y, x > 0 ? (y - x) / x * 100 : 0) }
        if let x = a["seconds_per_image"]?.objectValue?["fast"]?.doubleValue, let y = b["seconds_per_image"]?.objectValue?["fast"]?.doubleValue { return String(format: "fast image %.1f → %.1f s", x, y) }
        return nil
    }

    private func load() {
        settings = host.spec.settings; defaults = host.spec.defaults
        Task { await refresh() }
    }

    private func refresh() async {
        let r = await runVerb(engine, "hosts.stats", ["id": .string(host.id)])
        if let o = try? JSONDecoder().decode(JSON.self, from: Data(r.text.utf8)).objectValue {
            runs = o["bench"]?.arrayValue ?? []
            stats = "status: \(o["status"]?.stringValue ?? "?")" + (o["memory_bytes"]?.intValue.map { " · memory " + ByteCountFormatter.string(fromByteCount: Int64($0), countStyle: .memory) } ?? "")
        }
    }

    /// A merge patch that replaces `new` for `key`: removed keys become null.
    private func patch(old: [String: JSON], new: [String: JSON]) -> JSON {
        var o: [String: JSON] = new
        for k in old.keys where new[k] == nil { o[k] = .null }
        return .object(o)
    }

    private func save() async {
        let p: JSON = .object(["settings": patch(old: host.spec.settings, new: settings), "defaults": patch(old: host.spec.defaults, new: defaults)])
        let r = await runVerb(engine, "hosts.update", ["id": .string(host.id), "patch": p])
        message = r.isError ? r.text : "Saved. Settings apply the next time the host starts."
    }

    private func bench() async {
        benching = true; defer { benching = false }
        let r = await runVerb(engine, "hosts.bench", ["id": .string(host.id)])
        if r.isError { message = r.text }
        await refresh()
    }
}

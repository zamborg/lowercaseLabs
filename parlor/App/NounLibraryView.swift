import SwiftUI
import ParlorCore

/// One screen for every noun: its entries (loaded or broken, with their issues), the file in full, and Validate / Save,
/// Duplicate, Delete. Everything it does is a verb (`<plural>.list`, `.read`, `.validate`, `.update`, `.duplicate`, `.delete`).
struct NounLibraryView: View {
    @Environment(ParlorEngine.self) private var engine
    let noun: String
    @State private var rows: [[String: JSON]] = []
    @State private var selected: String?
    @State private var original: JSON = .null
    @State private var text = ""
    @State private var report: String?
    @State private var adding = false

    private static let plurals = ["host": "hosts", "engine": "engines", "harness": "harnesses", "tool": "tools", "agent": "agents"]
    private var plural: String { Self.plurals[noun] ?? noun + "s" }

    /// Changes whenever any noun list changes, so the rows reload when a file appears on disk.
    private var refreshKey: String {
        let cat = engine.catalog
        return [cat.hosts.map(\.id), cat.engines.map(\.id), engine.harnessList.map(\.id), engine.toolList.map(\.id), engine.agents.map(\.id)]
            .map { $0.joined(separator: ",") }.joined(separator: "|")
    }

    var body: some View {
        HSplitView {
            list.frame(minWidth: 260, idealWidth: 300, maxWidth: 380)
            detail.frame(minWidth: 380)
        }
        .task(id: refreshKey) { await reload() }
        .sheet(isPresented: $adding) { AddHostSheet() }
        .accessibilityIdentifier("noun-library-\(noun)")
    }

    private var list: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(plural.capitalized).font(.headline)
                Spacer()
                if noun == "host" { Button("Add Remote Host…") { adding = true }.controlSize(.small).accessibilityIdentifier("add-host") }
                Button { Task { await reload() } } label: { Image(systemName: "arrow.clockwise") }.buttonStyle(.borderless).help("Reload")
            }.padding(12)
            Divider()
            List(selection: Binding(get: { selected }, set: { id in selected = id; Task { await open(id) } })) {
                ForEach(rows.indices, id: \.self) { i in
                    let r = rows[i]; let id = r["id"]?.stringValue ?? "?"
                    HStack(spacing: 8) {
                        Circle().fill(color(r)).frame(width: 9, height: 9)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(id).font(.body)
                            Text(subtitle(r)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                        }
                    }
                    .tag(id)
                    .accessibilityIdentifier("\(noun)-\(id)")
                }
            }
        }
    }

    private func color(_ r: [String: JSON]) -> Color {
        if r["ok"]?.boolValue == false { return .red }
        if let s = r["state"]?.stringValue, s != "ready", s != "running" { return .orange }
        return .green
    }

    private func subtitle(_ r: [String: JSON]) -> String {
        var parts: [String] = []
        if r["managed"] != nil { parts.append("built in") }
        if r["ok"]?.boolValue == false { parts.append("\((r["issues"]?.arrayValue ?? []).count) issue(s)") }
        if let s = r["state"]?.stringValue { parts.append(s) }
        if let e = r["engine"]?.stringValue { parts.append(e) }
        if parts.isEmpty, let t = r["title"]?.stringValue ?? r["name"]?.stringValue { parts.append(t) }
        return parts.joined(separator: " · ")
    }

    private var isBuiltIn: Bool { original.objectValue?["managed"] != nil }

    @ViewBuilder private var detail: some View {
        if let id = selected {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text(id).font(.title3.bold())
                    if isBuiltIn { Text("built in, duplicate to change").font(.caption).foregroundStyle(.secondary) }
                    Spacer()
                    actions(id)
                }
                TextEditor(text: $text).font(.system(.body, design: .monospaced)).disabled(isBuiltIn)
                    .padding(6).background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
                    .accessibilityIdentifier("noun-editor")
                if let report { Text(report).font(.caption).textSelection(.enabled).foregroundStyle(report.hasPrefix("Saved") || report.hasPrefix("No problems") ? Color.secondary : Color.red) }
                HStack {
                    Button("Validate") { Task { await validate() } }.disabled(isBuiltIn)
                    Button("Save") { Task { await save(id) } }.buttonStyle(.borderedProminent).disabled(isBuiltIn).accessibilityIdentifier("noun-save")
                    Spacer()
                    Button("Reveal in Finder") { reveal(id) }
                }
            }.padding(16)
        } else {
            ContentUnavailableView("Pick one", systemImage: "list.bullet", description: Text("Entries that do not load are listed too, with their problems."))
        }
    }

    @ViewBuilder private func actions(_ id: String) -> some View {
        if noun == "host" {
            Button("Start") { Task { _ = await runVerb(engine, "hosts.start", ["id": .string(id)]); await reload() } }.controlSize(.small)
            Button("Stop") { Task { _ = await runVerb(engine, "hosts.stop", ["id": .string(id)]); await reload() } }.controlSize(.small)
            Button("Test") { Task { report = "Testing…"; report = (await runVerb(engine, "hosts.test", ["id": .string(id)])).text } }
                .controlSize(.small).accessibilityIdentifier("test-host-\(id)")
        }
        if noun == "harness" {
            Button("Try") { Task { await tryHarness(id) } }.controlSize(.small).accessibilityIdentifier("try-harness-\(id)")
        }
        if noun != "tool" {
            Button("Duplicate") { Task { await duplicate(id) } }.controlSize(.small).accessibilityIdentifier("noun-duplicate")
        }
        if !isBuiltIn {
            Button(role: .destructive) { Task { await delete(id) } } label: { Image(systemName: "trash") }.controlSize(.small).accessibilityIdentifier("noun-delete")
        }
    }

    // MARK: verbs

    private func reload() async {
        if case .success(let j) = await runVerbJSON(engine, "\(plural).list") {
            rows = (j.arrayValue ?? []).compactMap(\.objectValue).sorted { ($0["id"]?.stringValue ?? "") < ($1["id"]?.stringValue ?? "") }
        }
        if let s = selected { await open(s) }
    }

    private func open(_ id: String?) async {
        guard let id else { return }
        report = nil
        if case .success(let j) = await runVerbJSON(engine, "\(plural).read", ["id": .string(id)]) { original = j; text = j.pretty }
        else if let r = rows.first(where: { $0["id"]?.stringValue == id }) {
            original = .object(r); text = "// This entry does not load.\n" + ((r["issues"]?.arrayValue ?? []).map { ($0.objectValue?["message"]?.stringValue ?? "") + " (\($0.objectValue?["path"]?.stringValue ?? ""))" }.joined(separator: "\n"))
        }
    }

    private func edited() -> JSON? { try? JSONDecoder().decode(JSON.self, from: Data(text.utf8)) }

    private func validate() async {
        guard let d = edited() else { report = "Not valid JSON."; return }
        let r = await runVerb(engine, "\(plural).validate", ["draft": d])
        if r.isError { report = r.text; return }
        let j = (try? JSONDecoder().decode(JSON.self, from: Data(r.text.utf8)))?.objectValue
        report = (j?["issues"]?.arrayValue ?? []).isEmpty ? "No problems." : r.text
    }

    private func save(_ id: String) async {
        guard let d = edited() else { report = "Not valid JSON."; return }
        guard let patch = mergePatch(from: original, to: d) else { report = "No changes."; return }
        let r = await runVerb(engine, "\(plural).update", ["id": .string(id), "patch": patch])
        if r.isError { report = r.text } else { report = "Saved."; await reload() }
    }

    /// One throwaway turn on this harness, using the first host that can chat.
    private func tryHarness(_ id: String) async {
        let cat = engine.catalog
        guard let host = cat.hosts(capability: "chat").first(where: { cat.status(host: $0.id).usable }) else { report = "No chat host is ready to try it on."; return }
        report = "Trying on \(host.id)…"
        let draft: [String: JSON] = ["id": .string("try-" + id), "name": .string("Try"), "harness": .string(id), "hosts": .object(["chat": .string(host.id)]),
                                     "persona": .string("You are a short, friendly test assistant."), "tools": .strings([])]
        let r = await runVerb(engine, "agents.try", ["draft": .object(draft), "message": .string("Say hello in five words.")])
        report = r.text
    }

    private func duplicate(_ id: String) async {
        var args: [String: JSON] = ["id": .string(id)]
        if noun != "agent" { args["new_id"] = .string("my-" + id) }
        let r = await runVerb(engine, "\(plural).duplicate", args)
        report = r.isError ? r.text : "Duplicated."
        await reload()
    }

    private func delete(_ id: String) async {
        let r = await runVerb(engine, "\(plural).delete", ["id": .string(id)])
        if r.isError { report = r.text } else { selected = nil }
        await reload()
    }

    private func reveal(_ id: String) {
        let h = engine.home
        let url: URL = switch noun {
        case "host": h.hostsDir.appendingPathComponent(id + ".json")
        case "engine": h.enginesDir.appendingPathComponent(id)
        case "harness": h.harnessesDir.appendingPathComponent(id)
        case "tool": h.toolsDir.appendingPathComponent(id)
        default: h.agentsDir.appendingPathComponent(id)
        }
        if FileManager.default.fileExists(atPath: url.path) { NSWorkspace.shared.activateFileViewerSelecting([url]) }
        else { NSWorkspace.shared.open(url.deletingLastPathComponent()) }
    }
}

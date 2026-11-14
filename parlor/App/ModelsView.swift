import SwiftUI
import ParlorCore

struct ModelsView: View {
    @Environment(ParlorEngine.self) private var engine
    @State private var kind: ModelKind = .language
    @State private var query = ""
    @State private var results: [ModelSearchResult] = []
    @State private var searching = false
    @State private var searchError: String?
    @State private var memory: (Int64, Int64) = (SystemMemory.usedBytes(), SystemMemory.totalBytes)

    private static let recommended: [(repo: String, kind: ModelKind, note: String, bytes: Int64)] = [
        ("mlx-community/Qwen3-4B-4bit", .language, "Fast, good at tools. A safe first choice.", 2_300_000_000),
        ("mlx-community/Llama-3.2-3B-Instruct-4bit", .language, "Small and quick. Good for characters.", 1_800_000_000),
        ("mlx-community/Qwen3-30B-A3B-4bit", .language, "Much smarter, still fast. Needs about 17 GB.", 17_400_000_000),
        ("mlx-community/Z-Image-Turbo-bf16", .image, "Image generation in a few steps.", 12_000_000_000),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                memoryBar
                ConsoleSection()
                installedSection
                if !engine.library.downloads.isEmpty { downloadsSection }
                recommendedSection
                searchSection
            }
            .padding(24)
            .frame(maxWidth: 900, alignment: .leading)
        }
        .navigationTitle("Models")
        .onAppear { engine.library.refresh() }
    }

    private var memoryBar: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Memory").font(.headline)
                Spacer()
                Text("\(memoryString(memory.0)) used of \(memoryString(memory.1))").foregroundStyle(.secondary).monospacedDigit()
            }
            ProgressView(value: Double(memory.0), total: Double(max(memory.1, 1)))
            Text("Models load when you chat and unload when you switch to an agent that uses a different one. Idle timeouts for both are under Performance.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .onReceive(Timer.publish(every: 2, on: .main, in: .common).autoconnect()) { _ in
            memory = (SystemMemory.usedBytes(), SystemMemory.totalBytes)
        }
    }

    private var installedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Installed").font(.headline)
            if engine.library.installed.isEmpty {
                Text("Nothing downloaded yet. Pick one below.").foregroundStyle(.secondary)
            } else {
                VStack(spacing: 0) {
                    ForEach(engine.library.installed) { m in
                        HStack {
                            Image(systemName: m.kind == .image ? "photo" : "text.bubble").frame(width: 22).foregroundStyle(.secondary)
                            VStack(alignment: .leading) {
                                Text(m.shortName).fontWeight(.medium)
                                Text(usedBy(m)).font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if engine.loadedModel == m.repo { Text("loaded").font(.caption).padding(.horizontal, 8).padding(.vertical, 2).background(.green.opacity(0.2), in: Capsule()) }
                            Text(byteString(m.bytes)).foregroundStyle(.secondary).monospacedDigit()
                            Button(role: .destructive) { engine.library.delete(m.repo) } label: { Image(systemName: "trash") }
                                .buttonStyle(.borderless).help("Delete from disk")
                        }
                        .padding(.vertical, 8).padding(.horizontal, 12)
                        .accessibilityIdentifier("installed-\(m.shortName)")
                        if m.id != engine.library.installed.last?.id { Divider() }
                    }
                }
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    private func usedBy(_ m: InstalledModel) -> String {
        let names = engine.agents.filter { $0.model == m.repo || $0.imageModel == m.repo }.map(\.name)
        return names.isEmpty ? "Not used by any agent" : "Used by " + names.joined(separator: ", ")
    }

    private var downloadsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Downloading").font(.headline)
            ForEach(Array(engine.library.downloads.values).sorted { $0.repo < $1.repo }) { d in
                HStack(spacing: 12) {
                    VStack(alignment: .leading) {
                        Text(shortModelName(d.repo)).fontWeight(.medium)
                        Text(d.repo).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    switch d.status {
                    case .running:
                        ProgressView(value: d.fraction).frame(width: 160)
                        Text("\(Int(d.fraction * 100))%" + (d.bytesPerSecond > 0 ? " · \(byteString(Int64(d.bytesPerSecond)))/s" : ""))
                            .font(.caption).monospacedDigit().foregroundStyle(.secondary).frame(width: 120, alignment: .trailing)
                        Button("Pause") { engine.library.pause(d.repo) }
                    case .paused:
                        Text("Paused").foregroundStyle(.secondary)
                        Button("Resume") { engine.library.download(d.repo, kind: d.kind, bytes: d.totalBytes) }
                    case .failed(let msg):
                        Text(msg).font(.caption).foregroundStyle(.red).lineLimit(2).frame(maxWidth: 240)
                        Button("Retry") { engine.library.download(d.repo, kind: d.kind, bytes: d.totalBytes) }
                    case .done: EmptyView()
                    }
                }
                .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
                .accessibilityIdentifier("download-\(shortModelName(d.repo))")
            }
        }
    }

    private var recommendedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recommended").font(.headline)
            VStack(spacing: 0) {
                ForEach(Self.recommended, id: \.repo) { r in
                    HStack {
                        Image(systemName: r.kind == .image ? "photo" : "text.bubble").frame(width: 22).foregroundStyle(.secondary)
                        VStack(alignment: .leading) {
                            Text(shortModelName(r.repo)).fontWeight(.medium)
                            Text(r.note).font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(engine.library.fit(bytes: r.bytes).label).font(.caption).foregroundStyle(.secondary)
                        Text("~" + byteString(r.bytes)).foregroundStyle(.secondary).monospacedDigit()
                        action(repo: r.repo, kind: r.kind, bytes: r.bytes)
                    }
                    .padding(.vertical, 8).padding(.horizontal, 12)
                    if r.repo != Self.recommended.last?.repo { Divider() }
                }
            }
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        }
    }

    @ViewBuilder private func action(repo: String, kind: ModelKind, bytes: Int64?) -> some View {
        if engine.library.installedDirectory(for: repo) != nil {
            Label("Installed", systemImage: "checkmark.circle.fill").foregroundStyle(.green).font(.caption)
        } else if engine.library.downloads[repo] != nil {
            Text("Downloading…").font(.caption).foregroundStyle(.secondary)
        } else {
            Button("Download") { engine.library.download(repo, kind: kind, bytes: bytes) }
                .accessibilityIdentifier("download-button-\(shortModelName(repo))")
        }
    }

    private var searchSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Search Hugging Face").font(.headline)
            HStack {
                Picker("Kind", selection: $kind) {
                    Text("Language").tag(ModelKind.language); Text("Image").tag(ModelKind.image)
                }.pickerStyle(.segmented).frame(width: 180).labelsHidden()
                TextField("Search mlx-community, or type org/name", text: $query)
                    .textFieldStyle(.roundedBorder).onSubmit(search)
                Button("Search", action: search).disabled(searching)
            }
            if searching { ProgressView().controlSize(.small) }
            if let e = searchError { Text(e).foregroundStyle(.red).font(.caption) }
            VStack(spacing: 0) {
                ForEach(results) { r in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(r.shortName).fontWeight(.medium)
                            Text("\(r.repo) · \(r.downloads.formatted()) downloads").font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(engine.library.fit(bytes: r.bytes).label).font(.caption).foregroundStyle(.secondary)
                        Text(r.bytes.map(byteString) ?? "?").foregroundStyle(.secondary).monospacedDigit()
                        action(repo: r.repo, kind: r.kind, bytes: r.bytes)
                    }
                    .padding(.vertical, 8).padding(.horizontal, 12)
                    Divider()
                }
            }
        }
    }

    private func search() {
        searching = true; searchError = nil
        Task {
            defer { searching = false }
            do { results = try await engine.library.search(query, kind: kind) }
            catch { searchError = error.localizedDescription }
        }
    }
}

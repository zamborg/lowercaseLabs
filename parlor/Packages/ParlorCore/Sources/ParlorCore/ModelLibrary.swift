import Foundation
import HuggingFace
import Observation

public enum ModelKind: String, Codable, Sendable, CaseIterable { case language, image }

public struct InstalledModel: Identifiable, Hashable, Sendable {
    public var repo: String
    public var kind: ModelKind
    public var bytes: Int64
    public var directory: URL
    public var id: String { repo }
    public var shortName: String { repo.split(separator: "/").last.map(String.init) ?? repo }
}

public struct ModelSearchResult: Identifiable, Hashable, Sendable {
    public var repo: String
    public var kind: ModelKind
    public var downloads: Int
    public var bytes: Int64?
    /// Why Parlor can't run this repo, or nil when it can (or when that is not known yet).
    public var unsupported: String? = nil
    public var id: String { repo }
    public var shortName: String { repo.split(separator: "/").last.map(String.init) ?? repo }
}

public enum ModelFit: Sendable {
    case roomy, tight, tooBig, unknown
    public var label: String {
        switch self {
        case .roomy: return "fits with room"
        case .tight: return "tight with other apps"
        case .tooBig: return "too big for this Mac"
        case .unknown: return "size unknown"
        }
    }
}

public struct DownloadState: Identifiable, Sendable {
    public enum Status: Sendable, Equatable { case running, paused, failed(String), done }
    public var repo: String
    public var kind: ModelKind
    public var fraction: Double
    public var bytesPerSecond: Double
    public var totalBytes: Int64?
    public var status: Status
    public var id: String { repo }
}

public enum SystemMemory {
    public static var totalBytes: Int64 { Int64(ProcessInfo.processInfo.physicalMemory) }

    /// Used = active + wired + compressed pages.
    public static func usedBytes() -> Int64 {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        let page = Int64(vm_kernel_page_size)
        return Int64(stats.active_count + stats.wire_count + stats.compressor_page_count) * page
    }
}

@Observable
public final class ModelLibrary: @unchecked Sendable {
    public let home: ParlorHome
    public private(set) var installed: [InstalledModel] = []
    public private(set) var downloads: [String: DownloadState] = [:]
    private var tasks: [String: Task<Void, Never>] = [:]

    public init(home: ParlorHome) {
        self.home = home
        refresh()
    }

    public func directory(for repo: String) -> URL {
        let parts = repo.split(separator: "/").map(String.init)
        var url = home.modelsDir
        for p in parts { url.appendPathComponent(p) }
        return url
    }

    private func marker(_ dir: URL) -> URL { dir.appendingPathComponent(".parlor-model.json") }

    /// Directory of a fully downloaded model, if present.
    public func installedDirectory(for repo: String) -> URL? {
        let dir = directory(for: repo)
        return FileManager.default.fileExists(atPath: marker(dir).path) && Self.hasWeights(dir) ? dir : nil
    }

    public func refresh() {
        let fm = FileManager.default
        var found: [InstalledModel] = []
        let orgs = (try? fm.contentsOfDirectory(at: home.modelsDir, includingPropertiesForKeys: nil)) ?? []
        for org in orgs where org.hasDirectoryPath {
            let names = (try? fm.contentsOfDirectory(at: org, includingPropertiesForKeys: nil)) ?? []
            for dir in names where fm.fileExists(atPath: marker(dir).path) && Self.hasWeights(dir) {
                guard let d = try? Data(contentsOf: marker(dir)),
                      let m = try? JSONDecoder().decode(Marker.self, from: d) else { continue }
                found.append(InstalledModel(repo: m.repo, kind: m.kind, bytes: Self.size(of: dir), directory: dir))
            }
        }
        installed = found.sorted { $0.repo < $1.repo }
    }

    struct Marker: Codable { var repo: String; var kind: ModelKind }

    /// A download that fetched only configs (for example a PyTorch-only repo) is not an installed model.
    static func hasWeights(_ dir: URL) -> Bool {
        guard let e = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil) else { return false }
        for case let u as URL in e where u.pathExtension == "safetensors" { return true }
        return false
    }

    static func size(of dir: URL) -> Int64 {
        var total: Int64 = 0
        if let e = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let u as URL in e { total += Int64((try? u.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
        }
        return total
    }

    public func installed(kind: ModelKind) -> [InstalledModel] { installed.filter { $0.kind == kind } }

    public func fit(bytes: Int64?) -> ModelFit {
        guard let bytes else { return .unknown }
        let ratio = Double(bytes) / Double(SystemMemory.totalBytes)
        if ratio < 0.4 { return .roomy }
        if ratio < 0.65 { return .tight }
        return .tooBig
    }

    // MARK: search

    public func search(_ query: String, kind: ModelKind, limit: Int = 20) async throws -> [ModelSearchResult] {
        var comps = URLComponents(string: "https://huggingface.co/api/models")!
        var author = "mlx-community"
        var term = query.trimmingCharacters(in: .whitespaces)
        if let slash = term.firstIndex(of: "/") {
            author = String(term[..<slash]); term = String(term[term.index(after: slash)...])
        }
        var items: [URLQueryItem] = [
            .init(name: "author", value: author),
            .init(name: "pipeline_tag", value: kind == .image ? "text-to-image" : "text-generation"),
            .init(name: "sort", value: "downloads"), .init(name: "direction", value: "-1"),
            .init(name: "limit", value: String(limit)),
            .init(name: "expand[]", value: "downloads"),
        ]
        if !term.isEmpty { items.append(.init(name: "search", value: term)) }
        comps.queryItems = items
        let (data, resp) = try await URLSession.shared.data(from: comps.url!)
        if let http = resp as? HTTPURLResponse, http.statusCode != 200 {
            throw ParlorError.message("Hugging Face search failed (HTTP \(http.statusCode)): \(String(decoding: data.prefix(160), as: UTF8.self))")
        }
        struct Row: Decodable { var id: String; var downloads: Int? }
        let rows = try JSONDecoder().decode([Row].self, from: data)
        // The search API no longer reports sizes, so sum each repo's file tree (concurrently; a failure just leaves "?").
        let infos = await withTaskGroup(of: (String, RepoInfo?).self) { group -> [String: RepoInfo] in
            for r in rows { group.addTask { (r.id, await Self.repoInfo(r.id)) } }
            var out: [String: RepoInfo] = [:]
            for await (id, i) in group { if let i { out[id] = i } }
            return out
        }
        return rows.map { r in
            let info = infos[r.id]
            return ModelSearchResult(repo: r.id, kind: kind, downloads: r.downloads ?? 0, bytes: info?.bytes,
                                     unsupported: info.flatMap { Self.unsupportedReason($0, kind: kind, repo: r.id) })
        }
    }

    struct RepoInfo: Sendable { var bytes: Int64; var weightBytes: Int64; var files: [String] }

    /// Sizes and file names from the repo's tree. Nil when the network call fails.
    static func repoInfo(_ repo: String) async -> RepoInfo? {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/main?recursive=true"),
              let (data, resp) = try? await URLSession.shared.data(from: url),
              (resp as? HTTPURLResponse)?.statusCode == 200,
              let entries = try? JSONDecoder().decode([TreeEntry].self, from: data) else { return nil }
        let files = entries.filter { $0.type == "file" }
        return RepoInfo(
            bytes: files.reduce(0) { $0 + ($1.size ?? 0) },
            weightBytes: files.filter { $0.path.hasSuffix(".safetensors") }.reduce(0) { $0 + ($1.size ?? 0) },
            files: files.map(\.path))
    }
    private struct TreeEntry: Decodable { var type: String; var path: String; var size: Int64? }

    /// Parlor loads safetensors weights (MLX conversions); the image worker runs Z-Image only.
    static func unsupportedReason(_ i: RepoInfo, kind: ModelKind, repo: String) -> String? {
        if i.weightBytes == 0 {
            let hint = repo.hasPrefix("mlx-community/") ? "" : " Look for an MLX conversion under mlx-community."
            return "No safetensors weights (PyTorch or GGUF only)." + hint
        }
        if kind == .image, !repo.lowercased().contains("z-image") { return "The image worker runs Z-Image models only." }
        return nil
    }

    /// Fail early, before anything is written, when a repo can't run here.
    func preflight(_ repo: String, kind: ModelKind) async throws -> Int64? {
        guard let info = await Self.repoInfo(repo) else { return nil }   // check failed offline: let the download report it
        if let why = Self.unsupportedReason(info, kind: kind, repo: repo) { throw ParlorError.message("\(repo): \(why)") }
        return info.weightBytes
    }

    // MARK: downloads

    public func download(_ repo: String, kind: ModelKind, bytes: Int64? = nil) {
        guard tasks[repo] == nil else { return }
        downloads[repo] = DownloadState(repo: repo, kind: kind, fraction: 0, bytesPerSecond: 0, totalBytes: bytes, status: .running)
        let dir = directory(for: repo)
        tasks[repo] = Task { [weak self] in
            do {
                guard let id = Repo.ID(rawValue: repo) else { throw ParlorError.message("Bad repo id \(repo)") }
                _ = try await self?.preflight(repo, kind: kind)
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                // Explicit extensions: an empty glob list makes the hub client try to write directory entries as files.
                let globs = ["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken", "*.md"]
                let started = Date()
                _ = try await HubClient().downloadSnapshot(
                    of: id, to: dir, matching: globs, progressHandler: { progress in
                        let f = progress.fractionCompleted
                        let elapsed = max(Date().timeIntervalSince(started), 0.5)
                        self?.update(repo) {
                            $0.fraction = f
                            if let total = $0.totalBytes, total > 0 { $0.bytesPerSecond = f * Double(total) / elapsed }
                        }
                    })
                try JSONEncoder().encode(Marker(repo: repo, kind: kind)).write(to: dir.appendingPathComponent(".parlor-model.json"))
                self?.finish(repo, nil)
            } catch is CancellationError {
                self?.finish(repo, .paused)
            } catch {
                if Task.isCancelled { self?.finish(repo, .paused) }
                else { self?.finish(repo, .failed(error.localizedDescription)) }
            }
        }
    }

    public func pause(_ repo: String) { tasks[repo]?.cancel() }

    /// Await a download to completion (used by the CLI).
    public func waitForDownload(_ repo: String) async {
        await tasks[repo]?.value
    }

    public func delete(_ repo: String) {
        try? FileManager.default.removeItem(at: directory(for: repo))
        refresh()
    }

    private func update(_ repo: String, _ f: (inout DownloadState) -> Void) {
        guard var s = downloads[repo] else { return }
        f(&s)
        downloads[repo] = s
    }

    private func finish(_ repo: String, _ status: DownloadState.Status?) {
        tasks[repo] = nil
        if let status { update(repo) { $0.status = status } }
        else { downloads[repo] = nil; refresh() }
    }
}

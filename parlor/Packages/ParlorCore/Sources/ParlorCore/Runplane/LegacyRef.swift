import Foundation

/// Maps a V2 agent's `model:` string onto a host. Temporary: deleted when agents become JSON in step 4.
enum LegacyRef {
    /// `repo` → that model's host; `host/model` → that host; `mlx/<repo>` → the repo's host.
    static func host(for name: String, catalog: Catalog) -> HostInfo? {
        let repo = name.hasPrefix("mlx/") ? String(name.dropFirst(4)) : name
        if let h = catalog.host(forModel: repo) { return h }
        if let slash = name.firstIndex(of: "/"), let h = catalog.host(String(name[..<slash])) { return h }
        return catalog.host(name)
    }
}

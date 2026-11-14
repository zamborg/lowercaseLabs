import AppKit

struct AppChoice {
    let name: String
    let bundleID: String
}

enum AppCatalog {
    static func choices(including selected: Set<String>) -> [AppChoice] {
        var choices: [String: AppChoice] = [:]
        func add(_ url: URL) {
            guard let bundle = Bundle(url: url), let id = bundle.bundleIdentifier,
                  id != Bundle.main.bundleIdentifier else { return }
            let name = (bundle.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String)
                ?? (bundle.object(forInfoDictionaryKey: "CFBundleName") as? String)
                ?? url.deletingPathExtension().lastPathComponent
            choices[id] = AppChoice(name: name, bundleID: id)
        }
        for app in NSWorkspace.shared.runningApplications where app.activationPolicy == .regular {
            if let url = app.bundleURL { add(url) }
        }
        // Include installed apps even when they aren't running. Don't traverse app packages.
        let roots = [URL(fileURLWithPath: "/Applications"), URL(fileURLWithPath: "/System/Applications"),
                     FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications")]
        for root in roots {
            guard let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else { continue }
            var visited = 0
            while let url = files.nextObject() as? URL {
                visited += 1
                if visited > 4000 { break }
                if url.pathExtension == "app" { add(url); files.skipDescendants() }
                else if files.level > 3 { files.skipDescendants() }
            }
        }
        for id in selected where choices[id] == nil {
            if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: id) { add(url) }
            else { choices[id] = AppChoice(name: id + " (not installed)", bundleID: id) }
        }
        return choices.values.sorted {
            let order = $0.name.localizedCaseInsensitiveCompare($1.name)
            return order == .orderedSame ? $0.bundleID < $1.bundleID : order == .orderedAscending
        }
    }
}

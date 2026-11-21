import Foundation

public struct Registry: Sendable {
    public let modules: [Module]
    public init(modules: [Module]) { self.modules = modules }

    public func module(id: String) -> Module? { modules.first { $0.id == id } }

    public static let v1: Registry = {
        let files = Bundle.module.urls(forResourcesWithExtension: "json", subdirectory: "Actions") ?? []
        do {
            let modules = try files.sorted { $0.lastPathComponent < $1.lastPathComponent }.map { file in
                let module = try JSONDecoder().decode(Module.self, from: Data(contentsOf: file))
                precondition(file.deletingPathExtension().lastPathComponent == module.id, "Manifest filename must match the action id")
                precondition((module.kind == .shortcut) == (module.shortcutHandler != nil), "Missing or unexpected shortcut handler")
                return module
            }
            precondition(!modules.isEmpty, "App bundle is missing its action manifests")
            return Registry(modules: modules)
        } catch {
            fatalError("Invalid bundled action manifest: \(error)")
        }
    }()
}

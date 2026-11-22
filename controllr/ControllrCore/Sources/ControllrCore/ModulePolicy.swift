public enum ConfirmationPreference: String, Codable, CaseIterable, Sendable {
    case useDefault, askFirst, runDirectly
}

public struct ModulePreference: Codable, Equatable, Sendable {
    public var enabled: Bool
    public var confirmation: ConfirmationPreference

    public init(enabled: Bool = true, confirmation: ConfirmationPreference = .useDefault) {
        self.enabled = enabled
        self.confirmation = confirmation
    }
}

public struct ModulePolicy: Sendable {
    public var preferences: [String: ModulePreference]

    public init(preferences: [String: ModulePreference] = [:]) {
        self.preferences = preferences
    }

    public func isEnabled(_ module: Module) -> Bool { preferences[module.id]?.enabled ?? true }

    public func requiresConfirmation(_ module: Module) -> Bool {
        guard module.hasConfirmationSetting else { return false }
        switch preferences[module.id]?.confirmation ?? .useDefault {
        case .useDefault: return module.requiresConfirmation
        case .askFirst: return true
        case .runDirectly: return false
        }
    }
}

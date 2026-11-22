import ControllrCore
import Foundation

enum ModuleSettings {
    private static let key = "modulePreferences"

    static var policy: ModulePolicy {
        let preferences = UserDefaults.standard.data(forKey: key)
            .flatMap { try? JSONDecoder().decode([String: ModulePreference].self, from: $0) } ?? [:]
        return ModulePolicy(preferences: preferences)
    }
}

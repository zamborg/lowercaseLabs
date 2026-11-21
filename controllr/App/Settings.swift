import Foundation
import ControllrCore

/// Non-secret settings (API keys live in the Keychain).
enum Settings {
    private static var d: UserDefaults { .standard }

    static var provider: String {
        get { d.string(forKey: "provider") ?? "openai" }
        set { d.set(newValue, forKey: "provider") }
    }
    static var openaiModel: String {
        get { d.string(forKey: "openaiModel") ?? "" }
        set { d.set(newValue, forKey: "openaiModel") }
    }
    static var anthropicModel: String {
        get { d.string(forKey: "anthropicModel") ?? "claude-haiku-5-5" }
        set { d.set(newValue, forKey: "anthropicModel") }
    }
    static var homeAddress: String {
        get { d.string(forKey: "homeAddress") ?? "" }
        set { d.set(newValue, forKey: "homeAddress") }
    }
    /// One "nickname=Contact Name" per line.
    static var nicknamesText: String {
        get { d.string(forKey: "nicknames") ?? "" }
        set { d.set(newValue, forKey: "nicknames") }
    }

    static var nicknames: [String: String] {
        var out: [String: String] = [:]
        for line in nicknamesText.split(separator: "\n") {
            let parts = line.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
            if parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty { out[parts[0].lowercased()] = parts[1] }
        }
        return out
    }

    static var routerSettings: RouterSettings {
        RouterSettings(homeAddress: homeAddress, nicknames: nicknames)
    }

    static func model(for provider: String) -> String { provider == "anthropic" ? anthropicModel : openaiModel }
}

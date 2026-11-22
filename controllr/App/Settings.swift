import Foundation
import ControllrCore

/// Non-secret settings (API keys live in the Keychain).
enum Settings {
    private static var d: UserDefaults { .standard }

    static var provider: CompletionProvider {
        get { d.string(forKey: "provider").flatMap(CompletionProvider.init(rawValue:)) ?? .openai }
        set { d.set(newValue.rawValue, forKey: "provider") }
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
        RouterSettings(homeAddress: homeAddress, nicknames: nicknames, systemInstructions: systemInstructions())
    }

    static func systemInstructions(defaults: UserDefaults = .standard) -> String? {
        defaults.string(forKey: "systemInstructions")
    }

    static func setSystemInstructions(_ text: String, defaults: UserDefaults = .standard) {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || text == Prompt.defaultInstructions {
            defaults.removeObject(forKey: "systemInstructions")
        } else {
            defaults.set(text, forKey: "systemInstructions")
        }
    }

    static func model(for provider: CompletionProvider, defaults: UserDefaults = .standard) -> String {
        defaults.string(forKey: "\(provider.rawValue)Model") ?? provider.defaultModel ?? ""
    }

    static func setModel(_ model: String, for provider: CompletionProvider, defaults: UserDefaults = .standard) {
        defaults.set(model, forKey: "\(provider.rawValue)Model")
    }
}

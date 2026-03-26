import Foundation

struct UserDefaultsAuthSessionStore: AuthSessionStore {
    private enum Keys {
        static let session = "blackhole.auth.session"
        static let baseURL = "blackhole.api.base_url"
    }

    private let defaults: UserDefaults = .standard

    func loadSession() -> StoredAuthSession? {
        guard let data = defaults.data(forKey: Keys.session) else {
            return nil
        }
        return try? JSONDecoder.blackhole.decode(StoredAuthSession.self, from: data)
    }

    func saveSession(_ session: StoredAuthSession) {
        guard let data = try? JSONEncoder.blackhole.encode(session) else {
            return
        }
        defaults.set(data, forKey: Keys.session)
    }

    func clearSession() {
        defaults.removeObject(forKey: Keys.session)
    }

    func loadBaseURL() -> String? {
        defaults.string(forKey: Keys.baseURL)
    }

    func saveBaseURL(_ baseURL: String) {
        defaults.set(baseURL, forKey: Keys.baseURL)
    }
}


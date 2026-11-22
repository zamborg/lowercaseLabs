import Foundation
import Security

/// Minimal Keychain wrapper for API keys. Never logs values.
enum Keychain {
    private static let service = "com.lowercaselabs.controllr"

    static func set(_ value: String, for account: String) -> Bool {
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let valueData = Data(value.utf8)
        let update = SecItemUpdate(base as CFDictionary, [kSecValueData as String: valueData] as CFDictionary)
        guard update == errSecItemNotFound else { return update == errSecSuccess }
        var add = base
        add[kSecValueData as String] = valueData
        // Readable while locked-after-first-unlock so intents work from the Lock Screen.
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(add as CFDictionary, nil) == errSecSuccess
    }

    static func get(_ account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var out: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &out) == errSecSuccess,
              let data = out as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

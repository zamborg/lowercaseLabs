import CryptoKit
import Foundation

/// What a caller may do. `destructive` is never a tier: destructive verbs always ask.
public enum Tier: String, Codable, Sendable, CaseIterable, Comparable {
    case none, read, act, configure

    var rank: Int { switch self { case .none: 0; case .read: 1; case .act: 2; case .configure: 3 } }
    public static func < (a: Tier, b: Tier) -> Bool { a.rank < b.rank }
}

/// Who is calling a verb.
public struct Caller: Sendable {
    public enum Kind: Sendable { case ui, agent, token, local }
    public var kind: Kind
    public var name: String
    public var tier: Tier

    /// The person at the keyboard (the app or the CLI): everything, with no extra prompt.
    public static let local = Caller(kind: .local, name: "local", tier: .configure)
    public static let ui = Caller(kind: .ui, name: "app", tier: .configure)
    public static func agent(_ a: Agent) -> Caller { Caller(kind: .agent, name: a.id, tier: a.tier) }
    public static func token(_ name: String, _ tier: Tier) -> Caller { Caller(kind: .token, name: name, tier: tier) }

    /// A person is not watching a call made by an agent or a token, so destructive verbs ask.
    public var mustConfirmDestructive: Bool { kind == .agent || kind == .token }
}

public struct APIToken: Codable, Equatable, Sendable, Identifiable {
    public var name: String
    public var tier: Tier
    public var hash: String
    public var created: Date
    public var expires: Date?
    public var id: String { name }

    public var isExpired: Bool { expires.map { $0 < Date() } ?? false }

    public static func hash(_ secret: String) -> String {
        SHA256.hash(data: Data(secret.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    /// A fresh secret like `plt_9f2c…` and the record to store.
    public static func make(name: String, tier: Tier, expires: Date? = nil) -> (secret: String, token: APIToken) {
        let secret = "plt_" + (0..<32).map { _ in String(format: "%02x", UInt8.random(in: 0...255)) }.joined()
        return (secret, APIToken(name: name, tier: tier, hash: hash(secret), created: Date(), expires: expires))
    }
}

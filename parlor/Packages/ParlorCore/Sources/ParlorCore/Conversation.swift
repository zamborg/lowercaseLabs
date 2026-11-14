import Foundation

/// Someone in a conversation: an agent, or the user.
public struct Participant: Codable, Hashable, Sendable {
    public var agent: String?
    public var user: Bool?
    public static func agent(_ id: String) -> Participant { Participant(agent: id, user: nil) }
    public static let user = Participant(agent: nil, user: true)
    public var isUser: Bool { user == true }
}

public struct SchedulerSpec: Codable, Hashable, Sendable {
    /// `userDriven`, `roundRobin`, `mention` or `autonomous`. Only `userDriven` is implemented so far.
    public var kind: String
    public var maxTurns: Int?
    public var stepBudget: Int?
    public static let userDriven = SchedulerSpec(kind: "userDriven")
    public init(kind: String, maxTurns: Int? = nil, stepBudget: Int? = nil) {
        self.kind = kind; self.maxTurns = maxTurns; self.stepBudget = stepBudget
    }
}

/// `chats/<id>/conversation.json`. Absent means the V1 case: one agent and the user.
public struct ConversationConfig: Codable, Hashable, Sendable {
    public var version = 1
    public var title: String?
    public var participants: [Participant]
    public var scheduler: SchedulerSpec

    public var agentIDs: [String] { participants.compactMap(\.agent) }

    public static func oneToOne(agent: String) -> ConversationConfig {
        ConversationConfig(participants: [.agent(agent), .user], scheduler: .userDriven)
    }

    public init(participants: [Participant], scheduler: SchedulerSpec, title: String? = nil) {
        self.participants = participants; self.scheduler = scheduler; self.title = title
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        version = try c.decodeIfPresent(Int.self, forKey: .version) ?? 1
        title = try c.decodeIfPresent(String.self, forKey: .title)
        participants = try c.decodeIfPresent([Participant].self, forKey: .participants) ?? []
        scheduler = try c.decodeIfPresent(SchedulerSpec.self, forKey: .scheduler) ?? .userDriven
    }

    /// Reads `conversation.json` in a chat folder; falls back to the one-to-one default for `agentID`.
    public static func load(dir: URL, agentID: String) -> ConversationConfig {
        let url = dir.appendingPathComponent("conversation.json")
        guard let d = try? Data(contentsOf: url),
              let c = try? JSONDecoder().decode(ConversationConfig.self, from: d),
              !c.agentIDs.isEmpty else { return .oneToOne(agent: agentID) }
        return c
    }
}

/// Decides who speaks next, or that nobody does.
public protocol Scheduler: Sendable {
    func next(config: ConversationConfig, log: [ChatMessage]) -> String?
}

/// Today's chat: the agent replies whenever the user was the last to speak.
public struct UserDrivenScheduler: Scheduler {
    public init() {}
    public func next(config: ConversationConfig, log: [ChatMessage]) -> String? {
        guard let last = log.last(where: { $0.role != .marker }), last.role == .user else { return nil }
        return config.agentIDs.first
    }
}

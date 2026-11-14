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

/// What a scheduler may look at when deciding who speaks next.
public struct SchedulerState: Sendable {
    public var config: ConversationConfig
    public var log: [ChatMessage]
    /// Agent id to display name, used to find mentions.
    public var names: [String: String]
    /// Agent turns already taken since this run began (a user message or a manual run starts one).
    public var turnsThisRun: Int
    public init(config: ConversationConfig, log: [ChatMessage], names: [String: String] = [:], turnsThisRun: Int = 0) {
        self.config = config; self.log = log; self.names = names; self.turnsThisRun = turnsThisRun
    }
    var lastLine: ChatMessage? { log.last(where: { $0.role != .marker && !($0.role == .tool) && !($0.role == .assistant && $0.toolCalls != nil) }) }
}

/// Decides who speaks next, or that nobody does.
public protocol Scheduler: Sendable {
    func next(_ state: SchedulerState) -> String?
}

public extension Scheduler {
    func next(config: ConversationConfig, log: [ChatMessage], names: [String: String] = [:], turnsThisRun: Int = 0) -> String? {
        next(SchedulerState(config: config, log: log, names: names, turnsThisRun: turnsThisRun))
    }
}

/// Today's chat: the agent replies whenever the user was the last to speak.
public struct UserDrivenScheduler: Scheduler {
    public init() {}
    public func next(_ s: SchedulerState) -> String? {
        guard s.turnsThisRun == 0, s.lastLine?.role == .user else { return nil }
        return s.config.agentIDs.first
    }
}

/// Agents take turns in order, up to `maxTurns` (default: one round).
public struct RoundRobinScheduler: Scheduler {
    public init() {}
    public func next(_ s: SchedulerState) -> String? {
        let ids = s.config.agentIDs
        guard !ids.isEmpty, s.turnsThisRun < (s.config.scheduler.maxTurns ?? ids.count) else { return nil }
        return ids[s.turnsThisRun % ids.count]
    }
}

/// Whoever the last message names speaks next; a user message that names nobody goes to the first agent.
public struct MentionScheduler: Scheduler {
    public init() {}
    public func next(_ s: SchedulerState) -> String? {
        let ids = s.config.agentIDs
        guard let first = ids.first, s.turnsThisRun < (s.config.scheduler.maxTurns ?? 8), let last = s.lastLine else { return nil }
        let speaker = last.speaker
        let mentioned = Self.firstMention(in: last.text, ids: ids, names: s.names, excluding: speaker)
        if last.role == .user { return mentioned ?? first }
        return mentioned
    }

    /// The participant named earliest in `text` (by display name or id, case-insensitive, whole word).
    static func firstMention(in text: String, ids: [String], names: [String: String], excluding: String?) -> String? {
        var best: (Int, String)?
        for id in ids where id != excluding {
            for needle in Set([id, names[id] ?? id]) where !needle.isEmpty {
                let pattern = "(?<![\\p{L}\\p{N}])" + NSRegularExpression.escapedPattern(for: needle) + "(?![\\p{L}\\p{N}])"
                guard let re = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]),
                      let m = re.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) else { continue }
                if best == nil || m.range.location < best!.0 { best = (m.range.location, id) }
            }
        }
        return best?.1
    }
}

/// Agents keep going without a user message, in order, for at most `stepBudget` turns.
public struct AutonomousScheduler: Scheduler {
    public init() {}
    public func next(_ s: SchedulerState) -> String? {
        let ids = s.config.agentIDs
        guard !ids.isEmpty, s.turnsThisRun < (s.config.scheduler.stepBudget ?? s.config.scheduler.maxTurns ?? 6) else { return nil }
        return ids[s.turnsThisRun % ids.count]
    }
}

public extension SchedulerSpec {
    func make() -> Scheduler {
        switch kind {
        case "roundRobin": return RoundRobinScheduler()
        case "mention": return MentionScheduler()
        case "autonomous": return AutonomousScheduler()
        default: return UserDrivenScheduler()
        }
    }
    /// Upper bound on agent turns in one run, whatever the scheduler says.
    func hardCap(agentCount: Int) -> Int {
        switch kind {
        case "userDriven": return 1
        case "roundRobin": return min(maxTurns ?? agentCount, 50)
        case "mention": return min(maxTurns ?? 8, 50)
        case "autonomous": return min(stepBudget ?? maxTurns ?? 6, 50)
        default: return 1
        }
    }
    var label: String {
        switch kind {
        case "roundRobin": return "round robin"
        case "mention": return "by mention"
        case "autonomous": return "autonomous"
        default: return "you drive"
        }
    }
}

/// A conversation as the sidebar lists it.
public struct ConversationInfo: Identifiable, Hashable, Sendable {
    public var id: String
    public var config: ConversationConfig
    public var title: String { config.title ?? id }
}

extension ConversationConfig {
    /// Writes `conversation.json` (atomically) into the chat folder.
    func save(dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(self).write(to: dir.appendingPathComponent("conversation.json"), options: .atomic)
    }

    /// A folder is a room when its file names more than the single V1 agent or a non-default scheduler.
    var isRoom: Bool { agentIDs.count != 1 || scheduler.kind != "userDriven" }
}

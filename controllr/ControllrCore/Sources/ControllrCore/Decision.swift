import Foundation

public enum DecisionAction: String, Sendable, Codable {
    case run, done, respond, clarify, error
}

/// What Decide hands back to the router shortcut.
/// `shortcut` is the module id the router branches on; spoken outcomes use "speak" with input {speak}.
public struct Decision: Sendable, Equatable {
    public var action: DecisionAction
    public var shortcut: String
    public var input: [String: JSONValue]
    public var confirm: Bool
    public var summary: String
    public var threadID: String? = nil
    public var actionID: String? = nil

    public static let speakID = "speak"
    public static let openURLID = "open_url"
    public static let openURLConfirmID = "open_url_confirm"
    public static let nativeActionID = "approved_native"

    public static func speak(_ action: DecisionAction, _ text: String) -> Decision {
        Decision(action: action, shortcut: speakID, input: ["speak": .string(text)], confirm: false, summary: text)
    }

    public var listensAfterSpeaking: Bool { action == .clarify }

    public var json: String {
        let voiceNext = listensAfterSpeaking ? "listen" : (summary.isEmpty && action != .run ? "silent" : "stop")
        let fields: [String: JSONValue] = [
            "action": .string(action.rawValue),
            "shortcut": .string(shortcut),
            "input": .object(input),
            "confirm": .bool(confirm),
            "confirmation": .string(confirm ? "ask" : "run"),
            "summary": .string(summary),
            "voice_next": .string(voiceNext),
            "thread_id": .string(threadID ?? ""),
            "action_id": .string(actionID ?? ""),
        ]
        return JSONValue.object(fields).jsonString()
    }
}

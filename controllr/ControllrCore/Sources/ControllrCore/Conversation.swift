import Foundation

public enum ConversationMessage: Codable, Equatable, Sendable {
    case user(String)
    case assistant(String)
    case tool(call: ToolCall, output: String, assistantText: String? = nil)
    case followUp(call: ToolCall, assistantText: String?, spoken: String)

    /// A human reply completes a follow-up tool call on the next request.
    static func completedHistory(_ messages: [Self]) -> [Self] {
        var history: [Self] = []
        var pending: (call: ToolCall, text: String?)?
        for message in messages {
            switch message {
            case .followUp(let call, let text, _):
                pending = (call, text)
            case .user(let reply):
                if let pending {
                    history.append(.tool(call: pending.call, output: reply, assistantText: pending.text))
                } else {
                    history.append(message)
                }
                pending = nil
            case .assistant, .tool:
                pending = nil
                history.append(message)
            }
        }
        return history
    }
}

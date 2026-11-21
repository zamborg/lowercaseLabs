import Foundation

public enum ConversationMessage: Codable, Equatable, Sendable {
    case user(String)
    case assistant(String)
    case tool(call: ToolCall, output: String)
}

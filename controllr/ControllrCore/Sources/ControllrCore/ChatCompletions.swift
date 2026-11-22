import Foundation

/// The chat message and function-call wire format shared by OpenAI and OpenRouter.
enum ChatCompletions {
    static func fields(model: String, system: String, messages: [ConversationMessage],
                       tools: [ToolSpec]) -> [String: JSONValue] {
        let history: [JSONValue] = messages.flatMap { message -> [JSONValue] in
            switch message {
            case .user(let text):
                return [.object(["role": .string("user"), "content": .string(text)])]
            case .assistant(let text), .followUp(_, _, let text):
                return [.object(["role": .string("assistant"), "content": .string(text)])]
            case .tool(let call, let output, let assistantText):
                let id = call.id ?? "call_\(UUID().uuidString)"
                let toolCall: JSONValue = .object([
                    "id": .string(id), "type": .string("function"),
                    "function": .object(["name": .string(call.name), "arguments": .string(call.arguments.jsonString())]),
                ])
                var assistant: [String: JSONValue] = ["role": .string("assistant"), "tool_calls": .array([toolCall])]
                if let assistantText { assistant["content"] = .string(assistantText) }
                return [
                    .object(assistant),
                    .object(["role": .string("tool"), "tool_call_id": .string(id), "content": .string(output)]),
                ]
            }
        }
        var fields: [String: JSONValue] = [
            "model": .string(model),
            "messages": .array([.object(["role": .string("system"), "content": .string(system)])] + history),
        ]
        if !tools.isEmpty {
            fields["tools"] = .array(tools.map { tool in
                .object(["type": .string("function"), "function": .object([
                    "name": .string(tool.name), "description": .string(tool.description), "parameters": tool.parameters,
                ])])
            })
        }
        return fields
    }

    static func parseCompletion(_ json: JSONValue) throws -> AssistantCompletion {
        guard case .array(let choices)? = json["choices"], let message = choices.first?["message"] else {
            throw ProviderError.badResponse("no choices")
        }
        let text = message["content"]?.stringValue
        guard case .array(let calls)? = message["tool_calls"], !calls.isEmpty else {
            return AssistantCompletion(text: text)
        }
        guard calls.count == 1, let first = calls.first, let fn = first["function"],
              let name = fn["name"]?.stringValue, let arguments = fn["arguments"]?.stringValue,
              let args = try? JSONValue.parse(Data(arguments.utf8)) else {
            throw ProviderError.badResponse("invalid function call")
        }
        return AssistantCompletion(text: text, toolCall: ToolCall(name: name, arguments: args, id: first["id"]?.stringValue))
    }
}

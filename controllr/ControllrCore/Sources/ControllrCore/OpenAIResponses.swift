import Foundation

enum OpenAIResponses {
    static func requestBody(model: String, system: String, messages: [ConversationMessage], tools: [ToolSpec],
                            effort: ReasoningEffort, maxTokens: Int) -> JSONValue {
        let input = messages.flatMap { message -> [JSONValue] in
            switch message {
            case .user(let text):
                return [.object(["role": .string("user"), "content": .string(text)])]
            case .assistant(let text), .followUp(_, _, let text):
                return [.object(["role": .string("assistant"), "content": .string(text)])]
            case .tool(let call, let output, let assistantText):
                let rawID = call.id ?? UUID().uuidString
                let id = rawID.hasPrefix("call_") ? rawID : "call_\(rawID)"
                var items: [JSONValue] = []
                if case .array(let original)? = call.continuation(for: .openAIResponses, model: model) {
                    items = original
                } else {
                    if let assistantText {
                        items.append(.object(["role": .string("assistant"), "content": .string(assistantText)]))
                    }
                    items.append(.object([
                        "type": .string("function_call"), "call_id": .string(id), "name": .string(call.name),
                        "arguments": .string(call.arguments.jsonString()),
                    ]))
                }
                items.append(.object(["type": .string("function_call_output"), "call_id": .string(id), "output": .string(output)]))
                return items
            }
        }
        var body: [String: JSONValue] = [
            "model": .string(model),
            "instructions": .string(system),
            "input": .array(input),
            "store": .bool(false),
            "include": .array([.string("reasoning.encrypted_content")]),
            "max_output_tokens": .number(Double(maxTokens)),
            "reasoning": .object(["effort": .string(effort.rawValue)]),
        ]
        if !tools.isEmpty {
            body["parallel_tool_calls"] = .bool(false)
            body["tools"] = .array(tools.map { tool in
                .object(["type": .string("function"), "name": .string(tool.name), "description": .string(tool.description),
                         "parameters": tool.parameters, "strict": .bool(false)])
            })
        }
        return .object(body)
    }

    static func parseCompletion(_ json: JSONValue, model: String) throws -> AssistantCompletion {
        if json["status"] == .string("incomplete"), json["incomplete_details"]?["reason"] == .string("max_output_tokens") {
            throw ProviderError.outputLimit
        }
        guard json["status"] == .string("completed"), case .array(let items)? = json["output"] else {
            throw ProviderError.badResponse("response did not complete")
        }
        let text = items.flatMap { item -> [String] in
            guard item["type"] == .string("message"), case .array(let content)? = item["content"] else { return [] }
            return content.compactMap { block in
                switch block["type"]?.stringValue {
                case "output_text": block["text"]?.stringValue
                case "refusal": block["refusal"]?.stringValue
                default: nil
                }
            }
        }.joined(separator: "\n")
        let calls = items.filter { $0["type"] == .string("function_call") }
        guard !calls.isEmpty else { return AssistantCompletion(text: text.isEmpty ? nil : text) }
        guard calls.count == 1, let item = calls.first, let id = item["call_id"]?.stringValue,
              let name = item["name"]?.stringValue, let arguments = item["arguments"]?.stringValue,
              let args = try? JSONValue.parse(Data(arguments.utf8)) else {
            throw ProviderError.badResponse("invalid function call")
        }
        var call = ToolCall(name: name, arguments: args, id: id)
        call.continuation = ToolContinuation(format: .openAIResponses, model: model, payload: .array(items))
        return AssistantCompletion(text: text.isEmpty ? nil : text, toolCall: call)
    }
}

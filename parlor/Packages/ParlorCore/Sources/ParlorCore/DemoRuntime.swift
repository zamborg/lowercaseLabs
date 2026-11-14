import Foundation

/// Deterministic stand-in model used for UI tests and `PARLOR_DEMO=1`. No weights needed.
public final class DemoRuntime: LanguageModelRuntime, @unchecked Sendable {
    public init() {}
    public func loadedModel() async -> String? { "demo" }
    public func unload() async {}

    public func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
        AsyncThrowingStream { c in
            Task {
                let last = request.messages.last
                let stats = LLMStats(promptTokens: request.messages.reduce(0) { $0 + $1.text.count / 3 + 8 }, completionTokens: 12, tokensPerSecond: 60)
                func say(_ text: String) async {
                    for word in text.split(separator: " ", omittingEmptySubsequences: false) {
                        c.yield(.text(String(word) + " "))
                        try? await Task.sleep(nanoseconds: 15_000_000)
                    }
                }
                if let last, last.role == .tool {
                    await say("Done. The tool said: \(String(last.text.prefix(120)))")
                } else if let last, last.role == .user {
                    let lower = last.text.lowercased()
                    if lower.contains("briefing"), let t = request.tools.first(where: { $0.name == "briefing" }) {
                        c.yield(.toolCall(ToolCallRecord(name: t.wireName, arguments: .object([:]))))
                    } else if lower.contains("draw"), let t = request.tools.first(where: { $0.wireName == "image__generate" }) {
                        c.yield(.toolCall(ToolCallRecord(name: t.wireName, arguments: .object(["prompt": .string(last.text)]))))
                    } else if request.messages.first?.text.contains("faithful, compact conversation summaries") == true {
                        await say("SUMMARY: they talked.")
                    } else {
                        await say("(demo) You said: \(last.text)")
                    }
                }
                c.yield(.stats(stats))
                c.finish()
            }
        }
    }
}

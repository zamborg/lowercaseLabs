import Foundation
import SwiftData
import ControllrCore

/// Shared by the Decide intent and the in-app "Try it" screen.
enum Engine {
    static func makeProvider() -> LLMProvider? {
        let kind = Settings.provider
        guard let key = Keychain.get(kind.rawValue), !key.isEmpty else { return nil }
        let model = Settings.model(for: kind).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else { return nil }
        return kind.makeProvider(apiKey: key, model: model, reasoningEffort: Settings.reasoningEffort(for: kind, model: model))
    }

    static func run(_ transcript: String, messages: [ConversationMessage]? = nil, source: String, execute: Bool) async -> RouteResult {
        let result: RouteResult
        if let provider = makeProvider() {
            let router = Router(provider: provider, settings: Settings.routerSettings, executor: execute ? NativeExecutor() : nil,
                                policy: ModuleSettings.policy)
            result = await router.route(messages: messages ?? [.user(transcript)], execute: execute)
        } else {
            result = RouteResult(decision: .speak(.error, "Add your API key and model in controllr."), toolCall: nil, latencyMs: 0,
                                 provider: Settings.provider.rawValue, model: Settings.model(for: Settings.provider), error: "no key or model configured")
        }
        await MainActor.run { record(result, transcript: transcript, source: source) }
        return result
    }

    @MainActor
    private static func record(_ r: RouteResult, transcript: String, source: String) {
        let ctx = ModelContext(Store.container)
        ctx.insert(RunRecord(source: source, transcript: transcript, provider: r.provider, model: r.model,
                             module: r.moduleName ?? "-", argumentsJSON: r.toolCall?.arguments.jsonString() ?? "",
                             decisionJSON: r.decision.json, latencyMs: r.latencyMs, error: r.error ?? "", systemPrompt: r.systemPrompt))
        try? ctx.save()
    }
}

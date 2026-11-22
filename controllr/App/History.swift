import Foundation
import SwiftData

@Model
final class RunRecord {
    var date: Date
    var source: String          // "intent" | "try"
    var transcript: String
    var provider: String
    var model: String
    var module: String
    var argumentsJSON: String
    var decisionJSON: String
    var latencyMs: Int
    var error: String
    var systemPrompt: String?

    init(date: Date = .now, source: String, transcript: String, provider: String, model: String,
         module: String, argumentsJSON: String, decisionJSON: String, latencyMs: Int, error: String,
         systemPrompt: String? = nil) {
        self.date = date; self.source = source; self.transcript = transcript; self.provider = provider
        self.model = model; self.module = module; self.argumentsJSON = argumentsJSON
        self.decisionJSON = decisionJSON; self.latencyMs = latencyMs; self.error = error
        self.systemPrompt = systemPrompt
    }

    var exportText: String {
        """
        [\(date.formatted(.iso8601))] \(source) \(provider)/\(model) \(latencyMs)ms
        transcript: \(transcript)
        module: \(module) args: \(argumentsJSON)
        decision: \(decisionJSON)
        \(error.isEmpty ? "" : "error: \(error)")
        """
    }
}

enum Store {
    static let container: ModelContainer = {
        do { return try ModelContainer(for: RunRecord.self, ChatThread.self, ChatEntry.self, SavedItem.self) }
        catch { fatalError("SwiftData store failed: \(error)") }
    }()
}

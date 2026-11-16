import SwiftUI
import ParlorCore

/// Everything the UI changes goes through a verb, the same ones an agent or an outside client calls. The engine's own
/// mutators are not called from this folder (`CommandTests.testTheAppOnlyMutatesThroughVerbs` checks).
@MainActor
func runVerb(_ engine: ParlorEngine, _ verb: String, _ args: [String: JSON] = [:]) async -> ToolOutput {
    await engine.commands.call(verb, .object(args), caller: .ui)
}

/// Run a verb and parse its JSON answer. Failures come back as the verb's own message.
@MainActor
func runVerbJSON(_ engine: ParlorEngine, _ verb: String, _ args: [String: JSON] = [:]) async -> Result<JSON, VerbError> {
    let r = await runVerb(engine, verb, args)
    if r.isError { return .failure(VerbError(message: r.text)) }
    return .success((try? JSONDecoder().decode(JSON.self, from: Data(r.text.utf8))) ?? .null)
}

struct VerbError: Error, LocalizedError { var message: String; var errorDescription: String? { message } }

/// A JSON merge patch that turns `old` into `new`: changed keys carry their new value, removed keys carry null.
func mergePatch(from old: JSON, to new: JSON) -> JSON? {
    guard case .object(let o) = old, case .object(let n) = new else { return old == new ? nil : new }
    var out: [String: JSON] = [:]
    for (k, v) in n {
        if let ov = o[k] {
            if ov == v { continue }
            if case .object = ov, case .object = v, let sub = mergePatch(from: ov, to: v) { out[k] = sub } else { out[k] = v }
        } else { out[k] = v }
    }
    for k in o.keys where n[k] == nil { out[k] = .null }
    return out.isEmpty ? nil : .object(out)
}

extension ParlorEngine {
    /// Save an edited agent as a patch against what is on disk.
    @MainActor func saveAgent(_ draft: Agent) async {
        guard let old = agent(draft.id) else { return }
        guard let patch = mergePatch(from: agentJSON(old), to: agentJSON(draft)) else { return }
        _ = await runVerb(self, "agents.update", ["id": .string(draft.id), "patch": patch])
    }

    /// A new agent from a harness template; returns its id.
    @MainActor func newAgent(harness: String, template: String) async -> String? {
        if case .success(let j) = await runVerbJSON(self, "agents.create", ["template": .string("\(harness):\(template)")]) { return j.objectValue?["id"]?.stringValue }
        return nil
    }
}

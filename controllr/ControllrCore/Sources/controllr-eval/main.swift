import ControllrCore
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("\(message)\n".utf8))
    exit(2)
}

var args = Array(CommandLine.arguments.dropFirst())
@MainActor func take(_ flag: String) -> String? {
    guard let index = args.firstIndex(of: flag), index + 1 < args.count else { return nil }
    let value = args[index + 1]
    args.removeSubrange(index...(index + 1))
    return value
}

let env = ProcessInfo.processInfo.environment
let providerName = take("--provider") ?? (env["OPENAI_API_KEY"] != nil ? "openai" : "anthropic")
let file = take("--file")
let recordNote = take("--record")
let reportPath = take("--report") ?? "../evals/RESULTS.md"
let model = take("--model")
let listModels = args.contains("--list-models")

let provider: LLMProvider
switch providerName {
case "openai":
    guard let key = env["OPENAI_API_KEY"], !key.isEmpty else { fail("Set OPENAI_API_KEY") }
    guard model != nil || listModels else { fail("Pass --model (use --list-models to see which are available)") }
    provider = OpenAIProvider(apiKey: key, model: model ?? "unset")
case "anthropic":
    guard let key = env["ANTHROPIC_API_KEY"], !key.isEmpty else { fail("Set ANTHROPIC_API_KEY") }
    provider = AnthropicProvider(apiKey: key, model: model ?? "claude-haiku-5-5")
default:
    fail("Unknown provider \(providerName)")
}

if listModels {
    do {
        for model in try await provider.listModels() { print(model) }
    } catch {
        fail("Could not list models for \(providerName)")
    }
    exit(0)
}

struct Case: Sendable {
    let scenario: ActionEval
    let expectedTool: String
}

let cases: [Case]
if let file {
    guard let text = try? String(contentsOfFile: file, encoding: .utf8) else { fail("Cannot read \(file)") }
    cases = text.split(separator: "\n").enumerated().map { index, line in
        let data = Data(line.utf8)
        guard let value = try? JSONValue.parse(data), let expected = value["expect"]?.stringValue,
              let scenario = try? JSONDecoder().decode(ActionEval.self, from: data) else {
            fail("Invalid eval case on line \(index + 1) of \(file)")
        }
        return Case(scenario: scenario, expectedTool: expected)
    }
} else {
    cases = Registry.v1.modules.flatMap { module in
        module.evals.map { Case(scenario: $0, expectedTool: module.id) }
    }
}
guard !cases.isEmpty else { fail("No eval cases") }

let router = Router(provider: provider)
struct Outcome: Sendable {
    let test: Case
    let tool: String
    let ms: Int
    let error: String?

    var correct: Bool { tool == test.expectedTool && error == nil }
}

let outcomes = await withTaskGroup(of: (Int, Outcome).self) { group in
    var next = 0
    var results: [(Int, Outcome)] = []
    func add() {
        guard next < cases.count else { return }
        let index = next
        let test = cases[index]
        next += 1
        group.addTask {
            let messages = (test.scenario.history?.map(\.message) ?? []) + [.user(test.scenario.utterance)]
            let result = await router.route(messages: messages, execute: false)
            let mismatch = test.scenario.arguments?.keys.sorted().first {
                result.toolCall?.arguments[$0] != test.scenario.arguments?[$0]
            }
            let unexpectedClarification = result.decision.action == .clarify && test.expectedTool != "clarify"
            let error = result.error ?? mismatch.map { "wrong \($0)" }
                ?? (unexpectedClarification ? "requested another turn" : nil)
            return (index, Outcome(test: test, tool: result.toolCall?.name ?? "error", ms: result.latencyMs, error: error))
        }
    }
    for _ in 0..<4 { add() }
    while let result = await group.next() {
        results.append(result)
        add()
    }
    return results.sorted { $0.0 < $1.0 }.map(\.1)
}

let correct = outcomes.filter(\.correct).count
let byModule = Dictionary(grouping: outcomes) { $0.test.expectedTool }
let latencies = outcomes.map(\.ms).sorted()
func percentile(_ value: Double) -> Int {
    latencies[min(latencies.count - 1, Int(Double(latencies.count) * value))]
}

var report = "provider=\(provider.name) model=\(provider.model) cases=\(cases.count)\n"
report += "accuracy \(correct)/\(cases.count) = \(Int(Double(correct) * 100 / Double(cases.count)))%\n"
for (module, results) in byModule.sorted(by: { $0.key < $1.key }) {
    report += "  \(module.padding(toLength: 19, withPad: " ", startingAt: 0)) \(results.filter(\.correct).count)/\(results.count)\n"
}
report += "latency ms: p50 \(percentile(0.50)) p95 \(percentile(0.95)) max \(latencies.last ?? 0)\n"
let misses = outcomes.filter { !$0.correct }
if !misses.isEmpty {
    report += "misses:\n"
    for miss in misses {
        let details = miss.error.map { " (\($0.prefix(80)))" } ?? ""
        report += "  \"\(miss.test.scenario.utterance)\" expected \(miss.test.expectedTool) got \(miss.tool)\(details)\n"
    }
}
print(report)

if let recordNote {
    let entry = "\n## \(Date().formatted(.iso8601)) — \(recordNote)\n```\n\(report)```\n"
    if let handle = FileHandle(forWritingAtPath: reportPath) {
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(entry.utf8))
        try handle.close()
    } else {
        try ("# Eval results\n" + entry).write(toFile: reportPath, atomically: true, encoding: .utf8)
    }
}

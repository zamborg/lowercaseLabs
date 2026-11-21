import Foundation
import ControllrCore

// controllr-eval: routes each utterance in evals/utterances.jsonl and reports accuracy + latency.
// Keys come from OPENAI_API_KEY / ANTHROPIC_API_KEY and are never printed.
//   controllr-eval [--provider openai|anthropic] [--model ID] [--file PATH] [--list-models] [--record NOTE]

func fail(_ msg: String) -> Never { FileHandle.standardError.write(Data((msg + "\n").utf8)); exit(2) }

var args = Array(CommandLine.arguments.dropFirst())
@MainActor func take(_ flag: String) -> String? {
    guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
    let v = args[i + 1]; args.removeSubrange(i...(i + 1)); return v
}
@MainActor func has(_ flag: String) -> Bool { if let i = args.firstIndex(of: flag) { args.remove(at: i); return true }; return false }

let env = ProcessInfo.processInfo.environment
let providerName = take("--provider") ?? (env["OPENAI_API_KEY"] != nil ? "openai" : "anthropic")
let file = take("--file") ?? "../evals/utterances.jsonl"
let listModels = has("--list-models")
let recordNote = take("--record")
var model = take("--model")

let provider: LLMProvider
switch providerName {
case "openai":
    guard let key = env["OPENAI_API_KEY"], !key.isEmpty else { fail("Set OPENAI_API_KEY") }
    provider = OpenAIProvider(apiKey: key, model: model ?? "unset")
case "anthropic":
    guard let key = env["ANTHROPIC_API_KEY"], !key.isEmpty else { fail("Set ANTHROPIC_API_KEY") }
    provider = AnthropicProvider(apiKey: key, model: model ?? "claude-haiku-5-5")
default: fail("unknown provider \(providerName)")
}

if listModels {
    do { for m in try await provider.listModels() { print(m) } } catch { fail("list-models failed: \(error)") }
    exit(0)
}
guard model != nil || providerName == "anthropic" else { fail("Pass --model (use --list-models to see what your key can use)") }
model = model ?? "claude-haiku-5-5"

struct Case { let utterance: String; let expect: String }
guard let text = try? String(contentsOfFile: file, encoding: .utf8) else { fail("Cannot read \(file)") }
let cases: [Case] = text.split(separator: "\n").compactMap { line in
    guard let v = try? JSONValue.parse(Data(line.utf8)), let u = v["utterance"]?.stringValue, let e = v["expect"]?.stringValue else { return nil }
    return Case(utterance: u, expect: e)
}
guard !cases.isEmpty else { fail("No cases in \(file)") }

let router = Router(provider: provider)
struct Outcome: Sendable { let c: Case; let got: String; let ms: Int; let error: String? }

let outcomes: [Outcome] = await withTaskGroup(of: (Int, Outcome).self) { group in
    var next = 0
    var results = [(Int, Outcome)]()
    func add() {
        guard next < cases.count else { return }
        let i = next, c = cases[i]; next += 1
        group.addTask {
            let r = await router.route(c.utterance, execute: false)
            return (i, Outcome(c: c, got: r.toolCall?.name ?? "error", ms: r.latencyMs, error: r.error))
        }
    }
    for _ in 0..<4 { add() }
    while let r = await group.next() { results.append(r); add() }
    return results.sorted { $0.0 < $1.0 }.map(\.1)
}

let correct = outcomes.filter { $0.got == $0.c.expect }.count
var byModule: [String: (ok: Int, n: Int)] = [:]
for o in outcomes { var s = byModule[o.c.expect] ?? (0, 0); s.n += 1; if o.got == o.c.expect { s.ok += 1 }; byModule[o.c.expect] = s }
let lat = outcomes.map(\.ms).sorted()
func pct(_ p: Double) -> Int { lat[min(lat.count - 1, Int(Double(lat.count) * p))] }

var report = "provider=\(provider.name) model=\(provider.model) cases=\(cases.count)\n"
report += "accuracy \(correct)/\(cases.count) = \(Int(Double(correct) * 100 / Double(cases.count)))%\n"
for (m, s) in byModule.sorted(by: { $0.key < $1.key }) { report += "  \(m.padding(toLength: 16, withPad: " ", startingAt: 0)) \(s.ok)/\(s.n)\n" }
report += "latency ms: p50 \(pct(0.5)) p95 \(pct(0.95)) max \(lat.last ?? 0)\n"
let misses = outcomes.filter { $0.got != $0.c.expect }
if !misses.isEmpty {
    report += "misses:\n"
    for o in misses { report += "  \"\(o.c.utterance)\" expected \(o.c.expect) got \(o.got)\(o.error.map { " (\($0.prefix(80)))" } ?? "")\n" }
}
print(report)

if let note = recordNote {
    let df = ISO8601DateFormatter()
    let entry = "\n## \(df.string(from: Date())) — \(note)\n```\n\(report)```\n"
    let path = (file as NSString).deletingLastPathComponent + "/RESULTS.md"
    if let h = FileHandle(forWritingAtPath: path) { h.seekToEndOfFile(); h.write(Data(entry.utf8)); try? h.close() }
    else { try? ("# Eval results\n" + entry).write(toFile: path, atomically: true, encoding: .utf8) }
}

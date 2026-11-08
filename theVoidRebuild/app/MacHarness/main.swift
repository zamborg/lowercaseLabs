import AVFoundation
import Foundation

// Mac harness: runs the app's real LiveTranscription and Tagger. No phone, no mic.
// Audio fixtures stream through the same append()/finish() path the mic tap uses.
// Exits non-zero on any failure or regression.
setvbuf(stdout, nil, _IOLBF, 0)
var failures = 0
func check(_ ok: Bool, _ message: String) { if !ok { failures += 1; print("FAIL \(message)") } }

func words(_ s: String) -> [String] {
    s.lowercased().split { !$0.isLetter && !$0.isNumber && $0 != "'" }.map(String.init)
}
func wordErrors(_ ref: [String], _ hyp: [String]) -> Int {
    var d = Array(0...hyp.count)
    for i in 1...max(ref.count, 1) where i <= ref.count {
        var prev = d[0]; d[0] = i
        for j in 1...max(hyp.count, 1) where j <= hyp.count {
            let tmp = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] == hyp[j - 1] ? 0 : 1))
            prev = tmp
        }
    }
    return d[hyp.count]
}

let root = ProcessInfo.processInfo.environment["VOID_FIXTURES"] ?? "../eval/transcription/fixtures/synthetic"
let refs = try String(contentsOfFile: "\(root)/reference.jsonl", encoding: .utf8)
    .split(separator: "\n").compactMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: String] }

// 1. Speech: fixtures -> LiveTranscription -> transcript, scored by WER.
print("== speech")
var refWords = 0, errors = 0
var transcripts: [String] = []
let warm: LiveTranscription.Warm
do { warm = try await LiveTranscription.warm() } catch { print("FAIL speech warm: \(error)"); exit(1) }
for entry in refs {
    guard let name = entry["file"], let text = entry["text"] else { continue }
    let file = try AVAudioFile(forReading: URL(fileURLWithPath: "\(root)/audio/\(name)"))
    let live = try LiveTranscription(warm: warm, inputFormat: file.processingFormat)
    while file.framePosition < file.length {   // 512-frame chunks, like the mic tap
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 512) else { break }
        try file.read(into: buffer)
        if buffer.frameLength == 0 { break }
        live.append(buffer)
    }
    let t = CFAbsoluteTimeGetCurrent()
    let got = await live.finish()
    let e = wordErrors(words(text), words(got))
    refWords += words(text).count; errors += e
    transcripts.append(got)
    print("\(name) finish_ms=\(Int((CFAbsoluteTimeGetCurrent() - t) * 1000)) errors=\(e) \"\(got)\"")
}
let wer = Double(errors) / Double(max(refWords, 1))
print(String(format: "WER %.1f%% (%d/%d)", wer * 100, errors, refWords))
check(wer <= 0.10, "WER \(wer) over 10% budget")

// 2. Tagging: the sentences the model must get right, plus speed and well-formed output.
print("== tagging")
let samples: [(String, Quadrant)] = [
    ("I'm so excited, I just got the job offer and I can't stop smiling.", .highPleasant),
    ("I'm exhausted and honestly just feel flat today, nothing is going right.", .lowUnpleasant),
    ("The deadline is tomorrow and I'm panicking, my heart is racing.", .highUnpleasant),
    ("Quiet evening, tea on the porch, feeling really grateful and calm.", .lowPleasant),
]
let tagger = Tagger()
do { try await tagger.warm() } catch { print("FAIL tagger warm: \(error)"); exit(1) }
var hits = 0
for (text, expected) in samples + transcripts.map({ ($0, Quadrant?.none) }).compactMap({ $0.1 == nil ? nil : ($0.0, $0.1!) }) {
    let t = CFAbsoluteTimeGetCurrent()
    do {
        let r = try await tagger.tag(transcript: text)
        let ms = Int((CFAbsoluteTimeGetCurrent() - t) * 1000)
        let hit = r.tags.first == expected
        if hit { hits += 1 }
        print("ms=\(ms) \(hit ? "hit " : "MISS") title=\"\(r.title)\" tags=\(r.tags.map(\.rawValue))")
        check(!r.title.isEmpty && !r.tags.isEmpty && Set(r.tags).count == r.tags.count, "malformed tagging for: \(text)")
        check(ms < 3000, "tagging \(ms) ms over budget on Mac")
    } catch { check(false, "tag threw: \(error)") }
}
print("accuracy \(hits)/\(samples.count)")
check(hits >= 2, "tag accuracy \(hits)/\(samples.count) below floor")

// 3. Optional full gate: `./validate.sh --eval` scores the shipped tagger on the labeled set.
if ProcessInfo.processInfo.environment["VOID_EVAL"] != nil {
    print("== eval (clean split)")
    let rows = try String(contentsOfFile: "../eval/data/memos.jsonl", encoding: .utf8)
        .split(separator: "\n").compactMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any] }
        .filter { ($0["split"] as? String) == "clean" }
    var correct = 0, total = 0, over = 0
    var confusion: [String: Int] = [:]
    var totalMs = 0
    for row in rows {
        guard let label = row["label"] as? String, let text = row["transcript"] as? String else { continue }
        let t = CFAbsoluteTimeGetCurrent()
        guard let r = try? await tagger.tag(transcript: text) else { check(false, "eval tag threw"); continue }
        let ms = Int((CFAbsoluteTimeGetCurrent() - t) * 1000)
        totalMs += ms; total += 1
        if ms > 3000 { over += 1 }
        let got = r.tags.first?.rawValue ?? "none"
        if got == label { correct += 1 } else { confusion["\(label) -> \(got)", default: 0] += 1 }
    }
    let acc = Double(correct) / Double(max(total, 1))
    print(String(format: "top1 %.1f%% (%d/%d), mean %d ms, %d over budget", acc * 100, correct, total, totalMs / max(total, 1), over))
    for (k, v) in confusion.sorted(by: { $0.value > $1.value }) { print("  \(v)x \(k)") }
    let baseline = (try? String(contentsOfFile: "tagging-baseline.txt", encoding: .utf8)).flatMap { Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) } ?? 0
    print(String(format: "baseline %.1f%%", baseline * 100))
    check(acc >= baseline - 0.02, "accuracy \(acc) regressed below baseline \(baseline)")
    if CommandLine.arguments.contains("--update-baseline") { try? String(acc).write(toFile: "tagging-baseline.txt", atomically: true, encoding: .utf8) }
}

print(failures == 0 ? "PASS" : "FAILED (\(failures))")
exit(failures == 0 ? 0 : 1)

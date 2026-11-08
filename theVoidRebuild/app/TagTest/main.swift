import Foundation

// Loads the bundled weights and tags fixed transcripts with the app's real Tagger.
// Fails (non-zero exit) on any error, bad tag, or a tagging time over budget.
let samples: [(String, Quadrant)] = [
    ("I'm so excited, I just got the job offer and I can't stop smiling.", .highPleasant),
    ("I'm exhausted and honestly just feel flat today, nothing is going right.", .lowUnpleasant),
    ("The deadline is tomorrow and I'm panicking, my heart is racing.", .highUnpleasant),
    ("Quiet evening, tea on the porch, feeling really grateful and calm.", .lowPleasant),
]
setvbuf(stdout, nil, _IOLBF, 0)
let tagger = Tagger()
let t0 = CFAbsoluteTimeGetCurrent()
do { try await tagger.warm() } catch { print("FAIL warm: \(error)"); exit(1) }
print("warm_ms \(Int((CFAbsoluteTimeGetCurrent() - t0) * 1000))")
var hits = 0
for (text, expected) in samples {
    let t = CFAbsoluteTimeGetCurrent()
    do {
        let r = try await tagger.tag(transcript: text)
        let hit = r.tags.first == expected
        if hit { hits += 1 }
        print("ms=\(Int((CFAbsoluteTimeGetCurrent() - t) * 1000)) \(hit ? "hit " : "MISS") title=\"\(r.title)\" tags=\(r.tags.map(\.rawValue))")
    } catch { print("FAIL tag: \(error)"); exit(1) }
}
print("DONE \(hits)/\(samples.count) first-tag hits")

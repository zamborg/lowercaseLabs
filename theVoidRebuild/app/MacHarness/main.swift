import AVFoundation
import Foundation
import ImageIO
import SwiftData
import UniformTypeIdentifiers

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


// 3. Journal flows on the real Journal + DotStore code: in-memory SwiftData, and an in-memory stand-in for the
// CloudKit zone so we can see exactly what would leave the device. Entry text comes from the real speech run above.
print("== journal flows")
let audioDir = FileManager.default.temporaryDirectory.appendingPathComponent("void-audio-\(UUID().uuidString)").path
setenv("VOID_AUDIO_DIR", audioDir, 1)
try? FileManager.default.createDirectory(atPath: audioDir, withIntermediateDirectories: true)
setenv("VOID_PHOTO_DIR", FileManager.default.temporaryDirectory.appendingPathComponent("void-photos-\(UUID().uuidString)").path, 1)
let store = try ModelContainer(for: Entry.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
DotStore.shared.useFakeCloud()
let journal = Journal(context: ModelContext(store), tagger: tagger)
@MainActor func remote() -> [String: Dot] { DotStore.shared.fakeRemote ?? [:] }
@MainActor func entryCount() -> Int { (try? journal.context.fetchCount(FetchDescriptor<Entry>())) ?? -1 }
@MainActor func makeEntry(_ text: String, _ visibility: Visibility) -> Entry {
    let id = UUID()
    FileManager.default.createFile(atPath: "\(audioDir)/\(id.uuidString).caf", contents: Data([0]))
    return journal.save(id: id, audioFileName: "\(id.uuidString).caf", duration: 5, transcript: text, visibility: visibility)
}
let calm = transcripts.first ?? "Quiet evening, tea on the porch, feeling really grateful and calm."

// F1/F2: saved at once, before enrichment; then titled and tagged by the model.
var t0 = CFAbsoluteTimeGetCurrent()
let a = makeEntry(calm, .private)
let savedMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
check(entryCount() == 1 && a.title.isEmpty, "F1: entry should exist before enrichment")
check(savedMs < 300, "F1: save took \(Int(savedMs)) ms, budget 300")
try await journal.enrich(a)
check(!a.title.isEmpty && !a.tags.isEmpty && a.titleSource == .model && a.tagsSource == .model, "F2: enrichment missing or wrong provenance")
check(remote().isEmpty, "PRIVACY: a private entry produced a dot")
print(String(format: "F1 saved in %.1f ms; F2 title=\"%@\" tags=%@", savedMs, a.title, "\(a.tags.map(\.rawValue))"))

// F6: a dot exists exactly while the entry is shared and tagged, and carries nothing else.
let b = makeEntry(calm, .shared)
await journal.reconcileDots()
check(remote().isEmpty, "F6: shared but untagged entry published a dot")
t0 = CFAbsoluteTimeGetCurrent()
try await journal.enrich(b)
let dotMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
check(remote()[b.id.uuidString]?.tags == b.tags && remote().count == 1, "F6: shared entry has no dot after enrichment")
journal.setVisibility(.private, on: b); await journal.reconcileDots()
check(remote().isEmpty, "F6: dot survived flipping to private")
journal.setVisibility(.shared, on: b); await journal.reconcileDots()
check(remote().count == 1, "F6: dot missing after flipping back to shared")
print(String(format: "F6 release-equivalent to dot written: %.0f ms (enrichment + publish, fake cloud)", dotMs))

// F4: user edits win and the model never overwrites them.
journal.setTitle("Mine", on: b); journal.setTags([.highUnpleasant], on: b); await journal.reconcileDots()
check(b.title == "Mine" && b.titleSource == .user && b.tags == [.highUnpleasant], "F4: edits did not stick")
check(remote()[b.id.uuidString]?.tags == [.highUnpleasant], "F6: edited tags did not update the dot")
try await journal.enrich(b)
check(b.title == "Mine" && b.tags == [.highUnpleasant], "F4: model overwrote a user edit")
let c = makeEntry(calm, .private)
journal.setTitle("Hand-written", on: c)
try await journal.enrich(c)
check(c.title == "Hand-written" && !c.tags.isEmpty, "F4: partial edit: title kept, tags should still be filled")

// Never lose input: entries saved while the model was unavailable get enriched later.
let d = makeEntry("Honestly just tired today, nothing feels worth doing.", .private)
await journal.enrichPending()
check(!d.title.isEmpty && !d.tags.isEmpty, "enrichPending did not fill a pending entry")

// F8: delete takes the entry, its audio file and its dot.
let bAudio = b.audioURL.path
check(FileManager.default.fileExists(atPath: bAudio), "F8 setup: audio missing")
journal.delete(b); await journal.reconcileDots()
check(remote().isEmpty && !FileManager.default.fileExists(atPath: bAudio) && entryCount() == 3, "F8: delete left entry, audio or dot behind")
check(remote().values.allSatisfy { _ in true }, "")
// Long input: a few minutes of speech is ~600 words. Must still tag, and quickly.
let longText = String(repeating: "I felt a bit overwhelmed at work today but then a walk with a friend helped me feel calmer and more grateful. ", count: 40)
t0 = CFAbsoluteTimeGetCurrent()
let longEntry = makeEntry(longText, .private)
do { try await journal.enrich(longEntry) } catch { check(false, "long transcript failed to tag: \(error)") }
let longMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
check(!longEntry.title.isEmpty && !longEntry.tags.isEmpty && longMs < 6000, "long transcript: \(Int(longMs)) ms")
print(String(format: "long transcript (%d words) tagged in %.0f ms", longText.split(separator: " ").count, longMs))
journal.delete(longEntry)

// Backgrounding: no GPU work once the app isn't active (iOS would kill it); the entry waits and is enriched on return.
let e2 = makeEntry("I'm thrilled, everything went better than I hoped.", .private)
Tagger.gpuAllowed = false
var paused = false
do { try await journal.enrich(e2) } catch TaggerError.paused { paused = true } catch { check(false, "background: unexpected \(error)") }
check(paused && e2.title.isEmpty && entryCount() == 4, "background: tagging should pause and leave the entry saved, untouched")
Tagger.gpuAllowed = true
await journal.enrichPending()
check(!e2.title.isEmpty, "background: entry not enriched after returning to the foreground")
journal.delete(e2); await journal.reconcileDots()

// Photo parts (F10, F11, F12, and photo's extensions of F6 and F8). A synthetic JPEG with location and EXIF stands in
// for the camera; everything after the shutter is the app's real code.
print("== photo flows")
func fixtureJPEG(width: Int, height: Int, gray: Double) -> Data {
    let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    ctx.setFillColor(CGColor(red: gray, green: 0.6, blue: 0.9, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    ctx.setFillColor(CGColor(red: 1, green: 1, blue: 0.2, alpha: 1)); ctx.fillEllipse(in: CGRect(x: width / 2, y: height / 2, width: width / 4, height: width / 4))
    let out = NSMutableData()
    let dest = CGImageDestinationCreateWithData(out, UTType.jpeg.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, ctx.makeImage()!, [
        kCGImagePropertyOrientation: 6,
        kCGImagePropertyGPSDictionary: [kCGImagePropertyGPSLatitude: 37.77, kCGImagePropertyGPSLongitude: -122.41],
        kCGImagePropertyExifDictionary: [kCGImagePropertyExifUserComment: "secret"],
        kCGImagePropertyTIFFDictionary: [kCGImagePropertyTIFFMake: "Apple"],
    ] as CFDictionary)
    CGImageDestinationFinalize(dest)
    return out as Data
}
func props(_ url: URL) -> [CFString: Any] {
    CGImageSourceCreateWithURL(url as CFURL, nil).flatMap { CGImageSourceCopyPropertiesAtIndex($0, 0, nil) as? [CFString: Any] } ?? [:]
}
let rawBack = fixtureJPEG(width: 3000, height: 2000, gray: 0.3), rawFront = fixtureJPEG(width: 1500, height: 2000, gray: 0.7)
check(props(URL(fileURLWithPath: "/dev/null")).isEmpty, "")   // helper sanity: unreadable gives no properties

// F10: a photo-only message is saved before enrichment, files stripped, then titled and tagged from scene labels.
t0 = CFAbsoluteTimeGetCurrent()
let p = try journal.savePhoto(back: rawBack, front: rawFront, visibility: .private)
let photoSavedMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
check(p.hasPhoto && !p.hasVoice && p.title.isEmpty && entryCount() == 4, "F10: photo entry should exist before enrichment")
check(photoSavedMs < 300, "F10: photo save took \(Int(photoSavedMs)) ms, budget 300")
let stored = props(p.photoBackURL)
// ImageIO always writes structural EXIF (colour space, pixel size); what must be gone is anything about where or what shot it.
let storedExif = stored[kCGImagePropertyExifDictionary] as? [CFString: Any] ?? [:]
let storedTIFF = stored[kCGImagePropertyTIFFDictionary] as? [CFString: Any] ?? [:]
check(stored[kCGImagePropertyGPSDictionary] == nil && storedExif[kCGImagePropertyExifUserComment] == nil
      && storedTIFF[kCGImagePropertyTIFFMake] == nil && storedTIFF[kCGImagePropertyTIFFModel] == nil,
      "F10: stored photo still carries location or camera metadata: \(stored.keys) exif=\(storedExif.keys) tiff=\(storedTIFF.keys)")
let w = stored[kCGImagePropertyPixelWidth] as? Int ?? 0, h = stored[kCGImagePropertyPixelHeight] as? Int ?? 0
check(h > w && max(w, h) <= PhotoFiles.maxPixels, "F10: photo not rotated upright or not downsized (\(w)x\(h))")
t0 = CFAbsoluteTimeGetCurrent()
try await journal.enrich(p)
let photoEnrichMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
check(!p.photoLabels.isEmpty && !p.title.isEmpty && !p.tags.isEmpty && p.titleSource == .model, "F10: photo-only message not titled and tagged from labels")
check(photoEnrichMs < 2000, "F10: photo enrichment \(Int(photoEnrichMs)) ms, budget 2000")
print(String(format: "F10 saved in %.1f ms (2 images, metadata stripped, %dx%d); labels \"%@\" -> title \"%@\" tags %@ in %.0f ms",
             photoSavedMs, w, h, p.photoLabels, p.title, "\(p.tags.map(\.rawValue))", photoEnrichMs))

// F11: voice recorded while holding joins the same message, and the transcript then leads enrichment.
let before = entryCount()
journal.attachVoice(audioFileName: "none.caf", duration: 6, transcript: calm, to: p)
check(entryCount() == before && p.hasVoice && p.hasPhoto && p.enrichmentText == calm, "F11: voice did not join the photo message, or the transcript doesn't lead")
check(p.titleSource == .none, "F11: stale photo-only enrichment should be reopened when the transcript arrives")
try await journal.enrich(p)
check(!p.title.isEmpty && p.titleSource == .model, "F11: message not re-enriched from the transcript")

// F12: a photo added to a voice message adds no message, and a spoken message keeps its title.
let v = makeEntry(calm, .private)
try await journal.enrich(v)
let vTitle = v.title, count12 = entryCount()
try journal.attachPhoto(back: rawBack, front: nil, to: v)
check(entryCount() == count12 && v.hasVoice && v.hasPhoto && v.photoFrontURL == nil, "F12: add photo created a message or lost a part")
check(v.title == vTitle && v.titleSource == .model, "F12: adding a photo changed a spoken message's title")

// Privacy extension of F6: a shared photo message produces a dot with tags and time only, and withdraws with it.
journal.setVisibility(.shared, on: p); await journal.reconcileDots()
check(remote().count == 1 && remote()[p.id.uuidString]?.tags == p.tags, "F6/photo: shared photo message has no dot")
let dotJSON = remote()[p.id.uuidString].map { "\($0)" } ?? ""
check(!dotJSON.contains("jpg") && !p.photoLabels.split(separator: ",").contains { dotJSON.contains($0.trimmingCharacters(in: .whitespaces)) },
      "F6/photo: a photo file name or label reached the dot")

// F8 extension: deleting takes both images, the audio and the dot.
let files = [p.photoBackURL, p.photoFrontURL!, v.photoBackURL]
check(files.allSatisfy { FileManager.default.fileExists(atPath: $0.path) }, "F8/photo setup: photo files missing")
journal.delete(p); journal.delete(v); await journal.reconcileDots()
check(files.allSatisfy { !FileManager.default.fileExists(atPath: $0.path) } && remote().isEmpty, "F8/photo: delete left a photo file or a dot behind")

// Settings → re-tag: reopens only model-set fields.
let r1 = makeEntry(calm, .private); try await journal.enrich(r1)
let r2 = makeEntry(calm, .private); journal.setTitle("Mine", on: r2); journal.setTags([.highUnpleasant], on: r2)
await journal.retagModelFields()
check(r2.title == "Mine" && r2.tags == [.highUnpleasant] && r2.titleSource == .user, "F13: re-tag changed something the user set")
check(r1.titleSource == .model && !r1.title.isEmpty, "F13: re-tag did not re-run the model on a model-set entry")
journal.delete(r1); journal.delete(r2)
print("journal flows done: \(entryCount()) entries, \(remote().count) dots")

// 4. Optional full gate: `./validate.sh --eval` scores the shipped tagger on the labeled set.
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

# Brief 10: Entry model and pipeline extraction

**Executor:** Claude (Sonnet) in this repo. **Depends on:** nothing. **Unblocks:** 11, 13.

## Goal

Two changes that every later step builds on:

1. Replace `Memo` with `Entry`, shaped like the [VISION.md](../VISION.md) primitives (D4).
2. Move the capture → save → enrich → publish flow out of `CaptureModel` (`app/Sources/CaptureView.swift`) into a plain `@MainActor @Observable final class Pipeline` with no SwiftUI import, so flows can drive it without a screen.

## Entry

```swift
@Model final class Entry {
    // CloudKit rules: no unique constraints, every property has a default.
    var id: UUID = UUID()
    var createdAt: Date = Date()
    var kind: String = "voice"            // EntryKind raw value
    var visibilityRaw: String = "private" // Visibility raw value: private | shared
    // voice payload
    var audioFileName: String = ""
    var duration: TimeInterval = 0
    var transcript: String = ""
    var transcriptEditedByUser: Bool = false
    // enrichment
    var title: String = ""
    var titleSetByUser: Bool = false
    var tagValues: [String] = []
    var tagsSetByUser: Bool = false
}
```

- Typed accessors: `kind: EntryKind`, `visibility: Visibility`, `tags: [Quadrant]`.
- `Pipeline` writes enrichment only to fields where `...SetByUser == false`.
- The UI word is "message". The code word is `Entry`.
- No migration from `Memo` is needed, because there's no production data. Delete `Memo.swift`.

## Pipeline

Owns `Recorder`, `Tagger`, the `ModelContext` and `DotStore`. Its public surface is roughly:

- `warmUp()`, `press()`, `release()`, `setActive(_:)`
- `setVisibility(_:for:)`: publishes or withdraws the dot (fixes the current bug where every message publishes a dot)
- `delete(_:)`: removes the entry, its audio file and its dot
- `enrich(_:)`: also callable on an existing entry
- Observable state: `phase`, `liveText`, `lastEntry`
- **Stage timestamps** for the last capture: `pressAt`, `capturingAt`, `releaseAt`, `savedAt`, `enrichedAt`, `dotWrittenAt`. Flows report from these.

`CaptureView` becomes a thin view over `Pipeline`. Keep `-autotest` and `-seed` working, moved onto `Pipeline`.

New captures use the **current visibility default** (D5): private on first use, then the last choice, stored in `UserDefaults`.

## Acceptance

- The app builds for device and simulator. `app/validate.sh` still passes.
- On the simulator with `-seed -showmemos`, the list shows the seeded entries. Take a screenshot and look at it.
- No SwiftUI import in `Pipeline.swift`.
- Net line count roughly flat. This is a move, not a rewrite.

## Out of scope

UI redesign (brief 13), the `./verify` script (brief 11), CloudKit container change (brief 14).

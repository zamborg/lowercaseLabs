# Brief 11: `./verify` and pipeline flows on the Mac (rig A)

**Executor:** Claude (Sonnet). **Depends on:** 10. **Unblocks:** everything after; this is the loop itself.

## Goal

One command, `theVoidRebuild/verify`, that builds the app and runs the infra flows on the real pipeline on this Mac, then writes a report. After this, an agent can change pipeline code and know if it broke without a phone.

## Rig A, extended

`app/MacHarness` already compiles the app's `Transcriber.swift` and `Tagger.swift` for macOS and runs audio fixtures through them. Extend it to compile `Pipeline.swift`, `Entry.swift`, `Taxonomy.swift` and the storage code, and to drive `Pipeline` with:

- An **in-memory** `ModelContainer`.
- Audio from fixture files fed through the same buffer path the mic tap uses. Factor the tap handler in `Recorder` so the harness can call it with file buffers. That's an internal function, not a protocol (D12).
- `DotStore` in **record-only** mode when iCloud is off: it records what it *would* publish or withdraw, with timestamps. Use the existing `VoidICloud` switch. No fake CloudKit.

`Recorder` uses `AVAudioSession`, which is iOS-only. Keep the iOS-only parts behind `#if os(iOS)`, minimally.

## Flows in rig A

Put each in `app/MacHarness/flows/`, one file per flow, named by id. Each prints a single machine-readable result line:

```
FLOW F1 PASS release_to_saved_ms=42 ...
```

| Flow | Assertions | Measures |
|---|---|---|
| F1 record | Entry saved, audio file exists with the right duration, transcript WER ≤ 10% vs fixture reference | release → saved |
| F2 enrich | Title 1–60 chars, 1–2 distinct valid tags, written to the entry | release → enriched |
| F4 edit (pipeline half) | User-edited title and tags survive a second `enrich` | — |
| F6 share (local half) | Shared entry → dot publish recorded. Flip to private → withdraw recorded. Private capture → no publish | release → dot publish requested |
| F8 delete | Entry, audio file and dot gone (dot withdraw recorded) | — |

## `verify` script

- `./verify`: xcodegen, build the iOS app (device SDK; catches compile breaks), build and run MacHarness flows. Rig B flows get added in brief 12.
- `./verify F2`: one flow.
- `./verify --full`: adds the eval gate (brief 15) and CloudKit (brief 14) once they exist. Until then it says they were skipped.
- Writes `reports/<YYYY-MM-DD>-<short-sha>.md`: a flow table (id, pass/fail, measured, budget from [VISION.md](../VISION.md)), the harness's raw output, and a **"Not covered by this run"** section listing rigs that didn't run and why.
- Exits non-zero on any failure.
- `reports/` is committed. Keep the reports small: no audio, and screenshots as compressed PNG only.

## Acceptance

- `./verify` passes from a clean checkout (after `app/fetch-model.sh`) in under 10 minutes.
- Break something on purpose (for example, make the tagger return no tags) and confirm F2 fails with a clear message. Then revert.
- Commit a report from a green run.

## Note on timing

Mac timings are not phone timings. The report labels them "Mac (M-series)". Phone budgets are judged only by rig D.

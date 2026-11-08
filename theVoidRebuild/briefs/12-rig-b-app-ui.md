# Brief 12: UI flows in the real app (rig B, or B′ fallback)

**Executor:** Claude (Sonnet). **Depends on:** 10, 11. Rig B needs SETUP step 1. Rig B′ doesn't.

## Goal

Drive the actual app's UI with XCUITest so UI flows are verified and screenshotted without Zubin.

## Step 1: try rig B, the iOS app running on this Mac

Run the app as "My Mac (Designed for iPad)" under XCUITest. If it works, it has real Metal (MLX), real SpeechAnalyzer, and real SwiftData: full end to end with no fakes.

- Requires Xcode signed into Zubin's Apple ID (SETUP step 1), so the Mac can be registered and a signed build can run. Check first with `defaults read com.apple.dt.Xcode DVTDeveloperAccountManagerAppleIDLists`. If the list is empty, stop rig B and do B′.
- Add `-fixtureAudio <path>`. When it's set, `Recorder` feeds that file through the tap path instead of the mic (the same function rig A uses). Press and release still come from real UI gestures.
- **Time-box: half a day.** Record the outcome in [DECISIONS.md](../DECISIONS.md) either way.

## Step 2, if B fails: rig B′, the simulator

The simulator can't run SpeechAnalyzer or MLX. Add `-fixtureEnrichment <file>`, a JSON file of transcript, title and tags that `Pipeline` uses in place of live transcription and the tagger, **only** when the flag is set. This is the one place a result is substituted. Every report from B′ must say "enrichment replayed, not live."

## UI test target

- Add `TheVoidUITests` in `project.yml`.
- One test per flow, named by id. Each takes a screenshot of every screen it passes through, attached to the result. `verify` extracts them into `reports/`.
- The agent **looks at the screenshots** before calling a UI change done.

## Flows in rig B

F1, F2, F3, F4, F5 and F8, as defined in [AGENTS.md](../AGENTS.md). Write the ones whose UI exists now (F1, F2, the list half of F8). The rest are written with brief 13.

## Acceptance

- `./verify` runs rig B (or B′) flows after rig A, and the report includes screenshots.
- DECISIONS.md records whether B works, and if not, why.

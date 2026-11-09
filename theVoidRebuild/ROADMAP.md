# Road to the first TestFlight

Written 2026-11-08 from a read of the code, the phone and the Apple setup. [AGENTS.md](AGENTS.md) says how to think, [VISION.md](VISION.md) says what v1 is. This file says **in what order, what is risky, and what gates the ship**. It is sequencing and risk, not task briefs: whoever picks up a slice writes their own plan against the principles.

**Goal:** Zubin has The Void on his phone from TestFlight, no cable, and uses it as his journal.

## Now (2026-11-08 night): photo mode and settings

The first TestFlight build is on Zubin's phone (1.0, 2611081842). The slices below are history. What's next, in order:

1. ~~**Parts (D15), before anything visible.**~~ Done 2026-11-08. Reshape `Entry` so a message is voice, photo or both. Done when every existing flow (F1–F9 on the Mac and simulator) is green on the new shape and an entry with only a photo part round-trips through `Journal`, enrichment and dots. This is the step that has to land before the production schema (D19).
2. ~~**The camera (D16), F10–F12.**~~ Built; real-camera timing still to measure with `-cameratest` on the phone. Highest feel-risk first: the circle as shutter, and touch-and-hold flowing from snap to voice. Then "add photo". Enrichment text from scene labels (D17).
3. ~~**Settings (D18), F13.**~~ Built. Small. Can run alongside 2 if a second implementer is free; it shares no files with the camera except the timings it shows.
4. **Phone check, then a TestFlight build** on Zubin's go-ahead, then the production schema deploy (D19), also on his go-ahead.

| Risk | Why | What to do |
|---|---|---|
| Camera fights the mic | The mic engine listens on Capture; a capture session can take over the audio session and break voice capture or the touch-to-record budget | Camera session has no audio input and doesn't configure the audio session. F1 runs after a camera visit |
| Dual camera cost | Running both cameras is heavy on power and heat, and not every camera format supports it | Run the camera only while its page is showing. If both can't run at once, fall back to back then front in quick succession, and say so in the report |
| Snap on touch vs. hold | The same touch must be a photo-only tap or the start of a recording, decided after the fact | The photo is taken on touch in both cases, so nothing is guessed at touch time. Only the voice part depends on how long you hold |
| Rigs can't see a camera | Simulator and Mac have no dual camera | Fixture images at the camera edge for flows. Real camera latency and the two-image timing are measured on the cable-connected phone by a launch-argument test, not by Zubin |
| Photo text is thin | Scene labels say "beach", not how you felt | Photo-only tags will be weaker than voice tags. That's expected, and the ML loop owns improving it (D17). Never block capture on it |

## Where we actually were (2026-11-08 morning)

About 1,050 lines. Capture, live transcript and tagging work on the phone (touch to recording ~5 ms, tagging ~1.45 s Release). The Mac harness runs the real transcriber and tagger. Everything else in VISION is missing or untested:

- No `Entry` primitive: the model is still `Memo`, with no kind, visibility or edit provenance.
- Privacy bug: every message publishes a dot, with no visibility choice at all.
- No tab bar, Void card, Social grid, lock gesture, voice-reactive circle, or editing.
- Dots and invites have never run against real iCloud accounts. Invites are one-way; VISION says friendship is mutual.
- No `verify` command. The `--eval` gate in the Mac harness is half-written and unrun.
- The code's iCloud container (`iCloud.com.lowercaselabs.thevoid`) contradicts D11 (`iCloud.com.lowercaseLabs.void`). Fix before any schema exists.
- No app icon, no archive/upload path, no privacy manifest.

## The shape of the plan: one drop, full v1, on the phone first

Zubin's call (D13): the full v1 feature set, Social included, is built and working on his phone **before** TestFlight. TestFlight exists to distribute to friends, so it comes after v1 is tight. The cable is fine for getting v1 onto his phone now; the no-cable promise starts at TestFlight.

That moves the risky parts up: iCloud signing and the CloudKit sharing spike are on the critical path to the phone build, not deferred to a second drop. Start them early.

## Slices, in order

Each slice is done when its flows pass on a rig that doesn't need Zubin. Slices marked ∥ can run in parallel with the one above.

### 0. Zubin's ten minutes (start now, ∥ everything)
[SETUP.md](SETUP.md) steps 1 and 3: sign Xcode into the Apple ID; create the App Store Connect API key. Both are quick and both sit on the critical path later, so ask early, not at the end. Step 2 (two test iCloud accounts) is needed for slice 6 and is the slowest to arrange, so mention it now too. D10 (does TestFlight replace the old app?) only matters at slice 7; see risks.

### 1. Shape the data, then make it verifiable
- `Memo` becomes `Entry`: kind, visibility (private/shared), per-field provenance for title/tags, created-at. Do this **before** any CloudKit schema exists (D4).
- Pull capture → save → enrich → publish decision out of `CaptureModel` into something UI-free that flows can drive.
- Fix the privacy bug as part of the same change, with a flow that fails if a private entry ever produces a dot.
- Finish the `--eval` gate and write the baseline.
- One-command `verify` that runs the flows available on this Mac and writes a report ending in **what this run did not cover**.

Done when: F1, F2, F4 (model never overwrites a user edit) and the "private ⇒ no dot" invariant pass on the Mac, with a report.

### 2. Prove the UI rig (timeboxed spike, ∥ with 1)
The riskiest unknown for milestone 2: can we exercise the real iOS UI without Zubin? Candidates, from AGENTS.md: the iOS app running on the Mac under UI tests, or the Simulator with replayed enrichment (MLX and SpeechAnalyzer can't run there). Pick whichever works, record what it can and can't see, and make it part of `verify`. If neither is workable inside the timebox, say so plainly and fall back to Mac-harness flows plus screenshots; do not fall back to "send it to the phone."

Done when: one UI flow (launch → press → release → entry appears in the list) runs from `verify` and produces a screenshot someone looked at.

### 3. The journal UI
Order inside the slice is chosen to put the highest feel-risk first, while the capture code is fresh.
1. **Capture:** voice-reactive circle, slide-up lock (D6), visibility control that remembers (D5). The circle must keep its touch-to-draw-same-frame behavior; level metering must not add work to press.
2. **Tab bar + Void:** list, card, playback with scrubbing, edit transcript/title/tags, switch visibility, delete (F4, F5, F8).

Done when: F1–F5 and F8 pass; budgets hold on Release (touch to recording, release to saved under 300 ms, release to title/tags under 2 s); screenshots checked against the taste bar in AGENTS.md.

### 4. Ship pipeline (later, after the phone build; needs slice 0)
Everything TestFlight needs that isn't the app: app icon, display name, version/build numbering, privacy manifest and the App Store Connect privacy answers, export-compliance flag, archive and upload from the command line with the API key. Do this early with a throwaway build so the first real upload isn't the first time any of it is tried.

Done when: a build with the current app is uploaded and processed in App Store Connect.

### 5. Full-v1 gate
`verify` green on main, then build to Zubin's phone over the cable with iCloud on. This is the milestone he asked for.

### 6. Friends
- **Spike first:** one zone-wide share between two test accounts on CloudKit development. It answers friend-cap, mutuality and latency questions before UI is built on them. Needs slice 0 steps 1 and 2.
- Social tab: friends' dots over recent days; tap shows only tags and time; invite from here; names instead of opaque ids.
- Flows F6, F7, F9 across both accounts; measure shared-to-visible.
- **Deploy the CloudKit production schema.** TestFlight builds use the *production* environment, so dots and iCloud sync do nothing until the schema is deployed. Irreversible, so Zubin approves. Entry/Dot shape must be final by then.

Done when: F6, F7, F9 green on two accounts and friend cap measured.

### 7. TestFlight (later, for friends)
After v1 is tight on Zubin's phone: D10, the production schema deploy and the upload below. He approves each.

## ML loop (alongside, never blocking)
Title quality is weak ("Emotional State - Low Pleasant") and tag accuracy is 2/4 on a tiny sample. A model or prompt change is a separate asynchronous loop against the eval baseline from slice 1, and a good candidate for Codex through `~/.shepherd`. It ships only when it passes the flows and the 2 s budget. Zubin's labeled real messages (SETUP step 5) make it honest; until then synthetic scores overstate quality.

## Risks to watch

| Risk | Why it matters | What to do |
|---|---|---|
| **TestFlight replaces the old app (D10)** | Same bundle id means the old app's on-device data and store sit under the new app, and the new SwiftData schema may fail to open it. Zubin's old journal could be affected | Decide the bundle id before slice 4. Either a new App Store Connect record (new name needed), or confirm the new app opens cleanly over the old store with a copy of the old data. Zubin approves either way |
| **CloudKit production schema is one-way** | Wrong shape is permanent | Settle the Entry shape in slice 1; development environment only until slice 7; deploy to production only then |
| **Container id mismatch** | Code and D11 disagree | Pick one, change the code and entitlements together, before any schema is created |
| **Press-immediacy regressions** | The lock gesture and level metering both touch the press path, which is priority 2 | Budget measured on Release in `verify` for every capture change |
| **No UI rig** | Without one, "done" falls back to hoping | Slice 2 is a spike for this reason; escalate if it fails |
| **App size** | Bundled weights are 633 MB | Fine for TestFlight; expect slow uploads and a long first install |
| **Audio isn't synced** | Entries will sync to a second device with no audio there | Out of scope for single-device v1 (VISION); the card must handle a missing file gracefully |

## What Zubin decides, and when

| When | What |
|---|---|
| Now | Sign Xcode in (SETUP 1), needed for iCloud on the phone build. API key (SETUP 3) can wait until TestFlight |
| Before slice 4 | Bundle id for TestFlight and whether it replaces the old app (D10) |
| Before slice 6 | Create the two test iCloud accounts (SETUP 2) |
| At TestFlight | Bundle id / replacing the old app (D10), the CloudKit production schema deploy, the upload |

Nothing else should need him. Everything not on this list is the implementer's call.

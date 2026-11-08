# Road to the first TestFlight

Written 2026-11-08 from a read of the code, the phone and the Apple setup. [AGENTS.md](AGENTS.md) says how to think, [VISION.md](VISION.md) says what v1 is. This file says **in what order, what is risky, and what gates the ship**. It is sequencing and risk, not task briefs: whoever picks up a slice writes their own plan against the principles.

**Goal:** Zubin has The Void on his phone from TestFlight, no cable, and uses it as his journal.

## Where we actually are

About 1,050 lines. Capture, live transcript and tagging work on the phone (touch to recording ~5 ms, tagging ~1.45 s Release). The Mac harness runs the real transcriber and tagger. Everything else in VISION is missing or untested:

- No `Entry` primitive: the model is still `Memo`, with no kind, visibility or edit provenance.
- Privacy bug: every message publishes a dot, with no visibility choice at all.
- No tab bar, Void card, Social grid, lock gesture, voice-reactive circle, or editing.
- Dots and invites have never run against real iCloud accounts. Invites are one-way; VISION says friendship is mutual.
- No `verify` command. The `--eval` gate in the Mac harness is half-written and unrun.
- The code's iCloud container (`iCloud.com.lowercaselabs.thevoid`) contradicts D11 (`iCloud.com.lowercaseLabs.void`). Fix before any schema exists.
- No app icon, no archive/upload path, no privacy manifest.

## The shape of the plan: two drops, not one

**Drop 1 (TestFlight 1): the journal.** Capture and the Void, local-only, iCloud off. Milestones 1 and 2.
**Drop 2 (TestFlight 2): friends.** Social tab, dots, invites. Milestone 3.

Why split: the two drops have very different risk. Drop 1 is all ours to verify and needs no Apple ID work except signing and upload. Drop 2 depends on CloudKit behaving between two real accounts, which nobody has seen yet. Splitting gets Zubin the app on his phone, uploaded without a cable, far sooner. It also teaches us the TestFlight mechanics (signing, icon, upload, export compliance) while the stakes are low, and gives him real journal use to judge taste before friends are layered on. Recorded as D13. If Zubin wants one drop, the work is identical; only the ship date moves.

## Slices, in order

Each slice is done when its flows pass on a rig that doesn't need Zubin. Slices marked ∥ can run in parallel with the one above.

### 0. Zubin's ten minutes (start now, ∥ everything)
[SETUP.md](SETUP.md) steps 1 and 3: sign Xcode into the Apple ID; create the App Store Connect API key. Both are quick and both sit on the critical path later, so ask early, not at the end. Step 2 (two test iCloud accounts) isn't needed until drop 2 but is the slowest to arrange, so mention it now too. D10 (does TestFlight replace the old app?) needs his answer before slice 6; see risks.

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

### 4. Ship pipeline (∥ with 3; needs slice 0)
Everything TestFlight needs that isn't the app: app icon, display name, version/build numbering, privacy manifest and the App Store Connect privacy answers, export-compliance flag, archive and upload from the command line with the API key. Do this early with a throwaway build so the first real upload isn't the first time any of it is tried.

Done when: a build with the current app is uploaded and processed in App Store Connect.

### 5. Drop 1 gate
`verify` green on main. Zubin approves the first ship (D10), then upload. After this, TestFlight is the delivery route; no more cable.

### 6. Friends (drop 2)
- **Spike first:** one zone-wide share between two test accounts on CloudKit development. It answers friend-cap, mutuality and latency questions before UI is built on them. Needs slice 0 steps 1 and 2.
- Social tab: friends' dots over recent days; tap shows only tags and time; invite from here; names instead of opaque ids.
- Flows F6, F7, F9 across both accounts; measure shared-to-visible.
- **Deploy the CloudKit production schema.** TestFlight builds use the *production* environment, so dots and iCloud sync do nothing until the schema is deployed. Irreversible, so Zubin approves. Entry/Dot shape must be final by then.

Done when: F6, F7, F9 green on two accounts, friend cap measured, then TestFlight 2.

## ML loop (alongside, never blocking)
Title quality is weak ("Emotional State - Low Pleasant") and tag accuracy is 2/4 on a tiny sample. A model or prompt change is a separate asynchronous loop against the eval baseline from slice 1, and a good candidate for Codex through `~/.shepherd`. It ships only when it passes the flows and the 2 s budget. Zubin's labeled real messages (SETUP step 5) make it honest; until then synthetic scores overstate quality.

## Risks to watch

| Risk | Why it matters | What to do |
|---|---|---|
| **TestFlight replaces the old app (D10)** | Same bundle id means the old app's on-device data and store sit under the new app, and the new SwiftData schema may fail to open it. Zubin's old journal could be affected | Decide the bundle id before slice 4. Either a new App Store Connect record (new name needed), or confirm the new app opens cleanly over the old store with a copy of the old data. Zubin approves either way |
| **CloudKit production schema is one-way** | Wrong shape is permanent | Ship drop 1 with iCloud off; settle the Entry shape; deploy only in slice 6 |
| **Container id mismatch** | Code and D11 disagree | Pick one, change the code and entitlements together, before any schema is created |
| **Press-immediacy regressions** | The lock gesture and level metering both touch the press path, which is priority 2 | Budget measured on Release in `verify` for every capture change |
| **No UI rig** | Without one, "done" falls back to hoping | Slice 2 is a spike for this reason; escalate if it fails |
| **App size** | Bundled weights are 633 MB | Fine for TestFlight; expect slow uploads and a long first install |
| **Audio isn't synced** | Entries will sync to a second device with no audio there | Out of scope for single-device v1 (VISION); the card must handle a missing file gracefully |

## What Zubin decides, and when

| When | What |
|---|---|
| Now | Sign Xcode in; create the API key (SETUP 1 and 3). Reply if he'd rather ship one drop than two |
| Before slice 4 | Bundle id for TestFlight and whether it replaces the old app (D10) |
| Before drop 2 | Create the two test iCloud accounts (SETUP 2) |
| At each ship | Approve the upload. For drop 2, also approve the CloudKit production schema deploy |

Nothing else should need him. Everything not on this list is the implementer's call.

# Working on The Void

For any coding agent (Claude Code, Codex, others) picking up work here. Read [VISION.md](VISION.md) first for what we're building. This file is how we build it.

## The one rule

**You verify your own work.** Zubin sends a request (a feature, a bug, a question) from wherever he is and does not run builds, hold the phone, or relay what he saw. You change the code, prove the affected user flows work by running them yourself, and come back with a report in the terms he asked in. If you can't verify something yourself, say so plainly. Never imply it was verified.

Until the app is on TestFlight, nobody pushes builds to Zubin's phone as part of the loop.

## The three layers

| Layer | What it is | How it's verified | Cadence |
|---|---|---|---|
| **UI** | Screens, gestures, navigation | Flows driven through the real app UI (rig B) | With every change |
| **Infra** | Capture, storage, enrichment pipeline, sync, dots | Flows on the real pipeline with fixture audio (rigs A, B, C) | With every change |
| **ML** | Transcription quality, title quality, tag accuracy, model choice | Eval sets with stored baselines (rig E) | Its own ongoing loop, asynchronous |

UI and infra changes are judged by flows. ML changes are judged by evals. A change to the ML layer still has to keep every flow passing.

## Flows: the unit of verification

A **flow** is a user-level scenario with assertions and measurements, written the way Zubin would ask about it ("when I record a message and share it, how long until a friend sees the dot?"). Unit tests are welcome where they give you confidence, but the flows are what you report on.

Each flow has an id, steps, what must be true, and what gets measured against [VISION.md budgets](VISION.md#budgets).

| ID | Flow | Must be true | Measures |
|---|---|---|---|
| F1 | **Record a message.** Launch, hold, speak, release | Message saved with audio and a transcript matching the speech | Launch to ready; touch to recording; release to saved |
| F2 | **Enrichment.** After F1 | Non-empty title of 60 characters or fewer, 1–2 distinct valid tags, shown on Capture and in the Void list | Release to title and tags shown |
| F3 | **Locked recording.** Hold, slide off, let go, keep talking, tap to stop | Recording continued after the finger lifted; one message saved | Same as F1 |
| F4 | **Edit.** Open a card, edit transcript, title and tags | Edits persist across relaunch; a later enrichment run doesn't overwrite them | — |
| F5 | **Playback.** Open a card, play | Audio plays; duration matches the recording | Tap to audio |
| F6 | **Share.** Record as shared; flip one to private; flip back | Dot exists exactly while the message is shared; no content in the shared zone | Release to dot written |
| F7 | **Friend sees it.** Second account, after F6 | Friend's Social tab shows the dot with tags and time only | Dot written to dot visible |
| F8 | **Delete.** Delete a shared message | Message, audio file and dot are all gone | — |
| F9 | **Invite.** Invite a friend, friend accepts | Both see each other in Social | — |

Flow definitions live in `flows/` as test code, one file per flow, named by id. When behavior changes, change the flow in the same commit. **Never weaken an assertion or loosen a budget to make a flow pass.** If a budget is wrong, say so in the report and let Zubin decide.

A new module adds its own flows (for a camera module: swipe to camera, snap, post, see it in the Void, share). They follow the same shape.

## Rigs: where flows run

Each rig runs the real code with as little substitution as possible. **No engine protocols.** The only test seams are launch arguments at the edges of the app, for example `-fixtureAudio <file>`, which feeds a WAV file into the recorder's tap in place of the mic.

| Rig | What it is | Real | Substituted | Covers | Status |
|---|---|---|---|---|---|
| **A. Mac pipeline** | `app/MacHarness`, a Mac command-line tool compiled from the app's own `Transcriber.swift` and `Tagger.swift` | Speech model, LLM, Metal | Mic → audio fixtures | Infra and ML truth: transcripts, enrichment output and timing, on Mac hardware | **Exists**, passes (`app/validate.sh`). Doesn't cover storage or sharing yet |
| **B. App on Mac** | The actual iOS app running on this Mac ("Designed for iPad"), driven by XCUITest | Everything: UI, SwiftData, speech, LLM on Metal | Mic → `-fixtureAudio` | UI + infra flows end to end: F1–F5, F8 | **Not built.** Needs Xcode signed into Zubin's Apple ID so the Mac can run a signed build. Unproven that MLX and SpeechAnalyzer behave there the way they do on the phone; validating that is the first job |
| **B′. Simulator** | iOS simulator with XCUITest | UI, SwiftData | Mic, and also speech and LLM (neither runs in the simulator: no speech model, and MLX crashes without Metal) | UI-only flows and screenshots at phone size | Fallback if B fails. Weaker, because it can't see real enrichment |
| **C. CloudKit** | Two test iCloud accounts against the CloudKit **development** environment: one in rig B, one in a simulator | CloudKit | — | F6, F7, F9 and share timing | **Not built.** Needs Zubin to create two test Apple IDs once. Agents never create accounts |
| **D. Phone** | Zubin's iPhone 14 Pro over `devicectl`, scripted with `-autotest` | Everything, including the real mic and A16 | — | Only what no other rig can see: on-device latency budgets, memory, heat | Exists, manual. Needs the phone connected and unlocked. Run on request or before a release, never as part of the default loop |
| **E. Evals** | `eval/` (Python and Swift) plus rig A's `--eval` gate | Models | Real voice → synthetic and labeled text | ML quality: WER, top-1 tag accuracy, title quality, per-memo latency | Synthetic evals exist. The rig A gate is written but **never run**. Real holdout data isn't collected yet |

## Commands

One entry point, run from `theVoidRebuild/`:

```bash
./verify            # build, then rig A and rig B flows. The default before any commit. Target: under 10 minutes
./verify --full     # adds rig C (CloudKit) and rig E (eval gates)
./verify --device   # adds rig D on the phone, if connected and unlocked
./verify F6         # run one flow
```

Every run exits non-zero on any failure and writes `reports/<date>-<short-sha>.md`: the flow table (pass/fail, measured vs budget), screenshots of each screen the flows touched, and **a section listing what this run did not cover**.

`./verify` doesn't exist yet. Building it is workstream 1 below.

## The loop for a request

1. **Restate the request as flows.** Which existing flows does it touch? Does it need a new one? If the request is ambiguous about product behavior, ask before building. That's the one place to involve Zubin early.
2. **Write or change the flow first** where practical, and watch it fail.
3. **Implement.** Keep the codebase small. No new protocols. No new dependencies without saying why.
4. **Run `./verify`** until green. For ML changes, also the eval gate (below).
5. **Look at the screenshots yourself** for any UI change. A passing flow with an ugly or broken screen is not done.
6. **Commit** with the report path in the message. Open a PR if asked.
7. **Report back** in Zubin's terms: what changed, the flow results and timings that answer his question, screenshots, and what wasn't verified and why.
8. **Ship.** After TestFlight is set up: a green `./verify --full` on `main` goes out as a TestFlight build via `./ship`. Before then: stop at step 7.

### Done means

- Every flow touched by the change passes, and no other flow regressed.
- Budgets met, or misses called out with numbers.
- Screenshots checked by you for any UI change.
- The report says honestly what wasn't covered.

### Ask Zubin first when

- A product decision isn't settled by [VISION.md](VISION.md).
- A change needs new permissions, entitlements, an Apple account step, or money.
- A change would touch real (non-test) iCloud data, or deploy a CloudKit schema to production.
- A budget looks wrong.

Everything else: decide, do it, and report.

## The ML loop

ML improves on its own schedule, usually by a Codex workstream, without blocking app work.

**The contract** between ML and the app is `Tagger.tag(transcript:) -> (title, tags)`: a title of 1–60 characters, and 1–2 distinct tags from the four quadrants. The app ships whatever model directory and prompt `Tagger.swift` points at.

**A change lands only if:**

- Top-1 tag accuracy on the labeled set (`eval/data/memos.jsonl`, clean split, plus the real holdout once it exists) is at or above the stored baseline, minus 2 points of noise.
- Mac latency per message stays within budget, and phone latency is re-measured (rig D) before shipping if the model changed.
- Every flow still passes.
- The baseline file is updated in the same commit.

Title quality has no automated score yet. Until it does, a title change includes a side-by-side of before and after for 20 real messages in the report.

**Data Zubin provides when he can:** about 50 hand-labeled real messages (the holdout), and about 20 recordings with corrected transcripts for WER. Synthetic data, generated and judged by models from the same vendor, overstates accuracy, so treat it as a regression check, not a quality claim.

## Bootstrap: making this real

Ordered. Each one makes the loop more autonomous.

1. **`./verify` and rig A flows.** Extract the capture → save → enrich → publish pipeline out of `CaptureView`'s model into a plain object the UI observes, so it can run without a screen. Extend rig A to run F1, F2, F6 (up to the CloudKit call) and F8 against an in-memory store with fixture audio. Write the report generator.
2. **Rig B.** Add `-fixtureAudio`. Get the app running on the Mac under XCUITest with real speech and LLM. If it can't be made to work, build rig B′ instead and record why in `DECISIONS.md`. *Needs: Xcode signed into Zubin's Apple ID.*
3. **v1 UI per [VISION.md](VISION.md)** (tab bar, lock gesture, Void card, Social grid, visibility, editing) with flows F3–F5 and F8 written alongside, not after.
4. **Data model to v1 shape**: `Entry` with kind, visibility and edit provenance, before any real data or CloudKit schema exists in production.
5. **Rig C**: dots end to end with two test accounts, measuring F6, F7 and F9. *Needs: two test Apple IDs from Zubin.*
6. **Rig E gate**: run the eval gate, store the first baseline, and hand the ML loop to Codex.
7. **TestFlight and `./ship`.** *Needs: an App Store Connect app record and an API key from Zubin.* After this, a green `main` goes to Zubin's phone without anyone touching it.

## Practicalities

- **Project**: XcodeGen. Edit `app/project.yml`, never the generated `.xcodeproj`. Build with `-skipMacroValidation -skipPackagePluginValidation`.
- **Model weights**: `app/Model/` is gitignored. `app/fetch-model.sh` restores it from the pinned snapshot.
- **iCloud** is off in builds until signing is set up: `VoidICloud` in `project.yml`.
- **Simulators** can't run MLX or SpeechAnalyzer. Don't spend time debugging either there.
- **Codex**: delegate deterministic work through `~/.shepherd` (persistent sessions and workstreams). Write a brief, treat Codex as an engineer, and review its output against the flows.

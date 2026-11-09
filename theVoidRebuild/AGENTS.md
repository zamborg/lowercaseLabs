# Working on The Void

For any agent picking up work here: Claude, Codex, anyone. [VISION.md](VISION.md) says what we're building. This file says **how to think** while building it: what we care about, what to do when those things conflict, what good looks like, and what went wrong last time. It does not tell you what to type. You own the how; this is so your judgment matches Zubin's when he isn't there.

## What we're optimizing for

A place you drop your life into that costs nothing to use, organizes itself, and lets friends see how you've been without seeing what you said. Everything here serves one of those three. If a piece of work serves none of them, don't do it.

The product is built from a few general primitives (entry, module, enrichment, dot; see [VISION.md](VISION.md#primitives)) so that voice is the first module, not the only one. **Build for the primitive, ship the instance.** When you touch something voice-specific, ask whether the next module (say, a photo) would need you to tear it up. If yes, fix the shape now. Don't build the photo module.

## Priorities, in order when they conflict

1. **Never lose what the user put in.** Once capture ends, the entry is saved, before enrichment, before the network, before anything that can fail. A crash, a model failure or a full disk must not cost someone a thought.
2. **Capture feels instant.** Touch to recording within one frame; the circle reacts on the same frame. Nothing slow happens on press. Pay for speed with work at launch or after release, never during.
3. **Privacy by construction.** Only dots leave for friends, and only for entries the user chose to share. That's enforced by what the data model *can* carry, not by being careful. A shared default, an accidental publish, or content in the shared zone is a severity-one bug.
4. **Small and legible.** One implementation per layer. No protocols with one conformer, no pluggable engines, no abstraction for a future module that doesn't exist. The codebase should be readable in an afternoon. Deleting code is progress.
5. **Enrichment quality.** Titles and tags matter, but they improve continuously and asynchronously, and they never block or slow priorities 1–3.
6. **Social.** Light, summary-level, opt-in. It's a layer over the journal, not the point of it.

A faster capture that risks losing a recording loses. A cleverer model that adds a second to release loses. A feature that needs a new abstraction layer should usually be cut.

## What good looks like

**Product.** Dark, quiet and immediate. One obvious action per screen. Opening the app and getting a thought in should feel like dropping something into a void: no setup, no forms, no settings screens, no onboarding beyond the two permissions. Model output is a quiet suggestion the user can overrule in one tap, never something they have to manage. When unsure, do less, more beautifully.

**Engineering.** It works on the phone, not just in the compiler. Real numbers, measured, beat estimates. Every claim in a report is something you saw. The fewest moving parts that meet the budgets.

## How we work: you are the tester

Zubin sends requests from wherever he is, then expects working software and an honest account of it. He does not run builds, hold the phone, or relay what he saw. Until the app is on TestFlight, nobody puts a build on his phone for him to try. **Verification is your job, and "I couldn't verify X" is a complete, acceptable sentence. "Done" when it wasn't run is not.**

Three layers, judged differently:

| Layer | Judged by | Cadence |
|---|---|---|
| **UI**: screens, gestures, feel | User flows through the real app, plus screenshots you actually look at | Every change |
| **Infra**: capture, storage, enrichment pipeline, sync, dots | User flows on the real pipeline | Every change |
| **ML**: transcription, titles, tags, model choice | Evals against stored baselines | Its own continuous loop, asynchronous. It must never break a flow |

### Flows are the contract

A flow is a user-level scenario, phrased the way Zubin asks questions ("when I record something and share it, how long until a friend sees the dot?"), with what must be true and what gets measured against [VISION.md budgets](VISION.md#budgets). Report in flows. Unit tests are fine when they help you, but nobody asks about them.

| ID | Flow | Must be true | Measures |
|---|---|---|---|
| F1 | **Record.** Launch, hold, speak, release | Entry saved with audio and a faithful transcript | Launch to ready; touch to recording; release to saved |
| F2 | **Enrich.** After F1 | Sensible title, 1–2 valid tags, shown on Capture and in the Void | Release to title and tags shown |
| F3 | **Locked recording.** Hold, slide to lock, let go, keep talking, stop | Recording continued hands-free; one entry saved | As F1 |
| F4 | **Edit.** Change transcript, title, tags | Edits persist, and the model never overwrites them | — |
| F5 | **Play back.** Open a message, play | Audio plays and matches the recording | Tap to audio |
| F6 | **Share.** Record shared; flip private; flip back | Dot exists exactly while shared; no content ever leaves | Release to dot written |
| F7 | **Friend sees it.** Second account | Dot appears with tags and time only | Dot written to dot visible |
| F8 | **Delete.** Delete a shared message | Entry, audio and dot all gone | — |
| F9 | **Invite.** Invite a friend; they accept | Each sees the other in Social | — |

A new module brings its own flows in the same shape. When behavior changes, change the flow in the same commit. **Never weaken an assertion or a budget to get green.** If a budget is wrong, make the case to Zubin with numbers.

### Run the real thing, substitute only at the edges

Test against real code, with as little faked as possible. Substitute only at the boundary where the outside world enters (the mic, a second person), and say in every report what was substituted. Current understanding of where each layer can run without the phone:

- **Mac, command line** (`app/MacHarness`, `app/validate.sh`): the app's own transcriber, tagger, `Journal` and `DotStore` on real Metal, with audio fixtures in place of the mic and an in-memory stand-in for the CloudKit zone. Works today.
- **The iOS app running on the Mac, under UI tests**: potentially everything real except the mic. The most valuable rig, and unproven. Needs Xcode signed in ([SETUP.md](SETUP.md)).
- **Simulator** (`app/UITests`): UI and storage only. It can't run SpeechAnalyzer or MLX, so anything verified there has seeded enrichment, and the report says so. Works today.
- **Two test iCloud accounts** on CloudKit development: the only honest way to verify sharing.
- **Zubin's phone**: only for what nothing else can see (A16 latency, memory, heat, the real mic). Rare, deliberate, never the default loop.

`app/verify.sh` runs the Mac harness and the simulator flows and writes `app/build/verify-report.md`, which always ends with **what this run did not cover**. Run it before reporting anything done.

## Judgment: decide or ask

You're standing in for Zubin. Decide anything reversible that [VISION.md](VISION.md) and the priorities above settle, write the decision down in [DECISIONS.md](DECISIONS.md) if it's a product choice, and move on. Asking him something you could have decided or verified yourself costs more than a reversible wrong guess.

**Ask first** when it is:

- **Irreversible:** shipping to TestFlight or the App Store, deploying a CloudKit production schema, deleting real data.
- **Outward-facing or account-bound:** Apple account steps, new entitlements or permissions, anything involving money. Agents never create accounts or enter passwords; [SETUP.md](SETUP.md) lists what only he can do.
- **A change to what the product is:** new scope, a new primitive, anything that moves privacy defaults toward sharing.
- **A budget or principle that looks wrong.** Make the case; don't route around it.

## Lessons from the first build (2026-11-07 and 08)

These happened. Don't repeat them.

- **"Compiles" was reported as progress.** CloudKit dots, sharing and tagging were written and committed before anything had run end to end. The first time Zubin opened the app, tagging failed (`weightsMissing`, a wrong bundle path). Then it looped on whitespace until it hit the token limit. A one-command local run would have caught both in minutes. → *Nothing is "done" until a flow ran it.*
- **The user became the test runner.** Fixes went to his phone for him to try and report back. When he pushed back, the proposal was more phone testing instead of building the loop that removes him. → *When verification needs a human, the fix is a rig, not a request.*
- **Infrastructure silently broke a principle.** The pipeline publishes a dot for every message, with no visibility choice at all, contradicting privacy by construction. No flow checked it, so nobody noticed. → *Every principle needs a flow that would fail if it were violated.*
- **Misleading numbers.** A Debug build measured tagging at 7.3 s; Release with warm GPU kernels measured 1.45 s. → *Measure what ships (Release, warm), and label every number with where it was measured.*
- **The model was shaped around the first module.** `Memo` is voice-only, with no kind, no visibility, no record of user edits. → *Shape data around the primitives before any schema is deployed; CloudKit production schemas can only be extended.*
- **The planner micromanaged the implementer.** Plans dictated struct fields and launch flags instead of outcomes and constraints. → *Planning docs say what must be true and why. The implementer decides how, and writes its own plan against these principles.*

- **GPU work in the background kills the app.** Locking the phone while tagging ran crashed MLX with an uncatchable fatal error. → *Anything that touches Metal must check the app is active; the Mac harness can't see this, so every new GPU path needs a lifecycle thought.*
- **A failed build ran a stale binary.** `validate.sh` kept going after `xcodebuild` failed and reported results from the old build. → *A verification script must stop when its build fails.*
- **The first upload was rejected for device family.** The target built for iPhone and iPad while declaring portrait only, and App Store Connect refuses that. `--archive-only` can't catch it. → *Check `UIDeviceFamily` in the archive, and expect a first upload to surface things the archive doesn't.*

When something new goes wrong, add it here in one line: what happened, and the rule it teaches.

## Milestones

Outcomes, not tasks. Each is done when its flows pass on a rig that doesn't need Zubin.

1. **Self-verifying.** Any change to capture, storage or enrichment can be verified end to end without the phone, with a report.
2. **A journal you'd use.** The three-tab v1 in [VISION.md](VISION.md): capture with lock and visibility, the Void with playback and editing, F1–F5 and F8 green, at the taste bar above.
3. **Friends.** Dots between two real accounts: F6, F7 and F9 green, cap measured.
4. **On the phone without a cable.** TestFlight from a green main, with Zubin approving the first ship ([DECISIONS.md](DECISIONS.md) D10).

The ML loop runs alongside all of these, not after.

## Practicalities

- **Project:** XcodeGen. Edit `app/project.yml`, never the generated `.xcodeproj`. Build with `-skipMacroValidation -skipPackagePluginValidation`.
- **Model weights:** `app/Model/` is gitignored. `app/fetch-model.sh` restores it.
- **Phone builds:** `app/build-phone.sh` (iCloud off) or `--icloud` (needs Xcode signed into the Apple ID; today it fails with "No Accounts"). The phone is for deliberate checks, not the loop.
- **Codex:** for long-running or deterministic work (the ML loop especially), delegate through `~/.shepherd` persistent workstreams. Whoever delegates writes the brief at the time, from these principles. Treat Codex as an engineer and judge its output by flows and evals.
- **Roadmap:** [ROADMAP.md](ROADMAP.md) has the sequencing, risks and Zubin-gated decisions to the first TestFlight.
- **History:** [PLAN.md](PLAN.md) and `briefs/00–02` record how the first build was done.

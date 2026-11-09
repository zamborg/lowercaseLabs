# The Void — Vision

Owner: Zubin. Last rewritten: 2026-11-08. This is the product source of truth. Engineering workflow lives in [AGENTS.md](AGENTS.md).

## What it is

The Void is a place to dump your life into. You open it, you drop something in, and it's kept. Today that "something" is your voice. Over time the Void becomes a perpetual record of what was going on and how you felt about it.

Three things make it work:

1. **Ingest that costs nothing.** Opening the app and getting a thought in has to be faster and easier than not bothering. Capture speed beats every other concern.
2. **Organization you never do yourself.** A model titles and tags everything you drop in. You can correct it, but you never have to.
3. **Light social on metadata, never content.** Friends see how you've been feeling (your tags, as dots over time), never what you said.

It is a personal journal first. The social layer is summary statistics over that journal.

## Primitives

These are the building blocks. Features are compositions of them. If a new feature needs a new primitive, that is a design conversation, not an implementation detail.

### Entry

One thing dropped into the Void. In the UI it's called a **message**.

- **Parts**: the raw content, one or more parts, each made by a module. v1 has two modules, so an entry is a **voice** part (audio and transcript), a **photo** part (a back-camera and front-camera image taken together), or both. Parts are the user's and are never shared. An entry always has at least one part; discarding a snap (the ✕) deletes the whole message.
- **Enrichment**: derived metadata, currently a **title** and **tags**. Produced by a model after the entry is saved. Every enriched field remembers whether the model or the user set it. A user edit always wins and is never overwritten by a later model run.
- **Visibility**: `private` or `shared`. Chosen at capture, changeable any time after.
- **Created at**: when it was captured.

Saving never waits on enrichment. An entry exists the instant capture ends, even if the model is slow or fails.

### Module

A way to make entries. A module provides exactly three things:

1. A **capture surface**: a page you can reach from the home screen.
2. A **part** shape: what gets stored.
3. **Text for enrichment**: how its part becomes text the models can read. Voice provides its transcript. Photo provides on-device scene labels for both images (D17). When an entry has both, the transcript leads.

Enrichment, the Void list, visibility and sharing don't know or care which module made an entry. That is what makes the next module cheap. Photo is the proof: if adding it means changing enrichment, dots or sharing beyond "an entry may have this part", the shape was wrong. Fix the shape, don't special-case photo.

### Tags

Exactly the four How We Feel quadrants: high-energy pleasant (yellow), low-energy pleasant (green), high-energy unpleasant (red) and low-energy unpleasant (blue). Each entry gets one dominant quadrant and optionally a second. Tags are a fixed enum ([taxonomy.json](taxonomy.json)), never free text, so they stay comparable across people and time.

### Dot

The only thing that leaves your device for friends: who, when, and which tags, for one shared entry. A dot exists exactly while its entry is shared. If the entry is made private or deleted, the dot is withdrawn.

### Friend

A mutual connection made by invitation, built on iCloud sharing. The friend list is small by design (cap around 30, unconfirmed). There are no public profiles, followers or feeds.

### Identity

Your iCloud account. There is no sign-up and no sign-in screen. First launch asks for microphone and speech permission, and that's it. Camera permission is asked the first time you open the camera, not before.

## V1: what you see

A tab bar with three tabs.

### 1. Capture (home, the app opens here)

- A black screen with one circle. The circle reacts to your voice level while it listens.
- **Press and hold** to record. Recording starts on touch, with no perceptible delay.
- **Slide off** the circle while holding to **lock** the recording, so you can let go and walk around with the phone out. Tap to stop.
- The transcript streams in live while you talk.
- On release, the message is saved immediately. The title and tag dots appear a moment later.
- You choose **private or shared** for this message. The control is right there and remembers your last choice.

#### The camera (photo mode)

Swipe left from the circle to reach the camera. A small camera glyph on the Capture screen shows it's there. Swipe right to go back.

- The back camera fills the screen, with the front camera inset, both live.
- **The same circle is the shutter.** One gesture does both modes:
  - **Tap:** both cameras snap at the moment of touch. That's a photo-only message, saved at once.
  - **Touch and keep holding:** the photo snaps on touch, then you keep holding to talk about it. Voice records into the **same** message. Slide up to lock, as on Capture.
- After a snap the two images stay on screen with the circle, the visibility control and an ✕ to discard. Discarding deletes the message; it's the only way a snap is lost, and it's explicit.
- **Adding a photo to a voice message:** right after you record on Capture, and on any message's card in the Void, "add photo" opens the camera, and the snap joins that message instead of starting a new one.
- Photos stay in the app. They aren't saved to your photo library, and location and other metadata are stripped.

### 2. Void (your messages)

- Every message, newest first: title, tag dots, date, length, and whether it's shared.
- Tap one to open its **card**:
  - Play back the audio, with scrubbing.
  - Read and edit the transcript.
  - Edit the title and tags.
  - Switch private or shared.
  - Delete.

### 3. Social (your friends' dots)

- Your friends' dots laid out over recent days, one row per friend, so you can see at a glance that a friend has been mostly blue this week.
- Tapping a dot shows only its tags and time. Never a title or transcript.
- Invite a friend from here.

There is no "my dots" tab in v1. The Void tab already shows your own history.

### Settings (one small panel)

A gear in the Void tab's header opens a sheet. There's no fourth tab. The rule: **the app is complete without ever opening it.** Every setting has a default that is already right, nothing in it is needed to capture, and it holds things you control, not things you have to manage. Settings stay on this device and don't sync.

v1 holds (D18):

- **Camera:** front and back (default), or back only.
- **Show timings:** the capture and enrichment timings on screen, for checking budgets on TestFlight builds, which can't take launch arguments.
- **Re-tag my messages:** runs enrichment again over every title and tag the model set. Never touches ones you edited.
- **About:** version and build, and whether iCloud and dots are on (and if not, why). Read-only.

A new setting needs a reason Zubin would recognise: he asked to control it, or it's needed to diagnose a build. "Someone might want this" isn't one.

## Budgets

These are product requirements, not aspirations. Each one is measured by a flow in [AGENTS.md](AGENTS.md).

| What | Budget | Measured on |
|---|---|---|
| Touch to recording | Within 1 frame (16 ms) | iPhone 14 Pro |
| Release to message saved | Under 300 ms | iPhone 14 Pro |
| Release to title and tags shown | Under 2 s | iPhone 14 Pro |
| App launch to ready to record | Under 1 s | iPhone 14 Pro |
| Arrive on camera to both cameras live | Under 500 ms | iPhone 14 Pro |
| Touch to photo captured (both images) | The frame at touch is the one kept; both images within 100 ms of each other | iPhone 14 Pro |
| Touch to photo message saved | Under 300 ms | iPhone 14 Pro |
| Photo-only message to title and tags shown | Under 2 s | iPhone 14 Pro |
| Back from camera to voice recording on touch | Unchanged: within 1 frame | iPhone 14 Pro |
| Shared message to dot visible to a friend | Next time they open Social (target under 10 s when both are online) | Two iCloud accounts |

Minimum device: iPhone 14 Pro (A16). iOS 26.

## Technical commitments

These hold until we decide otherwise, on purpose.

- **Everything runs on device.** Speech is Apple's SpeechAnalyzer. Titling and tagging use a local LLM (currently LFM2-1.2B, 4-bit, via MLX Swift, with JSON-schema-constrained output). No cloud models.
- **Local-first.** SwiftData on device. Your messages sync to your own iCloud private database. Dots live in a shared CloudKit zone.
- **Privacy by construction.** Only dots leave for friends. Nothing in the data model allows content into the shared zone.
- **One implementation per layer.** No pluggable engines and no protocols with one conformer. Test seams are launch arguments at the edges (see [AGENTS.md](AGENTS.md)).
- **A small codebase.** The previous app reached 12k lines with no clear shape. Keep this one small enough to read in an afternoon.

## Not in v1

- A third module. Photo is the second; designed 2026-11-08, see Capture.
- More than one photo per message, video, or picking from the photo library.
- Syncing audio or photos to your other devices (only text and metadata sync).
- A "my dots" tab, reactions, comments, push notifications for dots.
- Android, iPad, Mac apps; web; public feeds; cloud LLMs.
- Accounts beyond iCloud.

## Where the code is versus this doc (2026-11-08, end of day)

| Area | State |
|---|---|
| Capture: hold, live transcript, save | Working on iPhone 14 Pro: touch to recording ~5 ms, release to final transcript ~35-90 ms |
| Voice-reactive circle, slide-up lock, visibility toggle | Built and compiled; screen checked in the simulator. **Not yet felt or measured on the phone** (needs the mic) |
| Enrichment: title and tags | ~1.25-1.5 s on the phone (Release). Title quality weak, tag accuracy unproven on real messages. Stops when the app isn't active and resumes on return |
| Data model | `Entry` with kind, visibility, per-field provenance. Store is named `Entries`, separate from the first build's `Memo` store |
| Visibility and dots | Dots are derived from shared, tagged entries and reconciled after every change. Verified against an in-memory cloud |
| Void tab: list, card, playback, edit, delete | Built. List, edit, visibility and delete verified by UI flows in the simulator; **playback sound is unverified** |
| Social tab | Built: friends' dots over the week, tap for tags and time, invite, friend cap 30. Screen checked with seeded friends. **Never run against real iCloud** |
| Mutual friendship | Accepting an invite shows a prompt to send yours back; a single link can't auto-reciprocate |
| iCloud on the phone | Working. Signed with the iCloud entitlement; a `-cloudtest` run on the phone published a dot, withdrew it on flip to private, re-published, and removed it on delete, against the real CloudKit development zone (first publish ~2 s incl. zone creation, withdraw ~0.6 s). **Friends seeing each other still needs two accounts** |
| Audio sync across your devices | Missing. Only text syncs. Not needed for single-device v1 |
| TestFlight | Build 1.0 (2611081842) uploaded and installed by Zubin. CloudKit production schema **not deployed**, so dots and sync do nothing in TestFlight builds yet |
| Photo mode | Built (2026-11-08): `Entry` is voice/photo/both, camera page (swipe or glyph), circle as shutter, add photo from Capture and the Void card, scene-label enrichment, metadata stripped. **Verified**: Mac harness (save ~143 ms with 2 images, strip, labels, voice joins, delete) and simulator flows with fixture images. **Not verified**: the real dual camera and its timing, hold-to-talk in the UI, label quality on real photos |
| Settings panel | Built: gear in the Void tab. Persistence and back-only verified in the simulator, re-tag (never touches user edits) in the Mac harness |

## Open questions

Each one has a working default in [DECISIONS.md](DECISIONS.md) so building isn't blocked: default visibility (D5), lock gesture (D6), friend cap (D7) and old-message import (D8). Override any of them there.

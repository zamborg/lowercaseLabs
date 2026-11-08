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

- **Kind**: what produced it. v1 has one kind, `voice`.
- **Payload**: the raw content. For `voice`: the audio file and its transcript. The payload is the user's and is never shared.
- **Enrichment**: derived metadata, currently a **title** and **tags**. Produced by a model after the entry is saved. Every enriched field remembers whether the model or the user set it. A user edit always wins and is never overwritten by a later model run.
- **Visibility**: `private` or `shared`. Chosen at capture, changeable any time after.
- **Created at**: when it was captured.

Saving never waits on enrichment. An entry exists the instant capture ends, even if the model is slow or fails.

### Module

A way to make entries. A module provides exactly three things:

1. A **capture surface**: a page you can reach from the home screen.
2. A **payload** shape: what gets stored.
3. **Text for enrichment**: how its payload becomes text the models can read. Voice provides its transcript.

Enrichment, the Void list, visibility and sharing don't know or care which module made an entry. That is what makes the next module cheap. A camera module (a BeReal-style "swipe to the camera, snap, post") would be a new capture surface, a photo payload and a caption or vision-derived text, and nothing else would change. We are not building it. The primitives must just not make it hard.

### Tags

Exactly the four How We Feel quadrants: high-energy pleasant (yellow), low-energy pleasant (green), high-energy unpleasant (red) and low-energy unpleasant (blue). Each entry gets one dominant quadrant and optionally a second. Tags are a fixed enum ([taxonomy.json](taxonomy.json)), never free text, so they stay comparable across people and time.

### Dot

The only thing that leaves your device for friends: who, when, and which tags, for one shared entry. A dot exists exactly while its entry is shared. If the entry is made private or deleted, the dot is withdrawn.

### Friend

A mutual connection made by invitation, built on iCloud sharing. The friend list is small by design (cap around 30, unconfirmed). There are no public profiles, followers or feeds.

### Identity

Your iCloud account. There is no sign-up and no sign-in screen. First launch asks for microphone and speech permission, and that's it.

## V1: what you see

A tab bar with three tabs.

### 1. Capture (home, the app opens here)

- A black screen with one circle. The circle reacts to your voice level while it listens.
- **Press and hold** to record. Recording starts on touch, with no perceptible delay.
- **Slide off** the circle while holding to **lock** the recording, so you can let go and walk around with the phone out. Tap to stop.
- The transcript streams in live while you talk.
- On release, the message is saved immediately. The title and tag dots appear a moment later.
- You choose **private or shared** for this message. The control is right there and remembers your last choice.

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

## Budgets

These are product requirements, not aspirations. Each one is measured by a flow in [AGENTS.md](AGENTS.md).

| What | Budget | Measured on |
|---|---|---|
| Touch to recording | Within 1 frame (16 ms) | iPhone 14 Pro |
| Release to message saved | Under 300 ms | iPhone 14 Pro |
| Release to title and tags shown | Under 2 s | iPhone 14 Pro |
| App launch to ready to record | Under 1 s | iPhone 14 Pro |
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

- Photos, camera or any second module (designed for, not built).
- A "my dots" tab, reactions, comments, push notifications for dots.
- Android, iPad, Mac apps; web; public feeds; cloud LLMs.
- Accounts beyond iCloud.

## Where the code is versus this doc (2026-11-08)

| Area | State |
|---|---|
| Capture: hold, live transcript, save | Working on iPhone 14 Pro: touch to recording ~5 ms, release to final transcript ~60 ms |
| Enrichment: title and tags | Working on iPhone 14 Pro, Release build: ~1.45 s. Title quality is weak, tag accuracy unproven on real messages |
| Data model | `Memo`, with no `kind`, no visibility and no edit provenance. **Rename to `Entry` and add these before v1 ships**, because changing a CloudKit schema after release is painful |
| Visibility | Missing. The current code publishes a dot for every message, which contradicts this doc |
| Lock gesture, voice-reactive circle | Missing |
| Tab bar, Void card, Social grid | Missing. Today there are sheets for a plain list and a friends list |
| Editing transcript, title, tags | Missing |
| Dots and friends | Code exists, never run against real iCloud accounts. Friends show as opaque ids, and invites are one-way |
| Audio sync across your devices | Missing. Only text syncs. Not needed for single-device v1 |

## Open questions

- **Default visibility for a new message**: private or shared? I'd default to private on first use, then remember the last choice.
- **Lock gesture direction**: slide up (like iMessage voice notes) or any direction off the circle?
- **The friend cap**: confirm the limit CloudKit sharing actually imposes.
- **Old messages from the previous app**: import them or start fresh?

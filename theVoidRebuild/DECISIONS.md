# Decisions

Each one records what was decided, why, and who decided. Anything marked **(default)** was decided by Claude so execution isn't blocked. Zubin can override any of them, and the override goes in this file.

| # | Date | Decision | Why | By |
|---|---|---|---|---|
| D1 | 2026-11-07 | Tags are the four How We Feel quadrants only. 1 dominant tag, plus an optional second | Comparable across people and time. With four values, a third tag carries no information | Zubin (quadrants), Claude (count) |
| D2 | 2026-11-07 | Tagging model: LFM2-1.2B 4-bit, via MLX Swift, schema-constrained. Weights bundled in the app | Mac eval (`eval/tagging/report.md`). Bundling because the first-run download timed out on the phone | Claude |
| D3 | 2026-11-08 | Agents verify by user flows on local rigs, and Zubin is out of the build-and-test loop until TestFlight | Zubin's direction. See [AGENTS.md](AGENTS.md) | Zubin |
| D4 | 2026-11-08 | Rename `Memo` to `Entry` and add `kind`, `visibility`, and model-vs-user provenance for title and tags, **before** any CloudKit schema is deployed to production | CloudKit production schemas can only be extended, never changed. [VISION.md](VISION.md) primitives | Claude |
| D5 | 2026-11-08 | **(default)** A new message is **private** on first use. After that, the capture screen remembers the last choice | Sharing is the riskier mistake. One tap to change it | Claude |
| D6 | 2026-11-08 | **(default)** Lock a recording by **sliding up** past a threshold while holding, like iMessage voice notes. Tap the circle to stop | Familiar gesture. Up is clear of the tab bar | Claude |
| D7 | 2026-11-08 | **(default)** Design for a friend cap of 30. Rig C measures the real limit | Unconfirmed CloudKit sharing limit | Claude |
| D8 | 2026-11-08 | **(default)** No import of old-app messages in v1. Revisit after v1 | It isn't on the critical path. Old data stays in the old app | Claude |
| D9 | 2026-11-08 | Development builds keep bundle id `com.lowercaselabs.thevoid`, so the rebuild installs **alongside** the old app | Zubin keeps the old app and its data while v1 is unfinished | Claude |
| D10 | 2026-11-08 | **(default)** TestFlight ships under the **old app's** App Store Connect record, `com.lowercaseLabs.theVoid` (team `9G57J2JMF7`, store profile already exists). **The first TestFlight install replaces the old app on Zubin's phone, so Zubin approves that switch at the time** | Reuses the existing record, and avoids a new App Store name and review setup | Claude, pending Zubin at step 7 |
| D11 | 2026-11-08 | **(default)** New iCloud container, `iCloud.com.lowercaseLabs.void`, not the old app's `iCloud.com.lowercaseLabs.theVoid` | The old container holds the old app's schema and data. A clean schema avoids collisions | Claude |
| D12 | 2026-11-08 | Test seams are launch arguments at the app's edges only (`-fixtureAudio`, `-seed`, `-autotest`). No protocols with one conformer | Vision principle: one implementation per layer | Zubin (principle), Claude (mechanism) |
| D13 | 2026-11-08 | Full v1 (journal **and** Social) is built and running on Zubin's phone before any TestFlight. TestFlight is for distributing to friends, after v1 is tight. *(Supersedes Claude's earlier default of shipping the journal first.)* | Zubin's direction: tighten the whole feature set first | Zubin |

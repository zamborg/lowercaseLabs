# Rebuild plan (historical)

Superseded 2026-11-08 by [VISION.md](VISION.md) (product) and [AGENTS.md](AGENTS.md) (workflow and roadmap). Kept for the record of how the first build was done.

Legend: C = Claude (design, judgment, review), X = Codex (deterministic execution from a brief in briefs/).

| # | Step | Owner | Done when |
|---|------|-------|-----------|
| 0 | Synthetic labeled memos, two-model generate/judge (brief 00) | X | ~300 memos, clean/hard splits |
| 1a | Real holdout: hand-label ~50 real memos; record ~20 clips + corrected transcripts for WER | Zubin (C builds a tiny labeling CLI) | holdout.jsonl + reference.jsonl |
| 1b | Tagging eval: Python + mlx-lm, schema-constrained, N candidates | X (brief 01) | report.md with accuracy per quadrant, tok/s |
| 1c | Transcription eval: macOS Swift CLI on SpeechAnalyzer, WER | X (brief 02) | wer.md |
| 1d | Pick model, record decision | C | DECISIONS.md |
| 2 | Capture: hold button, AVAudioEngine single tap, live transcript | C (latency critical) | recording starts <= 1 frame, tested on device |
| 3 | Tagging on device via MLX Swift + constrained decoding | C, X for boilerplate | title+tags <= ~2s after release |
| 4 | SwiftData persistence + private CloudKit sync | X (brief 04) after C defines models | memo round trip across 2 devices |
| 5 | CKShare prototype w/ test accounts, confirm cap | Zubin + C | flow + cap documented |
| 6 | Dots: publish, invite, accept, view | C | friends see tags only |
| 7 | iPhone 14 Pro latency check, tune | Zubin + C | budgets set and met |

## Reuse from old app (theVoidLocal), only if proven
Old app is 12k lines of Swift. Mine it, do not port it: AppleSpeechTranscription.swift (SpeechAnalyzer notes), ICloudSyncService.swift (CloudKit gotchas), CloudKit share code in SocialFeatureClient.swift. Everything else (HealthKit, 40+ tag taxonomy, engine coordinators) is deleted by design.

## Open questions (carried)
- Tags per memo: recommend exactly 1 dominant quadrant + optional 2nd. With only 4 values, 3 tags is meaningless.
- Old memo migration: depends on step 0 export.

## Status (Nov 7, 2026 evening)
- Done + verified on iPhone 14 Pro: step 2 capture (press->capturing 5-12 ms, release->final 50-160 ms).
- Done, compiles, NOT yet run end to end: step 3 tagging (LFM2-1.2B, MLX Swift, schema-constrained), step 4 persistence (SwiftData, CloudKit private DB config), step 6 dots code (DotStore, zone-wide CKShare, friends view, share acceptance).
- Blocked: device builds with iCloud need an Apple ID signed in under Xcode > Settings > Accounts (or the iCloud container created at developer.apple.com). Step 5 needs two iCloud accounts to confirm the friend flow and the 30 cap.
- Known gaps: audio files are not synced (SwiftData syncs text only); friends are labeled by opaque id, not name; invite is one-way (accepting does not auto-invite back); tag accuracy unproven on real memos.

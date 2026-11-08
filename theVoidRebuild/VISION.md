# The Void: Rebuild Vision (Nov 7, 2026)

Source of truth is the vision doc pasted by Zubin on 2026-11-07. Summary:

- iPhone-only (iOS 26, Swift, SwiftUI), min device iPhone 14 Pro (A16). Everything on device.
- Core loop: hold -> record (AVAudioEngine, one tap -> file writer + SpeechAnalyzer) -> live transcript -> release -> local LLM (MLX Swift, JSON-schema constrained) makes title + tags -> save -> publish dot.
- Tags: fixed 4-quadrant enum (see taxonomy.json). Never free text.
- Storage: SwiftData + audio files. Sync/social: CloudKit private DB (memos) + shared zone (dots). Auth: iCloud only.
- Dots: friends-only, mutual CKShare, cap 30 (unconfirmed), pull on open.
- Principles: capture speed beats everything; one implementation per layer; no engine protocols; small codebase.
- Targets: recording starts within 1 frame; title+tags within ~2s of release on iPhone 14 Pro.
- Non-goals: Android/iPad/Mac, pluggable engines, cloud LLMs, public feeds, dot push.

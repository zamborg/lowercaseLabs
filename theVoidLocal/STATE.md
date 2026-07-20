# theVoidLocal State

Last updated: 2026-05-28 17:00 PDT

## Overall Snapshot

- iOS app is local-first and no longer depends on a Fly.io backend.
- Sign in with Apple establishes local identity only; no backend JWT/session is created.
- Private sync uses CloudKit private database records.
- Social dots use CloudKit Sharing and shared database zones.
- Backend, Docker Compose, and Fly deployment scaffolding have been removed.

## iOS App Behavior

- Recording is local.
- Transcription is on device.
- Insight extraction uses the local Liquid pipeline when prepared.
- Journal entries, transcripts, insights, audio, drafts, and health snapshots are stored locally and sync privately through CloudKit.
- Health data is not shared socially.

## Social Sharing

- `CloudKitSocialFeatureClient` owns social sharing.
- A social share root record lives in `VoidSocialZone`.
- Shared records contain only profile label fields and dot payloads.
- Friends exchange iCloud share links. Accepting a link adds the owner's shared zone to the recipient's shared CloudKit database.
- The Social tab reads accepted shared zones and displays newest friend dots first.

## Model Delivery

- Preferred packaging is local app resources under `theVoid/theVoid/ModelAssets/`.
- Developer overrides still support `thevoid.liquid.localBundlePath`, `thevoid.liquid.modelFileURL`, and `thevoid.liquid.modelManifestURL`.
- There is no default Fly-hosted model URL.

## Validation

- iOS simulator build passed on May 28, 2026.
- iOS simulator tests passed on May 28, 2026.

## Remaining Risks

- CloudKit sharing should be tested with two real iCloud accounts before release.
- Existing Swift 6 readiness warnings remain in audio, CloudKit operation callbacks, and model download code.
- `API*` model names are now legacy local DTO names; they are not backend-bound.

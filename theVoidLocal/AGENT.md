# theVoid Agent Playbook

## Product Shape

- `theVoid/` is the iOS app.
- The app is local-first: recording, transcription, insights, journal storage, and health scoring run on device.
- Private multi-device sync uses CloudKit private database records.
- Trusted social sharing uses CloudKit Sharing and accepted shared database zones.
- There is no Fly.io backend, server auth session, admin dashboard, Docker stack, or backend deployment path in this repo.

## Working Loop

Use Xcode/XcodeBuildMCP for normal validation.

```bash
make ios-build
make ios-test
```

For CloudKit sharing behavior, simulator builds are useful for compilation, but end-to-end acceptance should be tested with real iCloud accounts on two devices or simulator/account combinations that can accept CloudKit shares.

## Auth And Identity

- Sign in with Apple is used only to establish a local app identity.
- iCloud account availability is required for sync and social sharing.
- The app derives a stable local user id and anonymous handle from the Apple user identifier.
- No Apple identity token is sent to a backend.

## Social Model

- A user creates an iCloud share link for their social circle.
- Friends accept that share link through CloudKit.
- The Social tab reads accepted shared CloudKit zones and displays dot records from trusted friends.
- Journal transcripts, audio, and health data are not placed in social share records.

## Model Delivery

- Prefer bundled model artifacts in `theVoid/theVoid/ModelAssets/`.
- Custom local or remote model URLs remain developer overrides.
- There is no default hosted model endpoint.

## Definition Of Done

- Code change is applied.
- iOS build and tests pass, or any remaining failure is explicitly called out.
- Backend/Fly assumptions are not reintroduced.

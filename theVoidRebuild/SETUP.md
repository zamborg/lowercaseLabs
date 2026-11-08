# One-time setup only Zubin can do

Agents can't do these, because each one involves signing in, creating accounts, or Apple account settings. Each is done once. Agents check for them and stop with a clear message if one is missing.

What already exists (checked 2026-11-08): an Apple Development certificate for zubinaysola@gmail.com, team `9G57J2JMF7` ("Zubin Aysola"), development profiles, and an **App Store distribution profile for the old app** (`com.lowercaseLabs.theVoid`). So the paid developer account and an App Store Connect record are already there.

## 1. Sign Xcode into your Apple ID (about 2 minutes). Unblocks briefs 12, 14, 16

Xcode → Settings → Accounts → **+** → Apple ID → zubinaysola@gmail.com. Make sure team "Zubin Aysola" appears.

Today Xcode has **no account** signed in. Device builds only work through cached profiles. This step lets agents add the iCloud capability, register this Mac to run the app (rig B), and archive for TestFlight.

## 2. Two test iCloud accounts for sharing (about 20 minutes). Unblocks brief 14

1. Create **two** new Apple IDs for testing, for example `thevoid.test.a@…` and `thevoid.test.b@…`. They can be free accounts.
2. Boot two simulators (an agent can do this for you on request) and sign each into one test account: Settings app → Sign in → iCloud on.
3. Optional: add both to [App Store Connect → Users and Access → Sandbox] only if we ever need purchase testing. We don't for v1.

Agents then run sharing flows between these two accounts against the CloudKit **development** environment only.

## 3. App Store Connect API key (about 5 minutes). Unblocks brief 16

1. App Store Connect → Users and Access → Integrations → App Store Connect API → generate a key with the **App Manager** role.
2. Download the `.p8` file to `~/.appstoreconnect/private_keys/AuthKey_<KEYID>.p8`.
3. Create `~/.appstoreconnect/config` with two lines:

```
KEY_ID=<key id>
ISSUER_ID=<issuer id>
```

Don't paste the key into chat. Agents read the file and never print or commit it.

## 4. Decisions to confirm when convenient

[DECISIONS.md](DECISIONS.md) entries marked **(default)** were decided by Claude so work isn't blocked: private by default (D5), slide up to lock (D6), friend cap of 30 (D7), no old-message import in v1 (D8), TestFlight replacing the old app (D10), and a new iCloud container (D11). Override any of them by just saying so.

## 5. Real data for the ML loop, whenever you like

- About 50 of your real messages, hand-labeled with a quadrant (an agent will build a small labeling tool when you're ready).
- About 20 voice recordings with corrected transcripts, for transcription accuracy.

Nothing blocks on this, but synthetic data overstates quality until it exists.

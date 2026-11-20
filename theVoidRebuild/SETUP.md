# One-time setup only Zubin can do

Agents can't do these, because each one involves signing in, creating accounts, or Apple account settings. Each is done once. Agents check for them and stop with a clear message if one is missing.

What already exists (checked 2026-11-08): an Apple Development certificate for zubinaysola@gmail.com, team `9G57J2JMF7` ("Zubin Aysola"), development profiles, and an **App Store distribution profile for the old app** (`com.lowercaseLabs.theVoid`). So the paid developer account and an App Store Connect record are already there.

## 1. Sign Xcode into your Apple ID. DONE 2026-11-08

Note: Xcode's command-line build can't see the account on this Mac, so a **new** capability or bundle id needs one ⌘R in the Xcode window to create its provisioning profile. After that, `app/build-phone.sh --icloud` works from the command line.

(Original note) Unblocks running the app on the Mac, iCloud, TestFlight

Xcode → Settings → Accounts → **+** → Apple ID → zubinaysola@gmail.com. Make sure team "Zubin Aysola" appears.

Today Xcode has **no account** signed in. Device builds only work through cached profiles. This step lets agents add the iCloud capability, register this Mac to run the app (rig B), and archive for TestFlight.

## 2. Two test iCloud accounts for sharing (about 20 minutes). Unblocks milestone 3 (friends)

1. Create **two** new Apple IDs for testing, for example `thevoid.test.a@…` and `thevoid.test.b@…`. They can be free accounts.
2. Boot two simulators (an agent can do this for you on request) and sign each into one test account: Settings app → Sign in → iCloud on.
3. Optional: add both to [App Store Connect → Users and Access → Sandbox] only if we ever need purchase testing. We don't for v1.

Agents then run sharing flows between these two accounts against the CloudKit **development** environment only.

## 3. TestFlight for yourself: app record and API key (about 15 minutes). Unblocks milestone 4

1. **Create the app record** (only possible in the App Store Connect web UI): My Apps → **+** → New App. Platform iOS, bundle id `com.lowercaselabs.void` (appears once Xcode registered it), any SKU (for example `thevoid-rebuild`), and a name that isn't "The Void" (taken by the old app), for example "The Void Rebuild". The name only matters in the store; the home-screen name stays "The Void".
2. **Add yourself as an internal tester**: the new app → TestFlight → Internal Testing → **+** group → add zubinaysola@gmail.com. Internal builds need no Apple review.
3. **Create the API key**: Users and Access → Integrations → App Store Connect API → generate a key with the **App Manager** role. The `.p8` can only be downloaded once.
4. **Get the key to the Mac.** If you're away from it, pasting the key text into the chat is the only way, and it lives in this conversation's history. So: use a key made just for this, and **revoke it** (same page, Revoke) once TestFlight works or any time you like. A revoked key is dead. An agent then saves it as `~/.appstoreconnect/private_keys/AuthKey_<KEYID>.p8` (mode 600) and writes `~/.appstoreconnect/config`:

```
KEY_ID=<key id>
ISSUER_ID=<issuer id>
```

Agents never print or commit the key. Uploads go through `app/ship.sh --yes`, which refuses to run without that flag.

TestFlight builds use the **production** CloudKit environment. Dots and iCloud sync do nothing there until the schema is deployed to production from the CloudKit Console, which is permanent. Until you approve that, a TestFlight build is a fully working local journal with no dots.

## 4. Decisions to confirm when convenient

[DECISIONS.md](DECISIONS.md) entries marked **(default)** were decided by Claude so work isn't blocked: private by default (D5), slide up to lock (D6), friend cap of 30 (D7), no old-message import in v1 (D8), TestFlight replacing the old app (D10), and a new iCloud container (D11). Override any of them by just saying so.

## 5. Real data for the ML loop, whenever you like

- About 50 of your real messages, hand-labeled with a quadrant (an agent will build a small labeling tool when you're ready).
- About 20 voice recordings with corrected transcripts, for transcription accuracy.

Nothing blocks on this, but synthetic data overstates quality until it exists.

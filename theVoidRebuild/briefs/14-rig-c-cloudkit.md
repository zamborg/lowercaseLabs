# Brief 14: dots end to end on CloudKit (rig C)

**Executor:** Claude (Sonnet). **Depends on:** 10, 11, and SETUP steps 1 and 2. **Stop and report if SETUP isn't done**; don't work around it.

## Goal

Prove sharing works between two real iCloud accounts, and measure it: F6 (share), F7 (friend sees it), F9 (invite).

## Setup, done once by the agent after Zubin's SETUP steps

- Switch to the new container `iCloud.com.lowercaseLabs.void` (D11): entitlements and `DotStore`. Set `VoidICloud: true` and `CODE_SIGN_ENTITLEMENTS` in `project.yml`.
- Use the CloudKit **development** environment only. **Never deploy the schema to production.** That's a Zubin decision at TestFlight time.
- **Account A**: wherever rig B runs (the Mac app), or simulator 1. **Account B**: simulator 2. Zubin signs these in (SETUP step 2). Agents never type Apple ID passwords.

## Flows

| Flow | Steps | Must be true | Measures |
|---|---|---|---|
| F9 invite | A creates the share. The share URL is accepted on B (`xcrun simctl openurl`, or a test-only `-acceptShare <url>` launch argument) | B lists A as a friend. **Note:** today the invite is one-way. If mutual is needed, report it rather than redesigning silently | — |
| F6 share | A records a shared message | Dot record exists in A's shared zone with tags and time only. Assert the record has no title or transcript fields | release → dot saved on server |
| F7 friend sees | B refreshes Social | B sees A's dot | dot saved → visible on B (poll every 500 ms, max 60 s) |
| F6b private | A flips the message to private | Dot record deleted. B no longer sees it after refresh | — |

Also measure the **friend cap** (D7) with a scripted loop if it's cheap (participants on one share). If it needs more real accounts, write down what you found and stop.

## Fix while here

Friends currently show as opaque ids. Use the share participant's name components where CloudKit provides them.

## Acceptance

- `./verify --full` runs F6, F7, F6b and F9 and reports timings.
- [DECISIONS.md](../DECISIONS.md) gets the measured cap, or the reason it couldn't be measured.

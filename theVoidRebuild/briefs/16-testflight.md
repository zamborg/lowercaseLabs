# Brief 16: TestFlight and `./ship`

**Executor:** Claude (Sonnet). **Depends on:** 13, 14, and SETUP step 3. Needs Zubin's go-ahead **at the time** for the bundle id switch (D10).

## Goal

A green `main` reaches Zubin's phone through TestFlight with nobody touching a cable.

## Steps

1. **Confirm with Zubin**, in chat: the first TestFlight build ships as `com.lowercaseLabs.theVoid` and **replaces the old app on his phone** (D10). If he says no, register a new bundle id and app record instead (he creates the record; agents don't create accounts or app records).
2. Add a Release configuration that uses the store bundle id. Development builds keep `com.lowercaselabs.thevoid` (D9).
3. **CloudKit production schema.** Before the first TestFlight build with sharing, deploy the development schema to production. **Zubin approves this explicitly**, because production schemas can't be changed afterward, only extended (D4).
4. Write `./ship`:
   - Refuses unless `./verify --full` passed on this exact commit (read the latest report's sha).
   - Bumps the build number.
   - `xcodebuild archive`, then `-exportArchive` with an `ExportOptions.plist` for App Store Connect.
   - Uploads with the App Store Connect API key from SETUP step 3: key id, issuer id, and the `.p8` path read from `~/.appstoreconnect/` and **never printed or committed**.
   - Prints the build number and the TestFlight processing status.
5. The bundled model weights (~650 MB) make a large build. Check it's under App Store limits and that on-demand resources aren't needed. Note the size in the report.

## Acceptance

- One `./ship` produces a build that appears in TestFlight for Zubin.
- After this, the agent loop in [AGENTS.md](../AGENTS.md) ends with `./ship` instead of stopping at the report.

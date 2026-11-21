# Controllr status

## Device build

The signed iOS build is installed on the paired iPhone 14 Pro. **Controllr Voice** was imported on the Mac and appeared in iPhone Spotlight. The app's Threads screen opened on the phone and the first shortcut launch reached the Speech Recognition prompt.

The user confirmed the physical microphone test. Controllr Voice captured “Text Sam” and the spoken follow-up “Running ten minutes late” in one thread on the iPhone.

## Included in this build

- Controllr owns the append-only turns within a thread. The shortcut starts a thread or passes its ID back for clarification. Threads remain available in the app and can resume by voice.
- **Saved** provides editable memories, skills, and notes. The agent can search them and use the tool result in the same turn. Explicit save requests can create notes, update a memory, or save a named skill. Explicit Bear requests still hand off to Bear.
- Each action manifest provides the tool schema, required fields, confirmation rule, execution destination, and eval cases. The voice shortcut confirms text messages, calls, and ride handoffs, and records whether the step completed or was cancelled. A completion receipt describes the shortcut step.
- Phone actions use the generated Controllr Voice shortcut. The previous probe, test, and alternate router XML sources are in Git history. See [the current router spec](shortcuts/specs/router.md) for the build and action workflow.

## Verification

- All 26 core tests passed. They cover ordered conversation turns, confirmation and validation, and both providers' tool-result format.
- All three iOS simulator tests passed. They cover reopening a stored conversation, cancellation receipts, and querying memory, skills, and notes after reopening the database.
- Live OpenAI routing passed 60 of 61 cases, including the three multi-turn routing cases. One alarm request prompted for clarification. Separate live-model smoke checks used a saved memory and applied a saved skill. See [evals/RESULTS.md](evals/RESULTS.md).
- Shortcuts signing and plist lint passed. Playground's static check flags the Send Message body binding because it expects a named list variable. The generator uses the working text-token wiring from the repository's earlier device checks. The device test exercised the two-turn text flow. Live message delivery and call placement are unverified in this build.

Calendar and Lyft are follow-on actions. The old shortcuts and stray Clock timers on the devices and API-key rotation are still listed for cleanup.

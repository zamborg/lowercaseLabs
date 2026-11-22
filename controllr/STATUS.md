# Controllr status

## Agent setup and maintenance

- **Setup → System Prompt** shows the composed prompt with runtime context. **Edit Instructions** saves a shared override for all providers. To restore the bundled instructions, choose **Use Default Instructions** and save.
- **History → System Prompt for This Run** shows the prompt recorded for a new model request. The regular History export keeps its existing fields.
- The project-local [controllr-shortcuts skill](.agents/skills/controllr-shortcuts/SKILL.md) documents how to maintain the voice loop and add actions.

## Controllr Voice audit

**Controllr Voice 1.6** brings the router from 486 actions to 433 by sharing completion across handlers. The iPhone checks covered clipboard and group-chat reads, cancelling a text proposal, and resuming its thread through two consecutive follow-ups. The signed 1.6 workflow matches the audit workflow action for action and passed a spoken-answer smoke test. Controllr opens 1.6 for new and resumed voice sessions.

The 174 steps for literal timer durations remain. The iPhone accepted a literal 83-second control. Three runtime duration bindings failed to start that countdown without intervention. [Audit and device results](shortcuts/specs/audit.md)

## V1.5 on iPhone

The paired iPhone has the V1.5 Controllr build. The **Modules** tab now has an on/off switch per action and a confirmation setting for writes and handoffs. In **Setup**, Calendar, Contacts, and Reminders show their current access. Shortcuts manages access for its own actions.

- Calendar supports agenda, availability, event creation, and editing. Reminders supports listing, adding, completing, and rescheduling. Set the default calendar and reminder list under the corresponding Add modules.
- **Controllr Voice 1.5** passed the physical clipboard and Messages tests. It read the copied test phrase and retrieved a text by a distinctive phrase.
- **Controllr Voice 1.5.1** passed both sender and named group-chat lookup on the iPhone. It checks the 120 newest messages on the phone and forwards the latest matching body. Shortcut lookup results continue the same Controllr thread and can lead to another spoken question.
- Brightness and media volume take percentages. **Request Lyft** accepts the name of a user-owned Shortcut and hands off the destination as text.

The [capability map and extension audit](V1.5.md) track the remaining phone surfaces. The [router spec](shortcuts/specs/router.md) covers the protocol, signing, and the steps for adding an actuator.

V1.5 passed 37 core tests and seven simulator tests. A separate simulator smoke test created and edited a Calendar event and created, rescheduled, and completed a Reminder, using temporary fixtures. The live OpenAI eval matched 74 of 75 routes, including sender and conversation lookup. One note-creation prompt asked an unnecessary follow-up. [Routing results](evals/RESULTS.md)

## V1 device verification

The signed iOS build is installed on the paired iPhone 14 Pro. **Controllr Voice** was imported on the Mac and appeared in iPhone Spotlight. The app's Threads screen opened on the phone and the first shortcut launch reached the Speech Recognition prompt.

The user confirmed the physical microphone test. Controllr Voice captured “Text Sam” and the spoken follow-up “Running ten minutes late” in one thread on the iPhone.

In the first two-follow-up microphone test, the second question went out as a terminal answer. The revised app accepts native assistant text and optional tool calls. A `clarify` tool call keeps the session open, and Controllr returns the next dictation as its tool result. A direct question in plain text also keeps listening. The shortcut continues to use the same protocol. macOS imported the shortcut with the explicit speech wait as **Controllr Voice 1**. The user confirmed the new build on iPhone: Controllr Voice 1 asked and listened twice, then delivered the final recommendation.

## V1 provider setup

The signed three-provider build is installed. In **Setup**, select OpenAI, Anthropic, or OpenRouter, save that provider's API key, and choose from the search-enabled model picker. Keys and selected models are stored separately. OpenRouter shows text models whose catalog entries support tools and requests routing through compatible providers.

The OpenAI catalog and all 61 live routing cases passed. Anthropic and OpenRouter passed tests for discovery, assistant text, and matched follow-up tool replies using recorded response shapes. The three-provider build is the V1 baseline. Authenticated Claude and OpenRouter voice checks are deferred until keys are available.

## Included in this build

- Controllr owns the append-only turns within a thread. The shortcut starts a thread or passes its ID back for clarification. Threads remain available in the app and can resume by voice.
- **Saved** provides editable memories, skills, and notes. The agent can search them and use the tool result in the same turn. Explicit save requests can create notes, update a memory, or save a named skill. Explicit Bear requests still hand off to Bear.
- Each action manifest provides the tool schema, required fields, confirmation rule, execution destination, and eval cases. The voice shortcut confirms text messages, calls, and ride handoffs, and records whether the step completed or was cancelled. A completion receipt describes the shortcut step.
- Phone actions use the generated Controllr Voice shortcut. The previous probe, test, and alternate router XML sources are in Git history. See [the current router spec](shortcuts/specs/router.md) for the build and action workflow.

## Verification

- All 35 core tests passed, including Claude catalog pagination, OpenRouter filtering and errors, and native and Chat Completions tool replies.
- All four iOS simulator tests passed. They cover independent provider settings, persisting successive follow-up calls, cancellation receipts, and saved-item lookup after reopening the database. The three provider setup screens were also checked in the simulator.
- A live-model smoke test returned an ordinary final answer, used two successive `clarify` calls, and read a seeded memory and a seeded note through the lookup tool on separate turns.
- The latest live OpenAI routing run passed 61 of 61 cases. See [evals/RESULTS.md](evals/RESULTS.md).
- Shortcuts signing and plist lint passed. Playground's static check flags the Send Message body binding because it expects a named list variable. The generator uses the working text-token wiring from the repository's earlier device checks. The device test exercised the two-turn text flow. Live message delivery and call placement are unverified in this build.

The [V1.5 capability map](V1.5.md) covers first-party phone actions, per-module enablement and confirmation, and the actuator extensibility audit. The old shortcuts and stray Clock timers on the devices and API-key rotation are still listed for cleanup.

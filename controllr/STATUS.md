# Controllr status

## Device build

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

After V1, add phone actuators such as Lyft and Calendar and refine the agent loop and service. Additional completion stages can use the same provider-neutral message history and turn protocol. The old shortcuts and stray Clock timers on the devices and API-key rotation are still listed for cleanup.

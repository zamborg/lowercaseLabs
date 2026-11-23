# Controllr Voice audit

Keep one installed voice entry point. In the generator, give the voice loop and action dispatch separate responsibilities. Controllr should continue to own the thread, model decisions, action policy, and receipts. The shortcut needs the active thread ID, the latest decision, and the current action result.

## Where the steps go

| Controllr Voice 1.5.1 | Generated workflow steps |
| --- | ---: |
| Entire shortcut | 486 |
| Timer handler | 174 |
| Completion and bookkeeping inside each of 14 dispatch branches | 84 |
| Everything else | 228 |

The first iPhone timer experiment opened a duration picker when it tried to supply a variable. The generator uses 43 literal Start Timer actions and checks the normalized duration against each one. Even 90 seconds is absent from the list. The dry-run eval accepts it as a timer request, but the app asks for another duration before dispatch. The 174 steps include duration lookup, coercion to text, and the conditions and comments around each literal. [Original timer result](../../DECISIONS.md), [Timer handler](../gen.py)

The 1.5.1 dispatcher emitted six identical bookkeeping steps in each branch. Three handlers read data and eleven paths complete without returning text. The read handlers need to return their actual result to Controllr. For example, the phrase lookup needs only a phrase, Find Message, and the returned body. Controllr verifies the phrase against that body and resumes the model call in the same thread. [Shortcut generator](../gen.py), [Action completion](../../App/ConversationStore.swift)

## Voice flow and action flow

1. **Voice.** Dictate and send the transcript and optional thread ID to Decide. Save the returned ID. Speak a follow-up before listening again. A final reply speaks and exits. A silent decision exits. An action decision hands off to dispatch. The 32-iteration limit bounds this outer loop. Resuming from the app starts a new shortcut run with the saved thread ID.
2. **Dispatch.** Show a confirmation when requested. Record cancellation and exit, or authorize the action once. Pick the matching handler using its manifest ID. Keep a common completion path for reads and a common path for writes and handoffs. Return their decision to the voice loop. Clearing the action's output before the next dispatch prevents a missing read from picking up data from a previous one.
3. **Controllr.** Keep authorization, policy checks, result validation, and provider history in the app. A read result rejoins the original model turn with its tool call ID. It can produce a final answer, a follow-up, or another action. A write receipt means the Shortcut reached its callback. An interrupted system action can still show Started.

The repository previously tried a set of installed sub-shortcuts. Its device test recorded repeated Run Shortcut permission prompts. Keep the handler boundary in the generator and ship one shortcut while that constraint holds. Preserve the installed contract during the first pass so OpenAI, Anthropic, and OpenRouter keep the same turn history and tool results. [Earlier dispatch experiment](../../DECISIONS.md), [Router protocol](router.md)

## Refactor and verification

1. **Centralize completion.** The signed Router Audit prototype has 433 steps and two completion sites, down from 486 steps and 14 completion sites. It keeps the current timer handler and uses the same version 3 App Intents. Verify empty and successive reads, rejection and cancellation, and that an unknown handler exits without a completion receipt. Confirm a harmless write and check for one receipt and one effect.
2. **Prove runtime duration in Clock.** The signed Timer Audit probe has literal 2-minute and 83-second controls and two variable bindings. Run each variable binding with 37 and 83 seconds. Check Clock's actual countdown and whether the Start Timer action opens a second prompt. The first number prompt in the probe supplies the runtime input intentionally. A working binding would allow a numeric duration to replace the 43 cases as a separate change. Keep the current wire value supported for previously installed shortcuts. Apple documents parameter variables in general. Clock duration still needs this device test. [Apple Shortcuts variables](https://support.apple.com/en-am/guide/shortcuts/apdd02c2780c/ios)
3. **Verify the user-facing workflow.** Exercise two consecutive voice follow-ups, resume an existing thread from the app, cancel a text proposal, and ask for a recent text from a named group. Verify the group result goes into the same thread and that another dictation waits for speech to finish. Compare the real device workflow before changing the normal launch shortcut.

## Probe files

Run `python3 shortcuts/probes.py timer --team-id YOUR_TEAM_ID` and `python3 shortcuts/probes.py router --team-id YOUR_TEAM_ID`. Import the signed probes under their own names. The generated XML and signed files are in the ignored `build/shortcut-audit` directory. Use the signing command from the router spec with each probe's filename. [Probe generator](../probes.py), [Signing instructions](router.md#build-and-install)

On the paired iPhone 14 Pro, the literal 83-second control started the expected countdown. Passing the number directly opened Apple's duration picker. Neither a variable magnitude in a fixed-seconds quantity nor a `Measurement<UnitDuration>` returned from a temporary Controllr App Intent started the requested countdown without intervention. Keep the literal timer handler for the dispatcher refactor. The normal Controllr build was reinstalled after the measurement test. [App Intent result types](https://developer.apple.com/documentation/appintents/adding-parameters-to-an-app-intent)

The router prototype read a known clipboard value and answered from it. The first group lookup returned the same no-match response through the installed workflow and the prototype. A second group lookup used an identical fixed transcript on both paths. Both read the expected message. Cancelling a proposed text through the prototype showed **Cancelled** in the iPhone thread. **Continue by voice** resumed that same thread, listened after both follow-up questions, and appended the final answer. The generated **Controllr Voice 1.6** release uses exactly the tested 433 actions with its new name.

The 37 core and seven app simulator tests passed. The timer probe passed static validation. The router prototype has the same Send Message token warning as the installed shortcut. Both probes passed plist lint and were signed.

## Desktop audit for a modular router

The generator reproduced the checked-in Voice 1.6 plist exactly. The counts below include comments and control-flow boundaries, which appear as rows in the Shortcut editor. The workflow contains 84 Comment actions and zero Wait actions. Speech uses Speak Text's wait-until-finished setting. [Generator](../gen.py)

| Voice 1.6 section | Generated steps |
| --- | ---: |
| Voice loop, including setup and exit | 45 |
| Dispatch, confirmation, authorization, and callbacks | 109 |
| Clock timer handler | 174 |
| Recent Messages by sender or group | 35 |
| All other handler bodies | 70 |
| Total | 433 |

The voice loop is already separate in the source. The generator expands each handler inside its dispatch branch in the released workflow. The timer handler reads `seconds_total`, then emits an If block and a literal Start Timer action for each of 43 durations. The app enforces the same list before dispatch. An ordinary 90-second request can pass the routing eval and still get a follow-up on the phone. The clock problem is the duration binding. Moving the handler into its own workflow would remove those 174 rows from the router while preserving the current limitation inside the handler. [Generator](../gen.py) · [Timer manifest](../../ControllrCore/Sources/ControllrCore/Actions/start_timer.json) · [App preparation](../../App/NativeExecutor.swift)

### Proposed responsibilities

- **Controllr** owns clarification, input validation, action ordering, and receipts. Resolve handler names from the registry and offer one pending action at a time for authorization. Lookups that the model needs can resume the same turn with their tool call IDs. The router can run that same cycle for each successive action.
- **Controllr Voice** dictates and speaks, shows the Controllr confirmation when required, calls a registered handler with its inputs, and returns the result. Use one common authorization and completion path. Resolve the handler before marking the action Started, so a missing handler is visible in the thread.
- **Each handler** accepts a documented payload, uses native actions, and returns its result to the caller. The standalone timer handler takes seconds. The Messages handlers own the phone-side query. Controllr keeps recipient selection and verifies lookup results before passing them to the model.

The current wire format returns a single action. The provider adapters normalize one tool call, and the app only authorizes the action on the latest thread entry. Native writes configured to run directly can also execute inside Decide. Keep the current one-action contract while proving the runner. Then stage writes in Controllr and advance any ordered list through the same authorization and receipt flow. Test confirmation, cancellation, failure, and duplicate callbacks before permitting a second write in one request. Preserve tool call IDs across the three adapters if the model contract changes. [Router](../../ControllrCore/Sources/ControllrCore/Router.swift) · [Thread and receipt handling](../../App/ConversationStore.swift)

### First phone checks

1. **Prove nested Shortcut dispatch with a manually authored pair.** Have an echo handler receive a dictionary and return its value with Stop and Output. Use the editor's Run Shortcut action with a fixed target first. Repeat twice after Always Allow, then test a variable selected from registered names and compare the editor export with the generated plist. Check missing handlers and run the same caller again from the normal voice entry point before migrating actions. The old generated fixed and variable callers both prompted on every run. Apple's guide recommends Run Shortcut for calls between shortcuts. [Prior device result](../../DECISIONS.md) · [Apple Shortcuts guide](https://support.apple.com/en-ng/guide/shortcuts/apd624386f42/ios)
2. **Test Clock duration in the isolated timer workflow.** Use the editor to attach a runtime value in the Start Timer field if the editor supports it, export that working or testable action, and compare it with the generated bindings. Try 37, 83, and 90 seconds and check the actual Clock countdown and whether a picker appears. The 83-second literal is the control. Keep the existing Clock path as the fallback while validating arbitrary durations. [Prior Clock probes](#probe-files)
3. **Evaluate AlarmKit if Clock's binding still fails.** Apple's iOS 26 timer API accepts numerical seconds and can schedule a Controllr-owned system timer. The call type-checked against the iOS 26.5 SDK with the project's iOS 17 deployment target and an availability guard. A shipping countdown needs alarm authorization and a Live Activity in a widget extension. Check the lock-screen countdown, alert and stop controls, and whether the app-owned timer experience meets the user's expectations for Clock. [AlarmKit timer API](https://developer.apple.com/documentation/alarmkit/alarmmanager/alarmconfiguration/timer%28duration%3Aattributes%3Astopintent%3Asecondaryintent%3Asound%3A%29) · [Apple's countdown implementation](https://developer.apple.com/videos/play/wwdc2025/230/)

After the first two checks, move one read and one write behind the same runner. A clipboard fixture verifies the result returns to the original thread. Timer verifies an actual single side effect. Then check two voice follow-ups, thread resume, and a cancelled text before promoting the new entry point. Keep static dispatch as the fallback if nested permission prompts persist.

Desktop verification passed plist lint, exact workflow regeneration, 41 core tests, and 12 iOS simulator tests. The checks above still require physical-device verification.

### Modular candidate results

Voice 2 generates 86 actions and 13 independent handlers. There is one dynamic Run Shortcut call and one app completion callback in the router. The echo probe passed a dictionary and returned its value twice on macOS 26.5.2. The first run requested permission and the second returned the value directly. The iPhone permission check is pending.

The build stages arbitrary durations through AlarmKit when permission is granted. The existing literal Clock handler is isolated as a fallback and Voice 1.6 still regenerates byte for byte. The app passed 41 core tests and 13 iOS 26.5 simulator tests, launched on iOS 18.6, and installed on the paired iPhone. The 86-step router passed static validation. All 14 workflow files passed plist lint and were signed. Physical validation still needs the clipboard round trip, native timer countdown and stop control, and a cancelled text. [Build and install](router.md#controllr-voice-2-candidate)

Voice 2.1 adds 13 steps for a dynamic choice list and shares the same Decide call for selected and dictated answers. The Mac probe displayed `Stretch` and `Walk` from a decision dictionary and returned `Walk` when selected. Fixture tests passed across OpenAI, Anthropic, and OpenRouter. iPhone selector verification is pending. [Voice 2.1 protocol](router.md#controllr-voice-21)

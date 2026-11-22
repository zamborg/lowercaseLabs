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

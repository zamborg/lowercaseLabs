---
name: controllr-shortcuts
description: Maintain the Controllr Voice shortcut and add or audit Controllr phone actuators. Use for dictation, thread resumption, action manifests, Shortcut handlers, App Intent callbacks, and on-device routing checks.
---

# Controllr Voice and phone actions

Keep one installed voice entry point. The voice loop captures speech, submits a turn, speaks responses, and listens again when Controllr asks a question. Controllr owns thread history, model prompts, validation, action preferences, and receipts. The action dispatcher runs an authorized handler and returns the result to the current thread. Read [router protocol](../../../shortcuts/specs/router.md) before changing that contract.

## Add or change an actuator

1. Start with the action's manifest in `ControllrCore/Sources/ControllrCore/Actions/`. Define its inputs, operation, permissions, confirmation default, and a routing example.
2. Add an executor case in `App/NativeExecutor.swift` for native APIs, or register a handler in `shortcuts/gen.py` for Shortcuts actions. `shortcuts/modular.py` builds registered handlers into independent workflows for Voice 2. A handler consumes the validated payload and returns any read result. Use the shared authorization and completion path. Leave dictation and conversation state out of action handlers.
3. Exercise the requested action with its actual inputs. Routing evals can pass even when Clock or another system action mishandles the inputs. Regenerate and sign a test shortcut for handler changes. Check its real effect or returned value on the target iPhone before making it the usual entry point.

## Change or audit the voice loop

- Preserve new and resumed threads, consecutive follow-ups, and reads that return to the same model turn. Test confirmation and cancellation without sending a real message or placing a call unless requested.
- Keep prompt and conversation changes shared across providers. The effective prompt comes from `ControllrCore/Sources/ControllrCore/Prompt.swift` and can be inspected in Controllr. Follow [provider parity](../../../AGENTS.md) when changing the model contract.
- Compare the generated action count and device behavior. Record new platform findings in the audit or decision log. [The Shortcut audit](../../../shortcuts/specs/audit.md) captures the Clock duration experiments, nested Shortcut permission prompts, and shared-completion device checks.

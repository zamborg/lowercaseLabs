# controllr status (2026-11-21)

## Works on device
Answers/speak, call, send message (no clarify), alarm, timer (voice router), notes (Bear URL), open URL, math/time questions. Evals 52/52 (gpt-6-luna, ~1s p50).

## Broken / unverified
- **Clarify loop**: Siri asks the question but the follow-up answer fails (`requestValue` inside Decide). Fallback plan: unrolled dictate -> Decide(context) in the router shortcut, or drop clarify for v1 and speak "say it again with the detail".
- Not yet run end-to-end: Focus, media, flashlight, open_url_confirm, reminders, directions, ride (Uber link unverified), Siri phrase "controllr", Lock Screen/Action Button.
- Notes: the iPhone had Notes uninstalled (Unknown Action cause). Bear is the notes target now.

## Priorities (user)
1. NLP loop for the agent (clarify/loopback with context)
2. Shortcuts for actions (use shortcuts-playground plugin; uv python 3.12; builder agent)
3. Extensibility: one manifest per action -> registry, tool schema, evals, router branch
4. Agent state in controllr: persistent/queryable memory, skills, notes

## Simplify proposal
Keep the Shortcuts layer tiny: router + only actions that truly need Shortcuts (send message, media, flashlight, focus, timer, alarm). Everything else native/URL in the app. Drop test shortcut clutter (Probes, Alt, Txt, If tests, Note variants, Test *).

## Cleanup owed
Delete stray 15-minute timers in Clock; rotate the API key; remove test shortcuts on phone/Mac.

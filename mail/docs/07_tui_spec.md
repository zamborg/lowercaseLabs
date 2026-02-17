# TUI spec

Minimal TUI to test flow, not to ship UI.

## UX loop

- Single-email viewport, keyboard-only.
- Keybinds (MVP):
  - j/k: next/prev
  - a: archive
  - d: delete
  - s: snooze (opens small menu; agent suggestions shown first)
  - r: remember (asks why optional)
  - g: delegate reply (opens instruction prompt)
  - Enter: open thread context
  - ?: help
  - !: break-glass compose (deliberately multi-step confirm)

## Delegate reply flow

1. user hits g
2. prompt: Tell your assistant what to do
3. show draft when ready:
   - y send
   - e edit instruction + regenerate
   - c cancel

## Notes view

- n: open notes
- search within notes: /

Implementation preference:

- Python + textual or rich for speed.
- TUI calls the API only; no direct DB.

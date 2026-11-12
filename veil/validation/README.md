# Native TextEdit masking validation

Validated November 12, 2026 with Veil's standalone, locally built app and its own approved Accessibility access. All fixtures were synthetic. Computer use operated TextEdit and read Veil's native status; this was not the earlier JavaScript-only detector harness.

| Case | Expected masks | Native result |
| --- | --- | --- |
| Unrelated text | 0 | 0 |
| Five consecutive characters | 0 | 0 |
| Six consecutive characters | 1 | 1 |
| Fragment interrupted by a space | 0 | 0 |
| Full synthetic key | 1 | 1 |
| Full key after emoji | 1 | 1 |

Additional checks:

- Twelve adjacent occurrences of the synthetic key wrapped over three visible lines; Veil reported three masks and zero whole-element fallbacks.
- A key at the top of a long document produced one mask. Scrolling to the bottom removed the match and produced zero masks, even though full document text remained available in the computer-use accessibility snapshot.
- Restoring the combined fixture produced three masks. Pause reported “Paused. Masks removed.” Resume produced three masks again.
- Source text remained intact in accessibility readback after masking.
- The user confirmed on the laptop that three black asterisk masks were aligned over the six-character fragment and both full-key occurrences, while five-character and interrupted fragments remained visible.

[Machine-readable results](native-textedit.json) · [Native scan status screenshot](native-scan-status.png)

The screenshot records Veil's status, not a composited desktop view of the masks. Per-app captures omit the separate mask windows; desktop alignment was verified by the user. Clipboard behavior, fullscreen/multi-display compatibility, and iTerm2 remain unverified. Both terminal applications were blocked by the computer-use app policy, so no terminal access was attempted through another mechanism.

During setup, rebuilding the ad-hoc-signed app left a stale Accessibility approval. Toggling and restarting did not resolve it. Removing the old Veil entry and adding the current app bundle restored access. The tested build was then left unchanged.

Computer-use operations did not make the target globally frontmost. The user manually focused TextEdit; Veil's focus diagnostic confirmed that bundle ID before the scan. Polling was moved to common run-loop modes and monitoring now uses an activity assertion to avoid App Nap while allowing idle system sleep.

# Accessibility detector spot check

Date: November 12, 2026.

Result: GO for a scoped detector prototype. Terminal compatibility remains untested.

## Actual experiment

The Mac was accessible after the user unlocked it. Computer use successfully listed native apps. Attempting to select `com.apple.Terminal` was rejected by the computer-use tool: “Computer Use is not allowed to use the app 'com.apple.Terminal' for safety reasons.” No alternate mechanism was used to access Terminal.

Instead, opened a fresh TextEdit document, changed its content through UI accessibility actions, and read back a fresh, complete accessibility snapshot after each change. Ran a deterministic six-character sliding-window matcher against each returned snapshot, not against the input fixture. Full snapshots were necessary to avoid removed text from accessibility diffs contaminating later tests. A screenshot confirmed the final synthetic key was visibly rendered.

Configured synthetic key: `N8v2K6p4R9t3W5x7`, label: `Synthetic key A`. Its eleven overlapping six-character substrings were indexed in a map from fragment to key labels.

| Displayed content | Expected detection | Actual |
| --- | --- | --- |
| Ordinary harmless text. | No | No |
| p4R9t | No: five characters | No |
| p4R9t3 | Yes: six characters | Synthetic key A |
| p4R9 t3 | No: interrupted fragment | No |
| N8v2K6p4R9t3W5x7 | Yes: full key | Synthetic key A |

All five cases passed. Each non-baseline fixture was also checked for exact presence in the returned UI accessibility snapshot. The final synthetic fixture remains in an unsaved TextEdit document titled “Untitled 2.” No real secrets were used.

## What this establishes

- Remote native UI interaction works while the Mac is unlocked.
- TextEdit exposes displayed text through accessibility.
- The proposed fragment lookup detects the six-character boundary and identifies the configured key label.
- No OCR or LLM was used for matching.

## What still needs validation

- Terminal, iTerm, and the user's actual Claude Code surface. Terminal was blocked by this tool's app policy, not shown to lack accessibility support.
- A standalone app's own Accessibility permission and AX traversal; this test used the computer-use tool's existing access.
- Visible-range filtering. A full AX text value can include off-screen text, and snapshots can repeat the focused element. The prototype must avoid confusing those with multiple visible exposures.
- Continuous monitoring, notification delivery, latency, CPU use, alert deduplication, and lock/unlock handling.

Next implementation step for Sol: build the small detector and repeat these fixtures in the actual terminal with an authorized test surface, before expanding application coverage. This is evidence for the basic extraction-and-matching mechanism, not evidence that all on-screen text is detectable.

# Veil 0.2.0 settings validation

November 13, 2026. Native app UI exercised through computer use on Apple Silicon. Synthetic/no secret data only.

Verified in the actual settings window:

- The application menu lists installed apps by name, including apps that are not running. Terminal and iTerm2 appeared in the list; neither was launched or accessed.
- Selecting TextEdit retained the already-selected iTerm2, producing the summary “iTerm2, TextEdit”.
- All apps mode displayed “All apps — active window”. Choosing iTerm2 from that mode returned to iTerm2-only selection.
- Entering 8 updated both the number field and stepper.
- Quitting and relaunching preserved All apps mode and the minimum length of 8. The secret editor was empty after launch.
- Entering 0 restored 8 and displayed a whole-number range validation message.
- Entering 4 was accepted, and incrementing the stepper updated both controls to 5.
- Settings were restored to iTerm2-only, minimum 6, masking paused, and no secrets configured.
- The [settings screenshot](settings-0.2.0.png) was inspected for control visibility and clipping.

Nine automated core tests pass. Additional coverage includes thresholds 1/4/6/8/16 with exact boundary checks, UTF-16 offsets after emoji, rejecting a threshold longer than a secret, selected-app membership, All apps membership, and self-exclusion.

Both arm64 and x86_64 release builds succeed. The universal ZIP was extracted, its app signature and architectures checked, and its SHA-256 checksum verified.

No new cross-app overlay session was started for 0.2.0, so live threshold changes and active-app switching while masking have not been exercised end to end. The earlier [0.1.0 TextEdit validation](README.md) remains the evidence for desktop overlay behavior. iTerm2 compatibility is still unverified. All apps mode follows the active window and does not scan inactive visible windows simultaneously.

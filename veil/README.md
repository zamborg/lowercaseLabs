# Veil

A small native macOS app that covers known secrets with opaque asterisk masks. This is the first standalone prototype following the [accessibility detector experiment](../research/detector-spot-check.md).

For personal testing on another Mac, use the [universal development build and setup instructions](releases/README.md), or build from source below.

It reads the active window of your selected apps through Accessibility, matches consecutive fragments of configured secrets, and positions click-through panels over those matches. The minimum matching length defaults to six and is configurable. It never edits the source application. Full matches merge into one mask per visual line. If a known match has no usable character geometry, Veil covers the containing text element and reports that fallback.

## Build and run

Requires macOS 13 or later and Xcode Command Line Tools. No third-party dependencies.

```sh
cd veil
swift test
bash scripts/build-app.sh
```

Open `build/Veil.app` in Finder. The build script creates a local, ad-hoc-signed app; public signing, notarization, installation, and automatic updates are future distribution work. Generated builds are ignored by Git.

Rebuilding an ad-hoc-signed app can invalidate its existing Accessibility approval even when its Settings switch still appears on. After rebuilding, quit/relaunch and refresh that approval. If toggling it off/on is insufficient, remove the old entry and add the newly built app again.

1. Click **Accessibility access…** and enable Veil in System Settings → Privacy & Security → Accessibility. Return to Veil and start masking. This grants the app access to other apps' accessibility data; the app scans only the active window of an allowed app.
2. Open **Applications** and check one or more apps. Installed and running apps appear by name, including iTerm2 and Terminal. Select **All apps — active window** to follow the active app automatically. Veil excludes itself. **Refresh app list** discovers newly installed apps.
3. Set **Minimum matching substring** using the number field or stepper (1–1024, default 6). For example, at 8, a seven-character fragment does not match and an eight-character fragment does. Each configured secret must be at least that long. App selection and match length are saved between launches and apply immediately while masking; secret edits require pausing and restarting.
4. Enter one `label=secret` per line. Labels identify candidate keys. Secrets use printable ASCII characters without whitespace; values can include `=`. Matching is case-sensitive. Changing the minimum length while running is rejected if it exceeds any configured secret's length.
5. Click **Start masking**, then focus a selected app. The menu-bar indicator shows the number of active masks. Return to Settings to inspect the latest labels and coverage result.
6. Click **Pause** in the menu-bar menu to remove all masks. Quit also removes them.

App IDs, All apps mode, and minimum length are saved in local preferences. Secret values stay in process memory for the session. Veil does not save them, send them to a service, write them to logs, or include them in status messages. The configuration editor displays their values, so configure it before streaming. The source application still contains the original data, including copy/paste and scrollback.

## Synthetic TextEdit check

Click **Load TextEdit demo** to select TextEdit and populate a synthetic secret, then start masking. Open a disposable TextEdit document and paste:

```text
Unrelated: ordinary harmless text
Five characters: p4R9t
Six characters: p4R9t3
Interrupted: p4R9 t3
Full key: N8v2K6p4R9t3W5x7
Unicode prefix: 🔒 N8v2K6p4R9t3W5x7
```

Only the six-character fragment and the full-key occurrences should be covered. Change the text, scroll it, resize the window, switch apps, and pause. Masks should track the text, disappear when switching apps, and disappear when paused. Copying the text should still yield the original values. Use a desktop-level observation when checking the visible overlays: window-only screenshots can omit the separate panels.

## Scope and limits

This is best-effort visual concealment. Accessibility polling occurs every 200 ms with one scan in flight. The timer runs in common run-loop modes, and an activity assertion prevents App Nap during monitoring while still allowing idle system sleep. A key can appear before a mask does. Window-only broadcasting can omit the mask panels. A partially emitted secret shorter than the configured minimum remains visible. The app is not a pre-render terminal filter or a sanitized broadcast pipeline.

Only the active application's focused window is scanned, if selected or permitted by All apps mode. All apps is automatic active-app switching, not simultaneous coverage of every visible window. Inactive windows, menu bars, notifications, images, secure fields, and custom rendering are outside the current scope. Editable text areas must expose a visible character range. Missing support, traversal limits, and overly large text are reported as incomplete coverage. An apparently completed AX scan does not certify that every visible pixel was inspected. Spaces, fullscreen, multiple monitors, and unusual accessibility coordinate implementations need compatibility validation.

The scan is bounded to 800 elements and 100,000 UTF-16 units per text element. Character-range queries stop at the scan deadline and fall back to whole-element masking when appropriate. At most 128 mask panels are displayed. Multiple keys sharing a fragment are all reported as candidates. Newline-separated fragments are not joined; normal visual wrapping uses the original AX string ranges.

## Validation status

- `swift test`: nine tests pass, covering configurable thresholds, short-secret validation, app selection and self-exclusion, interruptions/case sensitivity, overlapping fragments, repeated occurrences, shared-key fragments, UTF-16 offsets after emoji, invalid configuration, and display-coordinate conversion.
- Release app builds and is ad-hoc signed successfully.
- Version 0.2.0 settings UI check passed: multiple app selection, All apps, thresholds above/below six, invalid input, the stepper, and persistence across relaunch. See the [settings validation](validation/settings-0.2.0.md). A new cross-app overlay session was not started for this settings update.
- Version 0.1.0 native TextEdit test passed: six boundary cases produced the expected mask counts; source text stayed intact. A repeated key wrapped over three visual lines produced three masks without whole-element fallbacks. Scrolling a key off-screen changed the count from one to zero. Pause and resume behaved as expected. This remains historical overlay validation; version 0.2.0 changes are covered by the core tests and settings UI check.
- The user visually confirmed three aligned asterisk masks over the six-character fragment and two full-key occurrences; five-character and interrupted fragments remained visible. See the [native validation report](validation/README.md).
- Computer-use actions can operate an app in the background without making it the Mac's frontmost app. For this foreground-only monitor, manually focus TextEdit on the laptop. Veil's waiting status reports the actual foreground bundle ID to help diagnose this condition.
- Both Terminal and iTerm2 were blocked by the computer-use tool's app safety policy. Neither has been compatibility-tested. There is no terminal-specific integration or bypass in this prototype.

Architecture: `VeilCore/Matcher.swift` holds the fragment index and pure range calculations; `AccessibilityScanner.swift` reads text and glyph geometry; `Overlay.swift` owns the opaque panels; `main.swift` owns configuration, permissions, polling, and pause controls. No AX setter is used on the source application.

---
name: iphone-mirroring-claude
description: Operate the user's physical iPhone on the local Mac with Claude's macOS computer-use tools and Apple iPhone Mirroring. Use for mirrored app navigation, screenshots, and reconnection tests in Claude Code.
---

# iPhone Mirroring in Claude

Read the [shared iPhone Mirroring skill](../../../skills/iphone-mirroring/SKILL.md) for device setup, navigation, and reconnection behavior. Use Claude's computer-use tools to implement that workflow on the local Mac.

1. **Load the tools.** If the `mcp__computer-use__*` tools are deferred, load them all in one `ToolSearch` call with query `computer` and `max_results: 60`. If none exist, computer use is off for this session. Report that as the blocker and end the run. Turning it on is the user's choice (Claude desktop app, Settings, Computer use). Do not substitute the browser tools, because they cannot see iPhone Mirroring.
2. **Request access.** Call `request_access` for **iPhone Mirroring** and wait for the user's approval if prompted. Use the bundle identifier and access tier returned for later calls. In an unattended run that requires approval, report the blocker and end the run.
3. **Open the window.** Use `open_application` for iPhone Mirroring. A process can be running before it has a capturable window. If a screenshot shows no window, ask the user to bring it forward in an interactive run. An unattended run should report that and end.
4. **Prefer background tools.** Use `app_screenshot`, `app_click`, `app_type` and the app's key-press tool, aimed at iPhone Mirroring. If keyboard input fails to reach the phone, use the Spotlight fallback below. The accessibility tree usually shows only the Mac window or toolbar, so ground every click in a fresh screenshot of the phone UI. Take a new screenshot after each input.
5. **Open Spotlight.** In the background, `app_menu` with path `["View", "Spotlight"]` opens it. Wait for the search field to appear before typing. Background `app_type` and `app_key` did not reach the focused search field in Claude's interactive test. If that happens, call `request_full_control` with reason `keyboard_blocked`. Once access is granted, call `app_release` and bring iPhone Mirroring forward. Send Command-3, wait for the field to appear, and press the query keys one at a time with `computer_batch` key actions. Typing `eight` found Eight Sleep as the Top Hit. Verify the intended result in a screenshot before pressing Return. Call `release_full_control` after opening the app or if the attempt ends early.
6. **Handle prompts.** Use the visible Try Again or reconnect control when the app offers it. If Apple requires authentication, asks you to unlock the iPhone, or reports that the phone is in use, let the user resolve it in an interactive run. In an unattended run, report the blocker and end the check.

Run any scheduled reconnect check on the same local Mac. Verify that the scheduled run itself has computer-use access and the mirroring window.

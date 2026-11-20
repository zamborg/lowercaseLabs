---
name: iphone-mirroring-claude
description: Operate the user's physical iPhone on the local Mac with Claude's macOS computer-use tools and Apple iPhone Mirroring. Use for mirrored app navigation, screenshots, and reconnection tests in Claude Code.
---

# iPhone Mirroring in Claude

Read the [shared iPhone Mirroring skill](../../../skills/iphone-mirroring/SKILL.md) for device setup, navigation, and reconnection behavior. Use Claude's computer-use tools to implement that workflow on the local Mac.

1. **Load the tools.** If the `mcp__computer-use__*` tools are deferred, load them all in one `ToolSearch` call with query `computer` and `max_results: 60`. If none exist, computer use is off for this session. Report that as the blocker and end the run. Turning it on is the user's choice (Claude desktop app, Settings, Computer use). Do not substitute the browser tools, because they cannot see iPhone Mirroring.
2. **Request access.** Call `request_access` for **iPhone Mirroring** and wait for the user's approval. Use the bundle identifier the response returns for later calls. The app is granted at tier "full", so clicks and typing work. In an unattended run where nobody can approve, report the blocker and end the run.
3. **Open the window.** Use `open_application` for iPhone Mirroring. A process can be running before it has a capturable window. If a screenshot shows no window, ask the user to bring it forward in an interactive run. An unattended run should report that and end.
4. **Prefer background tools.** Use `app_screenshot`, `app_click`, `app_type` and the app's key-press tool, aimed at iPhone Mirroring. Use the full-screen tools only if the user asks for them or a tool error directs you there. The accessibility tree usually shows only the Mac window or toolbar, so ground every click in a fresh screenshot of the phone UI. Take a new screenshot after each input.
5. **Open Spotlight.** In the background, `app_menu` with path `["View", "Spotlight"]` opens it. Wait about two seconds for the animation to finish, then take a fresh screenshot. Typing during the animation is dropped. Background `app_type` and `app_key` did not reach the phone's search field in the Claude test, even with the field focused. Keyboard input needs full-screen control: call `request_full_control` with reason `keyboard_blocked`, `app_release` the app, and bring iPhone Mirroring forward. Then send Command-3, wait two seconds, and press the query keys one at a time with `computer_batch` key actions. `type` also left the field blank. Typing `eight` found Eight Sleep as the Top Hit. When a screenshot shows the intended app highlighted, press Return. Call `release_full_control` as soon as the typing is done.
6. **Handle prompts.** An Apple authentication prompt, a "device in use" notice, or a locked-phone message is a blocker. Ask the user to resolve it in an interactive run. In an unattended run, report it and end the run. Use the visible Try Again or reconnect control when one is shown.

Run any scheduled reconnect check on the same local Mac. Verify that the scheduled run itself has computer-use access and the mirroring window. Access granted in an interactive chat does not carry over.

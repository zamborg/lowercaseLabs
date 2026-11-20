---
name: iphone-mirroring-codex
description: Operate the user's physical iPhone on the local Mac with Codex Computer Use and Apple iPhone Mirroring. Use for mirrored app navigation, screenshots, and reconnection tests in Codex.
---

# iPhone Mirroring in Codex

Read the [shared iPhone Mirroring skill](../../../skills/iphone-mirroring/SKILL.md) for device setup, navigation, and reconnection behavior. Use Codex Computer Use to implement that workflow on the local Mac.

1. Select iPhone Mirroring with `cua.getApp("iPhone Mirroring")`. If it reports `cgWindowNotFound`, try `cua.getApp("/System/Applications/iPhone Mirroring.app")`. A process can appear as running before it has a capturable window. If the app still has no window during an interactive run, ask the user to bring it to the foreground. An unattended run should report the window error and end.
2. If the accessibility tree contains only the Mac window or toolbar, use `getScreenshot()` or `getAXStateAndScreenshot()` to inspect the phone UI. Use the visible icon or control for coordinates and get a fresh observation after input.
3. Open Spotlight with `pressKey("super+3")`. If `typeText` leaves the field blank, send the query with individual `pressKey` calls. This worked for finding Eight Sleep with `eight`. When the screenshot shows the intended app as the highlighted result, open it with `pressKey("Return")`.

Run any scheduled reconnect check on the same local Mac. Verify that the scheduled run itself has access to Computer Use and the mirroring window.

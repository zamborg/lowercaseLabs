---
name: eight-sleep-control
description: Shared Eight Sleep iPhone app workflow for reading or setting the user's Pod temperature, applying a user-defined profile, or running authorized overnight checks and clock-time adjustments through iPhone Mirroring.
---

# Eight Sleep control

Use the [iPhone Mirroring skill](../iphone-mirroring/SKILL.md) to open Eight Sleep on the user's physical iPhone. Open the Pod temperature controls. Verify the selected Pod side on a split Pod and the displayed unit or level scale before reading or writing a target.

An **Off** temperature card means there is no active target to read. Record **Off** during a read-only check. Turning the Pod on is a separate action.

## Temperature profiles

- **Built-in schedule.** If the user's profile maps to Bedtime, Night, Dawn, and an optional thermal alarm, read the existing phase targets and use the temperature schedule to apply the values the user requested. Use the phase labels shown in the app. Verify the saved phase values. These preferences apply on future nights, and Autopilot can make its own adjustments. [Eight Sleep temperature schedule](https://help.eightsleep.com/en_us/what-is-the-temperature-schedule-BkRoPvR66)
- **Immediate adjustment.** Use the live temperature dial to apply an explicitly requested target for the current night. Re-read the displayed target after changing it and distinguish it from any measured temperature the app shows. [Eight Sleep temperature dial](https://www.eightsleep.com/eu/blog/temperature-dial-how-to-adjust-temp/)
- **Clock-time profile.** Use [the overnight profile workflow](references/overnight-profile.md) for checks or live adjustments at specified times. Use the same workflow when running one scheduled check. Verify the input and reconnection path before relying on unattended writes.

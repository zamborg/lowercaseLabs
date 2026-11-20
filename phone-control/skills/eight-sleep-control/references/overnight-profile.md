# Overnight profiles

Record the user's Pod side, local timezone, start and end times, nights to run, displayed temperature units, and check interval. For a run that can write, record the approved absolute targets, the time window for each target, and whether the user wants a specific final target. For a window crossing midnight, associate the run with the evening on which it starts and resolve each check to a full local date and time. Show the resulting timeline before enabling any requested schedule. It is valid to request read-only checks, temperature changes at specific times, or both.

For instance, a read-only trial might check at 23:00, 23:30, and 00:00 local time for one night. A temperature profile could add explicitly approved targets at 23:00 and 00:00, leaving the 23:30 check read-only. No example represents a recommended sleep temperature.

## Validate the runner

Use a local scheduler with access to the same Mac desktop and computer-control tool. Test a read-only scheduled check after the intended idle interval, using the same Mac power and lock configuration as the planned overnight run. Check whether the scheduled run can reach the app and read your side's target or verify that the Pod is off. Live access to the desktop in an interactive chat alone does not establish that the scheduler has access. If a scheduled run cannot complete that read, report the blocker before enabling overnight writes.

## Run a check

1. Verify that the check is still within its specified time window. Skip expired checks. Run one check at a time so that two jobs cannot manipulate the same window concurrently.
2. Use iPhone Mirroring to verify the selected side and read the current target or **Off** status. Read any separately labeled status or measured temperature the user asked to track. Log the time and the outcome. When the connection, account, side, or app state cannot be verified, record why and end the check.
3. For a write, use the approved absolute target for that check. If the target is already selected, record it and finish. Apply the target through the live dial, then verify the displayed target and log the result. If the write result is uncertain, re-read the app state before attempting another input. A later read-only check or a retry of an expired check should never reapply a past target.

At the end of the window, stop running checks and report observations, verified changes, and any skipped or failed checks. Apply a final target only if the user included one. Autopilot may subsequently move the live target, so report what the app showed at the time of each check. For a profile that depends on a live sleep signal, first verify that Eight Sleep exposes the signal in the current app. Historical charts and elapsed time cannot establish a live sleep stage.

# Decisions

- Plan §4 (repository layout) was empty in the source doc; using App/, shortcuts/{specs,src,signed}/, evals/ now and ControllrCore/ in Phase 1.
- DEVELOPMENT_TEAM is not set in project.yml (not guessed); set locally for device builds.
- Router has no If: the If condition could not be generated reliably (PLAN §2/§5). Decide ALWAYS returns `shortcut` + `input`; spoken results (done/respond/clarify/error) run the `Controllr · Speak` module with {speak}. Confirmation alerts (when confirm is true) must be handled another way in Phase 4.
- Run Shortcut prompts "Allow to run another shortcut?" on EVERY run (static or by variable, Always Allow doesn't stick), so the router runs modules inline instead. Verified on device: If chain on the module id, no prompt after the one-time "Allow to run actions from controllr" per shortcut.
- If step: inputs must be Text (add a Text action that wraps the dictionary value); format = WFCondition 4 + WFConditionalActionString + WFInput variable. A raw dictionary-value input leaves the value empty.
- Text-input test routers (`Controllr · Alt`, `Alt (network)`) use a fixed "hello" and show results in an alert so they can be driven and read over iPhone Mirroring.
- Phase 0 measured: network call 998 ms / 815 ms (list-models), Decide runs from Shortcuts without opening the app.
- Phase 1 eval baseline (v1 prompt): gpt-5.4-mini 94%, gpt-5.4-nano 90%. Misses were over-clarifying; prompt v2 (don't clarify when a name/relationship/place is given) -> both 98%. Models come from the user's list-models, never hard-coded. See evals/RESULTS.md.
- Native modules that need to open something (tel:, Maps, Uber) return `open_url` / `open_url_confirm` decisions; the router opens the URL (a background App Intent can't). Confirmation is encoded in which id is returned, so the router needs no runtime confirm If.
- Shortcut modules that need resolution (send_message: contact name -> number) go through `ModuleExecutor.prepare` inside Decide before the decision is returned.
- Uber deep link format is unverified (no network access to Uber's docs from the agent); check on device.

## Phase 4 (router + modules)
- Router is inline `If Module Text is <id>` branches, one per module; no Run Shortcut. Voice router = `Controllr · Router` (Dictate Text); `Controllr · Test *` are the same flow with fixed text + alert instead of speech, for testing via iPhone Mirroring.
- OpenAI: send `reasoning_effort: "none"` (gpt-6-luna rejects tools while reasoning is on); retry without it if the API 400s on that param. Eval gpt-6-luna: 52/52 (see evals/RESULTS.md), p50 ~1.0 s.
- Timer: the Start Timer duration field cannot take a variable (it shows "Ask each time", a 15:00 picker). Router enumerates durations (10/15/20/30/45 s, 1–120 min) as literal branches keyed on `seconds_total`, computed in `NativeExecutor.prepare`. Minute values use Unit "min"; Unit "sec" is unverified.
- Verified identifiers/params on device: pausemusic (WFPlayPauseBehavior), skipforward/skipback, timer.start, flashlight (WFFlashlightSetting), openurl, dnd.set (Enabled + FocusModes{DisplayString,Identifier}).
- NOT working yet: set_alarm (action `com.apple.mobiletimer-framework.MobileTimerIntents.MTCreateAlarmIntent` is recognised but its Time/Label param keys are unknown), create_note (no identifier found), send_message recipients (literal formats all render empty; variable-based form in the router is untested). These branches say "That isn't set up yet" (alarm/note). Needs an exemplar exported from a hand-built shortcut.
- Reading a stuck editor list through iPhone Mirroring: scroll gestures don't work; deleting the top actions shows the ones below.
- Importing the same shortcut name repeatedly via Replace creates numbered copies on the phone ("Test Timer 1/2/3", "Router 2"); the newest number is current.
- Exemplar export (File → Export, then `aea decrypt -sign-pub <leaf cert pubkey from auth data>` and `aa extract`) gave the real param shapes. Alarm (`MTCreateAlarmIntent`: dateComponents "HH:mm", name) and Send Message (recipients as a text variable, `IntentAppDefinition` com.apple.MobileSMS) WORK on device.
- create_note: the Mac-exported `com.apple.mobilenotes.SharingExtension` action (even byte-for-byte, re-signed) is an Unknown Action on the iPhone ("Note Exemplar/Wrap/Static/Var" all fail). Notes needs a different iOS identifier, or a non-shortcut route. Router currently still emits the Mac form; treat create_note as broken.
- Timer branches only live in the voice Router; test routers carry one (300 s) so they stay small.

# Decisions

- Plan §4 (repository layout) was empty in the source doc; using App/, shortcuts/{specs,src,signed}/, evals/ now and ControllrCore/ in Phase 1.
- DEVELOPMENT_TEAM is not set in project.yml (not guessed); set locally for device builds.
- Router has no If: the If condition could not be generated reliably (PLAN §2/§5). Decide ALWAYS returns `shortcut` + `input`; spoken results (done/respond/clarify/error) run the `Controllr · Speak` module with {speak}. Confirmation alerts (when confirm is true) must be handled another way in Phase 4.
- Run Shortcut prompts "Allow to run another shortcut?" on EVERY run (static or by variable, Always Allow doesn't stick), so the router runs modules inline instead. Verified on device: If chain on the module id, no prompt after the one-time "Allow to run actions from controllr" per shortcut.
- If step: inputs must be Text (add a Text action that wraps the dictionary value); format = WFCondition 4 + WFConditionalActionString + WFInput variable. A raw dictionary-value input leaves the value empty.
- Text-input test routers (`Controllr · Alt`, `Alt (network)`) use a fixed "hello" and show results in an alert so they can be driven and read over iPhone Mirroring.
- Phase 0 measured: network call 998 ms / 815 ms (list-models), Decide runs from Shortcuts without opening the app.

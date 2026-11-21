# Decisions

- Plan §4 (repository layout) was empty in the source doc; using App/, shortcuts/{specs,src,signed}/, evals/ now and ControllrCore/ in Phase 1.
- DEVELOPMENT_TEAM is not set in project.yml (not guessed); set locally for device builds.
- Router has no If: the If condition could not be generated reliably (PLAN §2/§5). Decide ALWAYS returns `shortcut` + `input`; spoken results (done/respond/clarify/error) run the `Controllr · Speak` module with {speak}. Confirmation alerts (when confirm is true) must be handled another way in Phase 4.

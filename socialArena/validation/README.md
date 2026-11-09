# Validation evidence

Verified on 2026-11-09. [Machine-readable results](results.json) records experiment IDs, counts and aggregate API accounting. These are operational checks, not model capability rankings.

- 31 tests pass: game rules, private information, simultaneous decisions, crash/resume, budget contention, ratings and provider phase handling.
- 100/100 local Avalon games and 4/4 local Dominion games complete; every final state hash matches deterministic replay.
- One final Luna game reaches 13 decisions with zero invalid actions before an API HTTP 400 `invalid_prompt` rejection. It is incomplete and excluded from outcome ratings. No further paid retry was made after that rejection.
- Earlier development attempts exposed API input requirements, speech-schema clarity and an adapter bug concatenating commentary with final answers. The latter was fixed and regression-tested. Earlier failures are not evidence of poor model gameplay.

The compressed [evidence archive](evidence.tar.gz) contains the final three experiments' manifests, ledgers, results and replay journals, including the incomplete live game. Request caches and redundant generated HTML are omitted. Raw development attempts remain locally under `.runs/`, ignored by Git. All paid attempts plus a diagnostic call are included in aggregate accounting; uncertain reservations are distinguished from known token charges.

To inspect portable evidence from the repository root:

```sh
mkdir -p socialArena/.runs/evidence
tar -xzf socialArena/validation/evidence.tar.gz -C socialArena/.runs/evidence
PYTHONPATH=socialArena python3 -m social_arena report socialArena/.runs/evidence/avalon-release --bootstrap 2000
PYTHONPATH=socialArena python3 -m social_arena report socialArena/.runs/evidence/dominion-release --bootstrap 2000
PYTHONPATH=socialArena python3 -m social_arena report socialArena/.runs/evidence/luna-release --bootstrap 20
PYTHONPATH=socialArena python3 -m social_arena replay socialArena/.runs/evidence/avalon-release/games/g0000-0
```

Open each generated `report.html`; game links open interactive decision replays. Keep this checkout for hash-verified replay. The archive is analyst evidence and contains hidden roles and private observations; its contents must not be supplied to playing agents. Do not resume the archived paid experiment as a way to inspect it: `report` and `replay` make no API calls.

# Social Arena

A runnable Python harness for reproducible agent games: five-seat Avalon and the two-player Dominion second-edition First Game kingdom. It records what each seat knew, actions, errors, usage, checkpoints and exact deterministic replays. Local policies spend **$0**. This is an experiment tool, with no hosted service, training or arbitrary game import.

**Verified 2026-11-09:** 31 tests pass; 100/100 local Avalon and 4/4 local Dominion games complete. All 104 completed games replay exactly. A live five-seat Luna smoke executed 13 decisions with zero invalid actions, then stopped on a provider prompt rejection; a complete live-model game is **not** claimed. See [validation and portable evidence](validation/README.md).

Python 3.10+; no runtime dependencies. Run from the repository root:

```sh
PYTHONPATH=socialArena python3 -m unittest discover -s socialArena/tests -v
PYTHONPATH=socialArena python3 -m social_arena run --config socialArena/experiments/free-avalon.json --out socialArena/.runs/avalon
PYTHONPATH=socialArena python3 -m social_arena run --config socialArena/experiments/free-dominion.json --out socialArena/.runs/dominion
```

Open `<output>/report.html` locally. Each game links a decision-by-decision replay with public, seat and visibly labeled analyst-only omniscient views. The report shows actual completions, failures, cost, Brier diagnostics, raw side records, paired ratings and whole-scenario bootstrap intervals. Default bootstrap count is 2,000; `--bootstrap 20` is useful for a quick operational smoke, with correspondingly coarse intervals. Estimates from scripted agents validate the harness; they are not model capability evidence.

To resume the exact frozen experiment, omit `--config`:

```sh
PYTHONPATH=socialArena python3 -m social_arena run --out socialArena/.runs/avalon
PYTHONPATH=socialArena python3 -m social_arena replay socialArena/.runs/avalon/games/g0000-0
PYTHONPATH=socialArena python3 -m social_arena view socialArena/.runs/avalon/games/g0000-0 --seat 1
PYTHONPATH=socialArena python3 -m social_arena report socialArena/.runs/avalon --bootstrap 2000
```

The manifest freezes policies, seat assignments, setup seeds, code hashes, protocol and accounting limits. Any policy/config/code change requires a fresh output directory; retain the matching checkout for old replays. Request caches survive crashes. Durable decision commits replay from setup, verify each state hash and RNG state, and rebuild checkpoints without applying an action twice. A partial final journal line is discarded. Public/seat exports filter recipient visibility; raw logs, checkpoints, initial setup and request caches are private analyst artifacts and must never be passed to players. Policies receive only their isolated observation, its trimmed public history and legal action specification; they have no tools or shared memory.

The default schedule creates independent scenarios with reversed alignment assignments in Avalon and reversed first-player assignments in Dominion. Same-policy self-play is supported. Explicit `schedule` entries permit registered mixed rosters; their immutable alignment composition IDs include role-to-policy assignments. The [32-game pilot template](experiments/pilot-template.json) encodes 12 swapped scenarios and four self-play games per policy. It is local until actual provider policies and an approved positive budget are frozen. One-seat intervention and discussion ablation can use explicit schedules; set `discussion_rounds: 0` in `game_config` for no speech, including before assassination. Analyze different communication protocols as separate pools. Do not tune prompts on held-out reported scenarios.

Optional OpenAI policies use the Responses API. Keep `OPENAI_API_KEY` in the environment; never put it in experiment JSON. A policy definition can contain:

```json
{"kind":"openai","model":"gpt-6-luna","reasoning":"none","max_output_tokens":256,
 "input_usd_per_million":0.1,"output_usd_per_million":0.5,
 "price_source":"https://developers.openai.com/api/docs/models/gpt-6-luna"}
```

Verify prices for the exact model before use; these are explicit configuration values, not an automatic price service. Set top-level `allow_paid: true` and `max_usd` to the approved cap. Standard service tier is explicit. Unknown/nonfinite prices block dispatch. The scheduler reserves a conservative input-byte/token bound plus maximum output before each attempt and reconciles known token usage. Timeouts/server failures retain a full uncertain reservation. Retries and one schema repair are separately charged. A request that may have reached the provider is never silently redispatched after resume. Infrastructure failures and caps produce incomplete games; exhausted illegal actions produce policy failures. `max_decisions` caps each game; `max_total_decisions` caps the experiment. The runner is sequential; the persisted budget ledger also prevents concurrent reservations exceeding its configured cap. Do not run distinct experiments against one shared external account cap and assume their independent ledgers coordinate account-wide spending.

`probes: true` enables detached, structured P(evil) probes at vote phases for uninformed good seats. They use a copied permitted observation and separate request/cost entries, do not alter policy history, and exclude mechanically known labels. Merlin has no eligible unknown labels in this fixed role set. Only actual probe results contribute Brier diagnostics. This implementation uses bounded observation-only memory; there is no hidden chain-of-thought capture.

[RATINGS.md](RATINGS.md) explains regularized Bradley–Terry and Davidson ratings. Only complete reversed pairs enter primary comparisons, with explicitly marked self-play accepted separately. Mixed-policy team IDs describe the whole roster, not individual Elo. The report flags disconnected pools, few scenario blocks and policy-failure bias; it provides strength-prior sensitivity and attributable failure-as-loss sensitivity. Paid pilot expansion, measured-cost power analysis, one-seat uplift estimation and held-out calibration remain study work; no large paid study is bundled or implied.

[EXECUTION.md](EXECUTION.md) records implementation coverage and limits. [SOURCES.md](SOURCES.md) records rules, upstream methodology and the bounded reuse audit.

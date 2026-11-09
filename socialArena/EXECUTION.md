# Execution handoff

**This file specifies future work. No commands, interfaces, or reports below are implemented yet.** Deliver the stages in order; the current spike stops at these documents.

## 1. Freeze the contract and Avalon protocol

Use a five-player game: Merlin, two loyal servants, Assassin, one minion. Quest sizes are 2/3/2/3/3. Strict-majority team approval; five consecutive rejected proposals gives evil the win. Good seats can only submit success; evil can submit success or failure. One failure fails a quest in this five-player variant. After three successes, resolve the Assassin's choice before declaring good the winner. Verify fixtures against the [original rulebook](https://avalon.fun/pdfs/rules.pdf).

Benchmark communication rule: leader proposes a team, then two public rounds with one message per seat per round, at most 100 words per message. Rotate speaking order deterministically from the setup. Votes are collected against the same frozen information snapshot and revealed together. Quest cards are collected privately and only aggregate counts revealed. Before assassination, allow one additional public round. No private chat. These speech limits and timing are harness choices, not claims about the tabletop rules.

The game adapter exposes:

```text
reset(config, seed) -> State
pending_decision(state) -> DecisionBatch
observe(state, seat) -> Observation
action_spec(state, seat) -> SchemaAndConstraints
resolve(state, batch_of_actions, rng) -> StateAndEvents
outcome(state) -> Pending | TerminalResult
```

`DecisionBatch` names eligible seats, visibility, and sequential/simultaneous collection. `Observation` contains only that seat's permitted information. Authoritative state never enters a policy call. Schemas support constrained choices without enumerating every combination. Dominion reactions become additional decision batches, not hacks around a global turn counter.

**Acceptance:** scripted fixtures cover all terminal paths, vote ties, leader rotation, quest privacy, and role knowledge; simultaneous collectors cannot observe earlier submissions; altering inaccessible secrets cannot alter a seat's observation or action schema.

## 2. Build the free vertical slice

Implement random-legal and simple scripted policies, then run 100 seeded Avalon games locally. Persist protocol/engine versions, complete seat-policy mapping, setup, RNG states/draws, observation hashes, action attempts, resolved events, and outcomes. Keep an authoritative event stream plus explicit recipient visibility; export seat views by filtering, never by sending raw logs to players.

Checkpoint at decision boundaries. A stable run/decision/request ID lets resume reuse saved responses and avoid applying an action twice. Replay uses recorded actions and randomness with pinned engine code; a new API run is not promised to reproduce model outputs.

**Acceptance:** same recorded run produces identical state hashes and outcome; a forced crash/resume matches uninterrupted execution; malformed actions cannot mutate state; every completed run passes the invariants above. Zero network/model calls.

## 3. Add providers, scheduling, and reports

Keep provider adapters behind `Policy.act(observation, action_spec)`. Version context trimming and private memory. Treat player speech as game content; it cannot override rules, visibility, or available actions. No external tools in the first experiment.

One schema-repair attempt is allowed with the same seat-visible context. Exhausted invalid responses stop the game as a recorded policy failure. Transport retries are bounded and recorded; provider errors never become game moves. Preserve known token usage and mark uncertain charges after timeouts.

The scheduler reserves worst-case call cost before dispatch and reconciles actual usage afterward, including probes and retries. Unknown model prices block paid dispatch. Cap both spend and total decisions; reaching a cap produces an incomplete run. Default `max_usd: 0` permits only local policies. A later work-funded configuration must explicitly enable paid providers and set a positive cap.

Belief probes run on a disposable fork of the permitted seat history and never write back to gameplay. Record structured probabilities; do not require private chain-of-thought access. For ordinary good seats, score P(evil) using mean Brier score across eligible targets; exclude mechanically known labels and report by role and phase. Probes have their own cost ledger entries.

Build a static report first. Proposed review layout:

```text
Avalon / protocol v1 / experiment ID / policy definitions
Completed: —   Incomplete: —   Independent scenarios: —   Cost: —

POLICY       RATING [95% INTERVAL]    GOOD W/L    EVIL W/L
<populated only from actual eligible runs>

Matchup matrix | Failure breakdown | Cost | Experiment comparison

Replay: [game] [decision] [public / seat / omniscient]
Visible evidence | Proposed action | Resolution | Optional belief probe
```

Omniscient mode is an analyst-only post-game view and must be visibly labeled. Reports carry no illustrative numbers that could be mistaken for measurements.

**Acceptance:** budget reservation prevents concurrent overspend against the configured accounting cap; failed calls and resumed actions remain auditable; synthetic rating fixtures recover known ordering, side advantage, and near-equal policies. Reordering the same dataset does not change the fitted rating.

## 4. Run a work-funded pilot, then choose sample size

Freeze two policy configurations. Run 12 independent side-swapped scenarios (24 cross-policy games), plus four self-play scenarios per policy (8 games): **32 games planned**. Balance leader positions as closely as counts allow. Inspect failures, cost, evidence visibility, and side bias before expanding. Treat any Elo from this small pilot as exploratory.

Use measured cost and simulated power under plausible effects to choose a fixed next-stage sample size within the approved work budget. Register endpoint, exclusions, allocation, and stopping rules before running it; do not stop because the preferred model is ahead. After the homogeneous baseline, add the one-seat replacement study with independently sampled evaluation scenarios.

**Acceptance:** one manifest regenerates the schedule and report from saved data; no private credentials or provider-specific work paths enter the repository.

## 5. Prove reuse with Dominion

Use the base second-edition First Game kingdom: Cellar, Market, Merchant, Militia, Mine, Moat, Remodel, Smithy, Village, Workshop. Start with a reusable engine audit; implement only this subset if no suitable engine passes. Preserve hidden hands/deck order, optional reactions, nested choices, card conservation, end conditions, and the fewer-turns tiebreaker. [Publisher rules](https://www.riograndegames.com/wp-content/uploads/2016/09/Dominion2nd.pdf).

Swap starting player across scenario pairs; use separate seeded shuffle streams by seat and record actual draws. Matching seeds controls initial randomness but does not force equivalent draws after strategies diverge. Add W/D/L and the draw-aware rating adapter described in [RATINGS.md](RATINGS.md).

**Acceptance:** random/scripted Dominion policies use the same runner, budget ledger, event/replay machinery, and report shell. New game rules and metrics live in adapters. Only generalize the core further when this second implementation demonstrates a concrete need.

# Towards Auto Research

> Status: Baseline scaffold, 2026-08-21 — Zubin dictating over this iteratively.
> Companion doc: [Workback Plan for GA](./workback-plan-for-ga.md).
> Notion canonical: app.notion.com/p/…
> Raw source: [`raw/2026-08-25-halo-docs-dump.md`](./raw/2026-08-25-halo-docs-dump.md) (parts 1–7).
> `[pending Zubin]` awaits the 2026-08-20 Shawn/Tim meeting outcomes.
> `[baseline]` = Opus-drafted from the fact-checked corpus, dictate over freely.

## 0 · The frame

W&B v2: the agentic layer for human–agent research collaboration — the next realm of
W&B where the majority of experiments run through agent interaction. The organizing
analogy: SDLC → software factory for coding agents, therefore AI research lifecycle →
AI Research Factory, with ARIA the first-party researcher.

Two facts anchor everything:

- The agent loop is commoditized. The verified record is not.
- The funnel that matters is not `analysis → tried autoresearch once` (onboarding,
  covered elsewhere) — it is `tried autoresearch once → finds recurring, meaningful
  research value`.

*[Zubin dictation slot — the thesis in your words]*

## 1 · Where we are

- Production is real: thousands of turns/week, high multi-turn share; canonical failure
  traces show the gap is infra and evidence, not intelligence.
- Autoresearch KR is early (~1/10) and its measurement is soft — the funnel's entry
  isn't reliably instrumented yet, let alone its exit.
- Competitive window ~6–12 months: the position (grounded, verified agentic research)
  is contested but untaken.

*[Zubin dictation slot — current-state honesty]*

## 2 · The triad

| | What it is | What it covers |
|---|---|---|
| Science | Research tooling to improve the agent effectively and efficiently | Offline evals, online signals, benchmarks (WBAF), triage flywheel |
| Infra | The agent's infrastructure | Sandboxes, Secret Store, network policy, memory, MCP, launching on infrastructure W&B doesn't own |
| Platform | The Weights & Biases service itself | Weave/Models as custody of evidence, the research object, the full-page agent experience |

(Prior framing mapped: L1 readiness → Platform+Infra, L2 capability → Infra+Science,
L3 interaction → Platform. Directions A/B live inside Science.)

## 3 · Science

### 3.1 Offline evals

State: this lane just got a real path. As of 2026-08-20 Core carries a
service-in-the-loop evaluator — `wbaf service-eval` in
`services/wb_agent/agent_evaluation_tasks/`, with README + AUTHORING merged (#51340).

- **It evaluates the deployment, not a library.** Suites drive the public turn API and
  import nothing under `src/`, so auth, scope resolution, config selection, sandbox
  execution and message persistence happen for an eval turn exactly as for a user.
  A config variant is a two-line YAML alias; re-sent on every continuation turn.
- **The loop is now one command per stage:** edit the config → try it in the local panel →
  run the suite → read `task_pass_rate`. Codified on `zaysola/wb-agent-eval-examples`
  (no PR yet): `docs/agent-development-loop.md`, a 7-task baseline (fundamentals /
  conversation / discipline, each task a copy-me template), and two skills,
  `author-eval-task` and `run-service-eval`.
- **The proof is retrospective, not promised.** The sharding change could not have shipped
  with confidence without offline evals. Worth more than any prospective argument —
  pending the specifics (dictation).
- **Not yet a gate.** Nothing in CI runs these suites. L1 must-pass / L2–L3 hill-climb
  exists as taxonomy, not as mechanism. The ARIA Eval Gate workflow is a different
  thing (WBAF main's sandbox suite against a PR's Core checkout).
- **Fail-open is the disease to design out** (2026-08-20 eval audit): zero conditions ⇒
  pass 1.0, so a typo'd scoring key or an unloaded scorer is silently green;
  stopped/timeout rows skip scoring AND discard the trajectory; one flaky judge call
  zeroes the whole row; token/cost read 0 over this transport.

### 3.2 Online evals & signals

State: the thinnest lane in the triad. Three pieces exist; none is an online-eval system.

- **The prod bridge.** The `wb_agent_prod` Tilt profile runs a local API + worker against
  production Gorilla, the Weave trace server and real Modal sandboxes, local turns
  traced to `wandb/wandb-agent-dev`. This makes offline evals honest — it is not online
  measurement.
- **`online_enrichments/` (WBAF).** Seven registered, versioned enrichers (client platform,
  sandbox security, docs-agent usage, policy signals, reference files, conversation
  stats, user cohort) — deterministic only, idempotent feedback IDs, safe to rerun,
  scheduled trigger deliberately disabled pending validation. It is a fork: ~610 LOC
  copied byte-identical from `factory/triage/` because WBAF isn't installable.
- **Weave publishing.** Scored evals land one evaluation record in `wandb/wbagent-experiments`;
  agent traces land separately in `wandb-agent-dev`.

One boundary already decided: native Weave Agent Signals own turn-level LLM judgment;
`online_enrichments` owns deterministic/derived enrichment and never calls an LLM.

**What is actually missing — the section to be honest in:**

- No named sentinel set, no owner, no review cadence.
- The handshake (offline serves capability and safety; online serves product signal) is
  asserted, not instrumented. The coupling claim — +X offline should show up as +X±ε in
  prod — has no measurement today, and it is the only thing that keeps the two lanes
  from taking each other's jobs.
- Online is structurally confounded (UX changes, user-base drift, rollouts). It is the
  falsifier, not the hill-climb target. Treating it as the target is how we get
  reactive drift.
- The trace→offline-task minting path must stay separate from the monitoring stack.

### 3.3 Benchmarks & the improvement loop

Harness variance is a first-class scientific object, not a nuisance term. Harness-induced
variance has been measured at 7.8× model-induced variance, with a 23.8-point spread
across six harnesses on identical tasks; internally, the same scaffold swung
9.9% → 23.2% between two models with no way to attribute the lift. You cannot improve
what you cannot attribute — the benchmark's job is attribution, not a leaderboard.

- **Unit of measurement.** A benchmark result must pin the full tuple: model version +
  harness version + task-set version + runtime image and tool permissions + verifier
  version + budget and retry policy. Report pass@1 PLUS repeated-trial confidence
  intervals and the full cost curve.
- **Candidate north star:** verified research progress per dollar-hour, reported as a
  vector (recognition · judgment · execution · improvement · validity · continuity ·
  human control · evidence · system value · economics), never as a scalar.
- **Two directions, both mandatory.** Direction A is fundamental science — "what is a good
  agent harness?" — and Direction B is engineering: tool-call robustness, shell calls
  that don't time out, resume from sandbox state. The flywheel dies if we ship only
  one. Corollary: build only what customers asked for and we lose coverage.
- **What blocks the loop from closing.** Replayable scoring does not hold today: rescore
  differs from the original run in scorer set, latency semantics and W&B endpoints, and
  the trajectory has 9 runtime representations and 17 independent re-parsers with no
  schema or version anywhere. Closing it needs row isolation, run-level resume, and a
  typed episode record (turn boundaries, typed tool-call pairs carrying
  `is_error`/`timed_out`, partial trajectories on stop, usage captured at production time).
- **Gating stays social, not mechanical:** every change is a fix (benchmark it, add a
  regression test, move on) or an upgrade (significant benchmarking before ship).
  Science owns the call.
- **Constraints:** build window closes 2026-08-29, Q3 closes 2026-09-30, Zubin OOO
  2026-08-27 → 09-08, capacity over by ~6 person-weeks, WB-34848 still has no DRI.

> **Open questions for Zubin (§3):** the sharding-change specifics (the one retrospective
> proof); the acceptance gate for "good at autoresearch" (stake the quarter on
> verified-progress-per-dollar-hour, or keep the KR proxy?); who owns online evals +
> the first five sentinels; does the coupled offline↔online subset get named this
> quarter; fix WBAF in place vs land the rebuild (not side-of-desk inside the 08-29
> window); symbolic environments vs real customer projects (Direction A is really
> "harness × environment pair").

## 4 · Infra

### 4.1 Execution: sandboxes, network policy, secrets

**Today.** Per-turn Modal sandbox, pre-authenticated with a short-lived user-scoped
delegated W&B token; `/workspace` is the only persistent path; completed turns store a
filesystem snapshot, so any turn is restorable and branchable. Limits are time-based
only — no token or cost budget object (GF3).

Network policy is a 3-branch allowlist: allow-all egress, dev-proxy CIDR, or the prod
domain allowlist (W&B endpoints + docs + GCS signed URLs + user-granted domains).
Everything else is default-deny. `request_network_access` is a TERMINAL tool — the
allowlist is frozen at sandbox start, so widening it ends the turn; grants are
inherited by descendant turns. Consequence: a research turn that discovers mid-flight
that it needs a third-party endpoint cannot continue — it must die and resume. That is
the sharpest execution constraint on launching anywhere W&B doesn't own.

**Secret Store** — today secrets are service-side only: GCP Secret Manager for
Modal/DB/worker credentials, plus automatic redaction of non-allowlisted env values in
sandbox output. There is NO user-scoped credential store the agent can use to
authenticate to a third party (a customer's K8s, Slurm, HF, cloud account, or an MCP
server). W&B Launch is the only sanctioned path to non-W&B compute, and
`skills/research-environment` already warns: "do not infer that a Launch worker can
access code, data, or credentials merely because the sandbox can."

*[needs Zubin dictation]* — scope (user/team/org), custody and audit model, injection
path into sandbox vs Launch job, and the rule that a secret never enters the context
window.

### 4.2 Continuity: memory, durable goals, scheduled work

**Today — state continuity is strong, intent continuity is absent.**

- Project memory is live on master (MySQL, 3 tools): durable per-project facts, names +
  descriptions at turn start with bodies on demand, admission gate on every write
  (2000-char body cap, refuse-don't-truncate), secret scanning, CAS versioning,
  teammate-editable. Scope is project — no org scope yet.
- State rides the turn tree and snapshots: replayable, branchable, and extractable as
  an eval task (the production-to-eval bridge). Compaction is in-place summary
  replacement.
- Intent has no home. Turns are user-initiated only. A long research effort is one ≤1h
  turn, not a resumable objective: no durable goal, no heartbeat, no scheduled or
  system-initiated turn, no typed research state.

**Direction.** GF12 — a durable goal object (objective + budget + gate reference)
re-prompted through the SAME turn path as user input, with heartbeat/cron turns as
system-initiated children. GF13 — a typed research-state object, starting as prose
discipline (`/workspace/RESEARCH.md`) and graduating to something the product renders.
Around those: memory org scope, thread/history search (GF11), lineage compaction (GF9)
so long horizons reset with a summary as a child turn instead of degrading in place.

**Why this is Infra, not UX.** A job on infra we don't own runs on a clock we don't
control. The requirement — launch there and still monitor, repeat, review — is
unimplementable without wake-on-event. Today monitoring an external job only happens
if a human comes back to the thread.

### 4.3 Reach: MCP, external infra, skills

**MCP.** Table stakes: a hosted agent must not cost the user tools they'd have locally.
Master has NO MCP service support. The 7 production skills sit in 3 cost tiers with no
dynamic tool loading, so breadth added today is billed to every turn's prompt.
Reconciliation already made: hosted MCP WITH deferred loading is in scope as table
stakes — the objection is only to eager schema injection (GF1's loader is the same
mechanism). Note the coupling: user-configured MCP servers mean user credentials, so
Secret Store is a dependency of MCP, not a parallel item.

**External infra.** W&B Launch (queue + agent + job artifact) is the current answer;
`skills/research-environment` already encodes the sandbox-vs-Launch fork plus four links
that must be proven independently — code capture, compute compatibility, override
safety, observability. The recurring failure is the unproven link: cluster access and
Launch-agent bootstrap need a human infra owner, and a live queue is not readiness.

**The prerequisite: capture has to beat stdout.** SkyPilot ran ~910 experiments on
CoreWeave Kubernetes with stdout and `sky logs` as the entire tracking layer. If the
agent can launch where W&B holds no record, it cannot monitor, repeat, or review — and
custody is theoretical. Reach and capture ship together.

**Skills as reach.** All 7 are human-authored; GF14 (self-authored skills with provenance
and regression evals) is sequenced last, behind hooks and gates (GF6–GF8).

> **Open questions for Zubin (§4):** Secret Store scope + first cut (BYO credentials for
> external compute, or W&B-only v1?); dynamic egress — break `request_network_access`
> out of the terminal set, or pre-declared per-project research egress profiles?; the
> big cost fork — Launch-only vs direct adapters (K8s/Slurm/SSH), everything in 4.3
> sizes off this; where does capture-beats-stdout live (Infra: agent auto-instruments
> its jobs, or Platform: zero-friction logging) — needs one named owner; GF12 durable
> goals — inside Q3's build window or explicitly post-GA?; personas/usage (branch-only
> today) — Infra policy carrier or Platform surface?

## 5 · Platform

### 5.1 Custody of evidence — the wedge

Every shipped completion-verification mechanism in 2026 judges the agent's own
transcript. Anthropic's docs, verbatim: "The evaluator… does not call tools, so it can
only judge what Claude has already surfaced in the conversation." That is structural
for anything inside a harness, and prose-reading judges don't work: SPOT's best is
21.1% recall / 6.1% precision, and 59% of MLReplicate's automated acceptances contained
fabricated or unsupported claims.

W&B holds the other side of the question: run metrics, eval scores, config lineage,
environment fingerprints, artifact hashes.

The primitive that falls out:

```
claim / card defaults to FAIL / UNVERIFIED
  -> agent proposes and executes work
  -> W&B observes the real run, environment, metrics, artifacts
  -> independent checks bind the result to the intended code/data/config
  -> repeats / holdouts / robustness checks clear
  -> VERIFIED at a declared tier -> only then COMPLETE
```

**Two non-negotiables.** The evaluator runs against telemetry, not the chat log. And
NO RESULT, INFEASIBLE, BLOCKED BY INFRA, BUDGET EXHAUSTED are valid terminal states —
forcing every branch to end in success manufactures the completion pressure under
which agents fabricate (20.6% → 3.2% when it is removed). The gate binds to a bundle,
not a number: commit, diff, config, raw logs, metric definitions, checkpoint and
dataset-split hashes, environment image, seed, hardware, job lineage, cost, retries,
verifier version.

**Why this and not "verification":** verification is contested by at least ten parties;
"the only evaluator that can see the telemetry" is contested by none — it is a claim
about custody, not agent intelligence, which is why Codex/Claude + MCP + Skills
structurally cannot replicate it. EviBound proves the mechanism (hallucination
100% → 0% when claims are gated on queryable run IDs) and that the position is open:
it was prototyped on MLflow.

Hard dependency: capture has to beat stdout (see §4.3).

### 5.2 The research object

Run grouping is not enough. The gate in 5.1 has to bind to something, and today there
is no object that holds hypotheses, decisions, dead ends, and promotion state. Two
positions, both live:

**Position A — the research graph is the object.** An event-sourced graph: hypotheses ↔
proposals ↔ code/config/env versions ↔ Runs ↔ artifacts ↔ decisions ↔ claims ↔ human
interventions. Append-only at the event layer, versioned at the state layer, forkable,
failed experiments preserved as first-class evidence. A Run Collection is a view over
this object, not the object itself. This is the platform expression of the resolved
spec-over-chat fork: "the research specification and evidence graph are canonical;
chat is a control and explanation surface over them."

**Position B — extend the grouping primitive we already ship.** `wandb.Study` /
Run Collection becomes the object rather than a view: less new surface area, lands
inside the Models data model users already have, reuses shipped UI. The cost: claims,
decisions, dead ends, and promotion state get bolted onto a view, and the promotion
gate has no durable place to bind.

**Ruling:** *[pending Zubin]* — outcome of the 2026-08-20 Shawn/Tim meetings; the workback
slots this as resolved before 2026-08-26.

Independent of the ruling: the state model must not be coupled to Kanban board columns.
The hypothesis board is one steering view over the object, an explicitly-argued bet,
not the headline.

### 5.3 The agent surface

Where the object gets steered from. Three states, because the ask differs per state:

- **Today (master):** a chat drawer/window mounted on W&B pages. No full-page `/agent`
  route. The system prompt, per-turn context, variant, and skill set are invisible to
  users; project memory is the one shipped exception.
- **Built, not shipped:** the full-page experience and the personas platform exist
  end-to-end on branch `zubin-ui-dreams-agent-page` — console, per-turn resolver,
  versioned personas, usage view. Landing it is a scheduling and migration question,
  not a design question.
- **North star (Tim's ArDE):** "Evolve beyond the chat-based experience into a
  fully-fledged development environment, focused on agentic ML research. Make W&B the
  de facto surface where ML researchers spend their time (and tokens)."

The rule that keeps the surface honest: it is a view over the research object, not a
second source of truth. Chat stays a control surface; the spec and the evidence graph
stay canonical. And surface work sequences AFTER custody — a hypothesis board built
before the promotion gate is a completion-pressure machine by construction.

Depth on the full-page experience — levers, scope cascade, console tabs — is a
separate one-pager (exists at `~/wandb/labs/aria-2/05-full-page-experience.md`); this
section only fixes its place in the stack.

> **Open questions for Zubin (§5):** the research-object ruling (graph-as-object vs extend
> Study/Run Collection — who owns the schema, does it land before 2026-08-26?); where
> the promotion gate lives and who provably cannot touch it (service-side verifier vs
> agent-runtime hook — the optimizing agent must be unable to modify verifier or
> promotion suite); scope of default-FAIL at GA (every claim, or only claims inside a
> declared research contract — and what does ARIA say when a claim is true but
> unpromotable?); zero-friction capture — commit to run binding for jobs launched
> outside Launch by 2026-09-30, or scope custody to jobs we launch?; claim typing in
> v0 or later; surface timing — full-page branch→master in Q3 or a 26H2 ArDE item,
> personas with it or separately?

## 6 · What we are not doing

*[Zubin dictation slot — cut lines]*

## 7 · Open decisions

- Research object ruling (research graph vs Run Collection) — *[pending Zubin: 2026-08-20 meeting outcomes]*
- WB-34848 owner + compute — *[pending]*
- GPU capacity — *[pending]*
- DRI — *[pending]*

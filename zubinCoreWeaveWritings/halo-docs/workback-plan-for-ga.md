# Workback Plan for GA

> Status: Baseline scaffold, 2026-08-21 — Zubin dictating over this iteratively.
> The science tight-alignment doc: what must be true, by when, owned by whom, for GA.
> Companion doc: [Towards Auto Research](./towards-auto-research.md).
> Notion canonical: app.notion.com/p/…
> Raw source: [`raw/2026-08-25-halo-docs-dump.md`](./raw/2026-08-25-halo-docs-dump.md) (parts 8–10).

## 1 · What GA means

*[Zubin dictation slot — the GA bar: which capabilities, which metrics, which gates]*

## 2 · Current state

- **Landed:** offline-evals path (service-eval workflow, task authoring docs, 7-task
  baseline), core↔WBAF handshake, local-dev-against-prod bridge.
- **Soft:** online evals (thinnest lane), funnel instrumentation, autoresearch KR
  measurement.

GA is undefined — no GA bar, no GA date exists anywhere in the corpus. Defining it is
row #1 of the workback, not an input to it.

- ARIA public preview since 2026-06-29; 3,069 turns / 1,095 conversations in
  2026-08-05→08-12. Blog series (WB-36938) posted 2026-08-13.
- KR ARIA-2 (10 users launching training jobs) ~1/10 AND not reliably measurable —
  WB-38136 is an unreliable LLM judge.
- Capacity: ~34 person-weeks of bets vs ~28 available; no cut list exists.
- Zubin OOO 2026-08-27→09-08, last full working day 2026-08-26 (Tim covers). Stefano
  European hours 09-05→09-18, PTO 09-17→09-25.

| Lane | Landed | Open |
|---|---|---|
| Online evals | Nightly pipeline merged 2026-07-21 (WBAF #763); prod bridge, daily-triage | Thinnest — needs definition. No signal set, no funnel instrumentation (WB-38136), sentinels unspecified |
| Offline evals | Baseline shipped; core↔WBAF handshake landed 2026-08-21 (#965, #51001/#50999/#50997/#51295) | Baseline branch `zaysola/wb-agent-eval-examples` unPR'd; audit defects (fail-open scoring, no trajectory schema, not installable, empty reference registry); WBAF rebuild plan uncommitted |
| Harness | max_steps→100 merged 2026-07-28 (#49097) | #49099 needs a reviewer; `apply_patch`→`write_file`/`edit` for GLM; durable wake/resume (WB-34983, archived); autoresearch skill not autopatched (WB-38237) |
| Docs | core #51340 + WBAF #1020 merged | WB-38429 unwritten; ARIA-packaged context docs unmapped (Forge churn risk) |

## 3 · Workback (candidate rows)

| Window | Lane | Row | Owner | Done when |
|---|---|---|---|---|
| 2026-08-21 → 08-26 (4 working days) | all | Define the GA bar — plus a GA date or an explicit deferral | [pending Zubin] | Numeric or binary gate per lane |
| | all | Lane owners named; WB-34848 gets a DRI + compute ask | [pending Zubin] | In Jira, not in a doc |
| | all | Cut list closing the ~6 pw overage | Zubin | Published |
| | online | Definition spike: signals, funnel stages, what replaces the WB-38136 judge | [owner?] | Spec Tim can review during OOO |
| | offline | PR the 7-task baseline branch + README | Zubin | PR open |
| | harness | Land or park in-flight PRs (#49099 gets a reviewer) | Anish / [owner?] | Merged or parked |
| | docs | WB-38429 lands; research-object ruling written down | Zubin [pending] | Ruling written down |
| 2026-08-27 → 09-08 (Zubin OOO) | all | No new direction — execute the cut list | Tim | Nothing re-scoped |
| | offline | WBAF rebuild plan committed with Stefano before his 09-17 PTO | Stefano | Plan merged |
| 2026-09-09 → 09-30 (Q3 close) | harness | Table stakes: recognition intrinsic (WB-38237), hosted parity (MCP, memory), durable wake/resume revived (WB-34983) | Anish + [owner?] | Plan-first fires with no CTA |
| | online | Nightly eval taxonomy live; funnel instrumented end to end | Ashraf / Julia | Running nightly |
| | offline | Zero-friction capture spike — instrumentation easier than stdout | [owner?] | Spike written up |
| 2026-10-01 → 12-31 (Q4 wedge) | all | Research graph + evidence-gated completion (default-FAIL, telemetry evaluator, valid NO RESULT / INFEASIBLE / BUDGET EXHAUSTED states) | [owner?] | VERIFIED needs a W&B-custodied artifact |
| | offline | WB-34848 ablation — how much better W&B makes autoresearch; claim-benchmark v0 | [pending Zubin] | Matched with/without number |
| 2026-12-31 | — | Skills + MCP + ARIA on one backbone; nothing infra-shaped left | — | — |
| 2027 | — | ArDE surface: hypothesis board as one steering view over the graph, cloud IDE, serverless RL, CoreWeave public cluster | — | After custody |

**Public-claim ordering (hard):** "W&B grounds and verifies agentic research" ships BEFORE
"ARIA is the best autoresearch agent." The second waits on WB-34848. Ablate, don't argue.

## 4 · Lanes & owners

- **Online evals** — Ashraf, Julia (execution); definition [owner?]
- **Offline evals** — Stefano + Zubin
- **Harness** — Anish; [owner?]
- **Docs** — Zubin

## 5 · Capacity & cut line

Committed work exceeds capacity (~34 vs ~28 person-weeks this quarter). Table stakes
must be a named list with an explicit cut line, not a vibe.

*[Zubin dictation slot — the cut]*

## 6 · Risks

1. **GPU capacity unresourced (OQ-Q3-01)** — blocks simulation at scale and autoresearch
   mid-quarter; a plan that assumes simulation-at-scale without naming this is fiction.
   *Ask:* the public spot/off-peak cluster already in the OKR doc. [pending Zubin]
2. **Capacity over-committed** — ~34 pw vs ~28; the binding frontend constraint is real;
   Zubin OOO and Stefano PTO both land inside the build window. *Ask:* cut list by
   2026-08-26.
3. **Split accountability** — both ARIA KRs DRI'd to Julia, Tim/Zubin own the ArDE
   workstream, WB-34848 unowned since 2026-04. *Ask:* one end-to-end DRI. [pending Zubin]
4. **Positioning split** — execs respond to "auto-research," practitioners recoil; preview
   customers chose Claude Code because IT SEES THEIR CODE. *Ask:* lead
   harness/evidence/control, autoresearch as the exec payoff; mountable repos +
   ARIA↔GitHub.
5. **CoreWeave ARENA collision** — overlaps the execution layer, one letter from ARIA.
   *Ask:* coordinate before public positioning. [owner?]
6. **Online evals is a definition problem** — the one lane where nobody can state what
   "working" means; it will silently absorb the OOO window. *Ask:* spike by 2026-08-26
   or cut it from GA.

## 7 · Open decisions

- DRI, WB-34848 owner+compute, GPU capacity, research-object ruling — *[pending Zubin: 2026-08-20 Shawn/Tim meeting outcomes]*

> **Open questions for Zubin (workback):** the GA bar and date (per-lane gates or one
> composite?); did 2026-08-20 land you as DRI end-to-end or only the autoresearch
> program?; WB-34848 owner + compute + was Morgan's reported 2026-04-27 execution
> ever located?; who defines online evals while you're out 08-27→09-08 — Julia,
> Ashraf, or Tim?; what comes off the list to close the ~6 pw overage (name cuts,
> not priorities); does offline evals survive Stefano's 09-17→09-25 PTO without you,
> or must the rebuild plan land before 2026-09-05?

# What the Elo number means

## What Dwarkesh actually computes

The inspected code fits a **regularized Bradley–Terry model**, then converts the fitted strengths into Elo points. It does not update a running Elo after each match. One observation is one game's good/evil outcome; whole alignments are populated by copies of an entrant policy. [Pinned rating implementation](https://github.com/dwarkeshsp/clocktower/blob/9fa9773bdcab793d972c6a42e614d3f8f32f12db/clocktower/ratings.py).

In that implementation:

- Good's win probability is `sigmoid(r_good - r_evil + h + setup)`.
- `h` estimates good's advantage. Setup features cover Minion identities and 11/12-player tables, relative to the reference setup.
- A combined fit estimates one strength per model; a separate fit estimates good-side and evil-side strengths.
- Gaussian penalties stabilize sparse/perfect records: strength prior SD 1.5 logits, setup SD 3, approximately flat side intercept.
- Multiply logit strength by `400 / ln(10)` and center the entrant mean at 1500.
- Report 90% intervals from 500 bootstrap resamples, keeping side-swapped games together.
- The loader selects particular v16/v17 batches and excludes a documented retry-bug cohort. This is not a fit to every historical game in the repository.

The checked output contains 386 games and estimates a +112 Elo good-side advantage at the reference setup. These are upstream results, not reproduced results from this spike. [Published fit data](https://github.com/dwarkeshsp/clocktower/blob/9fa9773bdcab793d972c6a42e614d3f8f32f12db/docs/plans/f2-ratings.json).

## Reading a score without overinterpreting it

For a binary, neutral matchup:

```text
P(A wins) = 1 / (1 + 10^(-(R_A - R_B) / 400))
```

| Rating advantage | Neutral win probability |
| --- | ---: |
| 0 | 50.0% |
| 100 | 64.0% |
| 200 | 76.0% |
| 400 | 90.9% |

1500 is an arbitrary reference, not a competency threshold. Absolute ratings from different games or separately centered pools are not comparable.

**Worked example from the published parameters:** Astra's 1859 minus Opus 4.6's 1432 gives a 427-point gap. Neutral conversion gives 92.1%. Including the fitted +112 side advantage, at the reference setup the combined model predicts Astra wins 95.7% when good and 86.0% when evil, averaging 90.8% across equal sides. These are calculations from rounded fit parameters, not observed head-to-head frequencies. [Parameter source](https://github.com/dwarkeshsp/clocktower/blob/9fa9773bdcab793d972c6a42e614d3f8f32f12db/docs/plans/f2-ratings.json).

This distinction matters: side-swapping reduces assignment confounding but does not make a neutral Elo conversion equal the side-averaged win rate.

## Our proposed rating contract

**Entrant:** immutable policy ID = provider/model snapshot + prompt hash + sampling/reasoning settings + memory/context policy + tool access + token limits. A mixed roster gets its own composition ID. Several copies of the same policy have separate private histories.

**Pool:** one frozen game/rules/communication protocol and budget regime. Keep Avalon and Dominion separate. Changing these conditions creates a new pool; longitudinal comparisons require unchanged reference entrants and overlapping matchups.

**Avalon estimator:** penalized Bradley–Terry with a good-side intercept. Fix the role set and player count initially, so no setup covariates are needed. Use a declared Gaussian strength prior of SD 1.5 logits, a weak SD 3 prior for the side intercept, and a sum-to-zero strength constraint. Freeze the pilot entrant mean at 1500 for reporting. Refit all eligible games together; publish sensitivity to strength prior SD 1 and 3.

**Design:** draw independent role/seat scenarios; play each twice with policy-to-alignment assignments reversed. Store scenario ID and pair ID explicitly. Rotate starting leader and role placements across scenarios. Keep a connected matchup graph; include same-policy games to check side advantage. Each game contributes one outcome, not five independent player results.

**Uncertainty:** bootstrap whole scenario blocks, including both swaps and all repetitions/interventions sharing the setup. Use 2,000 resamples and 95% intervals for the new harness. Report rating-gap intervals and predicted matchup intervals, rather than inferring significance from overlapping marginal intervals. With few independent blocks, label all estimates exploratory; flag disconnected or effectively unidentified fits instead of issuing a confident ranking.

**Diagnostics:** always show raw wins by side, matchup counts, independent scenario count, completion rate, invalid-action rate, and cost per finished game. Inspect held-out matchup predictions; cyclic matchups or strong side-specific differences can make a single scalar misleading. Only add separate good/evil ratings when the schedule supports both.

**Failures:** only complete paired scenarios enter the primary paired analysis. Log all excluded games and reasons by policy. If failures differ by policy, mark the capability ranking potentially biased and show a sensitivity analysis treating policy failures as losses. Infrastructure interruptions remain missing outcomes, never silent draws or losses. Report operational reliability separately.

## Team skill is not individual credit

A homogeneous-team rating answers how a policy performs when filling an alignment. It cannot identify which player caused a win. Mixed-team strength also need not equal the average of member ratings.

For one-seat contribution, compare an unchanged baseline roster against a roster with exactly one policy replaced, on the same scenario blocks and opponents. Rotate roles, report uplift separately for Merlin, ordinary good seats, and evil roles, and bootstrap the full intervention block. This estimates the effect of a replacement under the tested roster distribution; call it **win-rate uplift**, not individual Elo.

For Dominion, begin with W/D/L and expected score (`win + 0.5 × draw`). The later rating adapter should use a Davidson extension of Bradley–Terry with a tie parameter and first-player advantage. Do not relabel draws as losses or transplant Avalon's binary likelihood unchanged.

## Questions to settle with data

1. Does one rating predict held-out opponents well, or do we need a matchup matrix and side-specific ratings?
2. How much do ranks move with priors, scenario resampling, and extra games?
3. Do equal-token and equal-dollar budgets produce different conclusions? Keep those experiments separate.
4. Does a homogeneous-team advantage survive mixed rosters and a direct test of teammate recognition from writing style?
5. How much uncertainty remains after a realistic work-funded budget? A pilot estimates cost and variance; it does not guarantee a publishable ranking.

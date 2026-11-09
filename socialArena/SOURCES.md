# Source notes

Checked 2026-11-08. Recommendations in this folder are our proposed design unless explicitly attributed below.

| Source | Used for / limits |
| --- | --- |
| [Dwarkesh's essay, Nov 5, 2026](https://www.dwarkesh.com/p/clocktower) | Read the video companion/script for framing and questions. Direct YouTube retrieval failed; no caption extraction or verbatim transcript is claimed or bundled. |
| [Video: 1NQwfImTseU](https://www.youtube.com/watch?v=1NQwfImTseU) | Identified companion video. Use the essay for the text actually consulted. |
| [Clocktower repository](https://github.com/dwarkeshsp/clocktower/tree/9fa9773bdcab793d972c6a42e614d3f8f32f12db) | Inspected README, protocol, experiment questions, rating code, and published rating JSON. Snapshot: `9fa9773bdcab793d972c6a42e614d3f8f32f12db`. |
| [Rating implementation](https://github.com/dwarkeshsp/clocktower/blob/9fa9773bdcab793d972c6a42e614d3f8f32f12db/clocktower/ratings.py) | Source of estimator, priors, Elo conversion, exclusions, and bootstrap mechanics. |
| [Published rating output](https://github.com/dwarkeshsp/clocktower/blob/9fa9773bdcab793d972c6a42e614d3f8f32f12db/docs/plans/f2-ratings.json) | 386 games; worked example parameters. We did not download the full dataset or rerun the upstream fit. |
| [Frozen Clocktower protocol](https://github.com/dwarkeshsp/clocktower/blob/9fa9773bdcab793d972c6a42e614d3f8f32f12db/docs/PROTOCOL.md) | Seat isolation, communication constraints, paired assignments, detached probes, versioning. |
| [Upstream research questions](https://github.com/dwarkeshsp/clocktower/blob/9fa9773bdcab793d972c6a42e614d3f8f32f12db/docs/experiments.md) | Memory, single-seat upgrades, mixed teams, style recognition. Shared team memory can reveal alignment; keep any future learning private to independently assigned agents. |
| [Original Avalon rulebook, hosted mirror](https://avalon.fun/pdfs/rules.pdf) | Baseline game rules. Implementation must pin the edition; this plan excludes optional roles and expansions. |
| [Dominion publisher rulebook](https://www.riograndegames.com/wp-content/uploads/2016/09/Dominion2nd.pdf) | Second-edition setup, First Game kingdom, phases, reactions, ending and scoring. |

The benchmark website did not load through the research tool. The linked upstream repository supplied the methodology. We inspected its rating source directly; its documentation sometimes describes the split fit more prominently than the combined headline fit, so this plan follows the code.

No upstream source, game assets, transcripts, or API credentials are vendored. Before reusing an engine, record its license, pinned revision, supported rules, and missing mechanics. Existing framework/engine selection remains an implementation task, not a completed survey.

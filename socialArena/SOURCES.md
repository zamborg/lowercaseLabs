# Source notes

Implementation verification: the Responses API can return commentary and a final answer as separate messages. The provider adapter selects the completed `final_answer` instead of concatenating phases; a single unphased message is supported for legacy responses. Commentary is retained only in analyst records, never used as an action. This follows the [official phase documentation](https://developers.openai.com/api/docs/guides/reasoning), checked 2026-11-09, and was verified against an actual Luna response.

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


## Implementation reuse audit (2026-11-08)

A bounded audit inspected [OpenSpiel](https://github.com/google-deepmind/open_spiel) (Apache-2.0) and its [available game list](https://github.com/google-deepmind/open_spiel/blob/master/docs/games.md). The inspected list did not contain Avalon or Dominion. Its C++ dependency and absent required adapters did not improve this small Python vertical slice. No upstream engine/code is reused or vendored, so there is no reused revision to pin. This is a bounded candidate check, not an exhaustive engine survey. Both adapters are original implementations of the documented fixed rules scope, with no game assets.

OpenAI integration follows the official [Responses create reference](https://developers.openai.com/api/reference/resources/responses/methods/create), [structured output guide](https://developers.openai.com/api/docs/guides/structured-outputs), and [Luna model/pricing page](https://developers.openai.com/api/docs/models/gpt-6-luna). Requests explicitly choose standard service, JSON mode and bounded output; engine validation remains authoritative. Price values are supplied explicitly in immutable policy configs with provenance. No key is recorded.

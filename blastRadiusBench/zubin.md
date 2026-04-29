# blastRadiusBench Research Memo

Date: 2026-04-26

## Session Status

I used the existing code, docs, local fixtures, and the already-supported public Terminal-Bench rollout corpus rather than running new Harbor jobs. No new benchmark traces were generated in this session.

Concrete repo changes made:

- Added public-trace dynamic process metrics to `src/blast_radius_bench/public_terminalbench.py`.
- Added command-intent inference for public tool calls: `inspect`, `search`, `read`, `edit`, `test`, `build`, `dependency`, `run`, `plan`, and `other`.
- Added `agent_contrast_summary.csv`, which surfaces agent pairs with similar success but different process signatures.
- Added task-adjusted success lift to `agent_summary.csv` as a first-pass control for task mix.
- Added `process_correlation_summary.csv`, `agent_behavior_signature.csv`, `intent_summary.csv`, and `tool_intent_summary.csv`.
- Added `task_agent_contrast_summary.csv`, which finds within-task agent pairs with similar success and divergent process signatures.
- Added first-pass failure/recovery features: failure rate, first failure step, post-failure tool calls, and post-failure intent entropy.
- Added same-model/different-harness and same-harness/different-model contrast CSVs.
- Added `agent_task_centered_process_summary.csv` as a lightweight fixed-effect style residual analysis.
- Added intent-mix and intent-entropy plots to the static HTML report.
- Regenerated `reports/terminalbench-public/yoonholee-terminalbench-trajectories/` from `yoonholee/terminalbench-trajectories`.
- Added tests for the new parser behavior in `tests/test_public_terminalbench.py`.
- Fixed a README code-fence issue in the quickstart.

Verification:

```bash
uv run pytest
```

Result: 16 tests passed.

## Research Claim

The strongest version of the thesis is not just that agents differ in success rate. It is that each agent system, defined as a model plus harness tuple, has a measurable behavioral signature. That signature shows up in how it localizes the problem, expands context, repeats actions, switches tool categories, edits, verifies, and terminates.

Traditional benchmarks observe the final task outcome. `blastRadiusBench` should observe the path. The path is where many meaningful differences live.

## Core Vocabulary

I would separate the project into two layers.

### Ontological Pattern

The stable structure of an agent system before it touches a task:

- model family and capability profile
- harness action grammar
- tool surface
- editing mechanism
- search and read affordances
- planning or scratchpad affordances
- verification loop affordances
- termination policy
- trace observability

This is the "kind of thing" the agent is in the benchmark world.

### Dynamic Pattern

The temporal behavior produced when that tuple meets a task:

- first localization action
- breadth and depth of context expansion
- transition rhythm across shell, read, edit, plan, and other tools
- repetition and looping
- latency to first edit
- verification intensity
- recovery after failed commands or failed tests
- final convergence or abandonment

This is how the agent moves through the world.

## Refined Research Questions

### RQ1. What does offline trace data reveal about model-harness tuples that final benchmark scores hide?

Hypothesis: Agents with similar task success rates will have measurably different trajectory signatures. These signatures will be partially stable across tasks and will often be more attributable to the harness than to the model alone.

Testable signals:

- mean tool calls per run
- tool category mix
- time or step index to first edit
- pre-edit exploration count
- repeated command rate
- category switch rate
- context bloat and forbidden reads when gold annotations exist

### RQ2. Are outcome benchmarks sufficiently resolving for distinctions between coding agents?

Hypothesis: Outcome benchmarks are not sufficiently resolving. They compress heterogeneous behavioral policies into a single success scalar and therefore miss cost, risk, trust, and ergonomics differences.

This is already visible in public Terminal-Bench traces: among traced agents with at least hundreds of runs, mean tool calls range from about 6 to about 69 while success rates do not increase monotonically with tool use.

### RQ3. Which parts of an agent signature are harness effects versus model effects?

Hypothesis: Tool category mix, edit latency, command repetition, and trace observability are primarily harness-shaped. Within a harness, success rate and recovery quality vary more with model capability.

Study design:

- compare same model across harnesses when possible
- compare different models inside one harness
- use task fixed effects so task mix does not dominate the result

### RQ4. When is broad exploration useful, and when is it context bloat?

Hypothesis: Broad exploration helps on retrieval-required and dependency-frontier tasks only when it is targeted. On narrow tasks, high exploration predicts cost and failure more often than success.

Required next data:

- Harbor-native gold context labels
- task category labels
- file-level read/search traces
- verifier outcomes

### RQ5. Can trace shape predict failure mode before final failure?

Hypothesis: Failure runs will often show one or more early process markers: delayed localization, high repeated command rate, long same-category loops, broad shell search without later targeted reads, or verification loops with no narrowing action.

This can become a diagnostic layer rather than only a descriptive benchmark.

## What The Public Terminal-Bench Data Says So Far

The regenerated report covers:

- 52,104 runs
- 26 agents
- 49 models
- 89 tasks
- 39.6% overall success rate
- 66.1% runs with embedded steps

Important limitation: this public dataset is enough for process analysis, but not enough for full blast-radius analysis. It does not consistently expose normalized file reads with task-level gold context. So it can answer "how did the agent move?" better than "was every read justified?"

Selected traced-agent process signatures:

| Agent | Runs | Success | Trace Coverage | Mean Tool Calls | Shell Share | Read Share | Edit Share | Switch Rate | Repeat Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| terminus-2 | 17,431 | 0.336 | 0.915 | 40.2 | 0.961 | 0.000 | 0.000 | 0.091 | 0.236 |
| mini-swe-agent | 6,663 | 0.228 | 0.975 | 20.2 | 1.000 | 0.000 | 0.000 | 0.000 | 0.076 |
| openhands | 6,198 | 0.282 | 0.977 | 69.1 | 0.635 | 0.000 | 0.239 | 0.227 | 0.605 |
| codex | 3,532 | 0.452 | 0.399 | 6.0 | 0.947 | 0.000 | 0.000 | 0.145 | 0.080 |
| claude-code | 3,092 | 0.403 | 0.816 | 27.9 | 0.576 | 0.110 | 0.169 | 0.522 | 0.271 |
| gemini-cli | 1,766 | 0.335 | 0.446 | 7.3 | 0.557 | 0.125 | 0.253 | 0.550 | 0.259 |
| terminus-3-3 | 887 | 0.749 | 0.764 | 34.1 | 0.917 | 0.025 | 0.000 | 0.105 | 0.243 |
| judy | 445 | 0.719 | 0.353 | 15.8 | 0.732 | 0.095 | 0.082 | 0.370 | 0.753 |
| deepagent-harbor | 433 | 0.677 | 0.963 | 26.4 | 0.549 | 0.283 | 0.086 | 0.398 | 0.204 |

Interpretation:

- There are clear harness signatures. Some agents are almost pure shell loops. Others use explicit read/edit tools and switch categories frequently.
- More tool calls are not automatically better. In traced runs, run-level tool-call count has a weak negative correlation with success.
- Higher tool-category entropy has a weak positive relationship with success in traced runs, but this is not causal yet. It may mean that richer tool surfaces help, or simply that stronger agents expose richer traces.
- Trace coverage itself is uneven and must be treated as a measurement variable. Some high-performing agents have no embedded steps in this public corpus, so their process cannot be compared fairly from this data alone.

Run-level correlations on traced rows:

| Feature | Correlation With Success |
|---|---:|
| tool_call_count | -0.110 |
| step_count | -0.081 |
| category_switch_rate | 0.064 |
| repeated_tool_command_rate | -0.055 |
| tool_category_entropy | 0.149 |
| duration_seconds | -0.197 |
| cost_cents | -0.094 |
| input_tokens | -0.087 |

These should be treated as descriptive statistics, not causal claims. Task difficulty and harness trace coverage are confounders.

After adding command-intent parsing, the corpus-level action mix across parsed tool calls is:

| Intent | Count | Share |
|---|---:|---:|
| other | 492,521 | 0.344 |
| edit | 204,130 | 0.142 |
| run | 190,330 | 0.133 |
| read | 181,661 | 0.127 |
| inspect | 105,335 | 0.074 |
| dependency | 65,557 | 0.046 |
| search | 57,902 | 0.040 |
| finish | 42,026 | 0.029 |
| plan | 37,888 | 0.026 |
| build | 34,054 | 0.024 |
| control | 17,710 | 0.012 |
| test | 4,048 | 0.003 |

Two caveats matter:

- `other` is still too large, but the first parser tightening reduced it from about 48.1% to 34.4% of parsed calls by handling `cd ... && ...`, completion tools, interrupts, IPython execution, downloads, package installs, and arbitrary executable runs.
- `test` is surprisingly tiny, probably because many benchmark verifications are external to the agent trace or hidden behind commands not yet recognized as tests.

Task-centered correlations are more useful than raw correlations because they subtract much of the task difficulty baseline. On traced rows:

| Feature | Raw Success Corr | Task-Centered Success Corr |
|---|---:|---:|
| command_intent_entropy | 0.142 | 0.198 |
| post_failure_intent_entropy | 0.085 | 0.142 |
| tool_category_entropy | 0.149 | 0.135 |
| category_switch_rate | 0.064 | 0.062 |
| intent_switch_rate | -0.001 | 0.037 |
| repeated_tool_command_rate | -0.055 | -0.037 |
| failure_rate | -0.082 | -0.091 |
| tool_call_count | -0.110 | -0.059 |
| step_count | -0.081 | -0.061 |
| duration_seconds | -0.197 | -0.094 |

Interpretation: diversity of action types is a better positive process signal than sheer amount of action. More steps, more cost, and more duration are weakly negative after task centering. This supports a sharper hypothesis: useful agents are not merely broad; they are dynamically varied without becoming repetitive or slow.

The first failure/recovery pass adds another useful distinction: failure rate itself is negative, but post-failure intent entropy is positive after task centering. That suggests a more nuanced recovery hypothesis:

> Failing is bad; diverse recovery after failure is good.

Agent-level failure/recovery signatures among traced agents:

| Agent | Success | Failure Rate | Failed Tool Calls | Post-Failure Calls | Post-Failure Entropy |
|---|---:|---:|---:|---:|---:|
| gemini-cli | 0.335 | 0.339 | 2.8 | 5.5 | 1.413 |
| mini-swe-agent | 0.228 | 0.136 | 2.9 | 13.6 | 1.478 |
| codex | 0.452 | 0.115 | 0.7 | 4.1 | 1.377 |
| deepagent-harbor | 0.677 | 0.094 | 2.4 | 18.7 | 1.633 |
| terminus-2 | 0.336 | 0.087 | 3.3 | 19.6 | 1.862 |
| claude-code | 0.403 | 0.083 | 2.6 | 17.5 | 1.875 |
| openhands | 0.282 | 0.061 | 4.7 | 52.2 | 1.875 |
| terminus-3-3 | 0.749 | 0.055 | 2.0 | 19.8 | 1.995 |
| judy | 0.719 | 0.054 | 0.9 | 7.5 | 0.939 |

The current behavior-signature labels are crude but already separate recognizable movement styles:

| Agent | Signature | Success | Task-Adjusted Lift | Mean Tool Calls | Intent Entropy |
|---|---|---:|---:|---:|---:|
| terminus-3-3 | mixed | 0.749 | 0.352 | 34.1 | 1.676 |
| judy | compact | 0.719 | 0.323 | 15.8 | 1.090 |
| deepagent-harbor | multi-tool | 0.677 | 0.272 | 26.4 | 1.652 |
| codex | compact | 0.452 | 0.056 | 6.0 | 1.390 |
| claude-code | repetitive | 0.403 | 0.007 | 27.9 | 1.790 |
| gemini-cli | compact | 0.335 | -0.061 | 7.3 | 1.352 |
| terminus-2 | shell-linear | 0.336 | -0.061 | 40.2 | 1.598 |
| openhands | repetitive | 0.282 | -0.114 | 69.1 | 1.629 |
| mini-swe-agent | shell-linear | 0.228 | -0.168 | 20.2 | 1.108 |

This is still descriptive, but it gets closer to the model-harness tuple ontology. `codex` and `gemini-cli` are both compact in this public corpus, but their edit-intent shares differ substantially. `terminus-2` and `mini-swe-agent` are shell-linear, but with different volume and entropy. `claude-code` and `openhands` look repetitive under the current rule, but one has richer explicit read/edit affordances while the other is much higher volume.

The new contrast report directly supports the "same outcome, different movement" claim. Top examples from `agent_contrast_summary.csv`:

| Agent A | Agent B | Success Gap | Tool Call Gap | Switch Gap | Repeat Gap |
|---|---|---:|---:|---:|---:|
| terminus-2 | gemini-cli | 0.001 | 33.0 | 0.459 | 0.023 |
| terminus-3-3 | judy | 0.030 | 18.3 | 0.266 | 0.510 |
| codex | claude-code | 0.049 | 21.9 | 0.377 | 0.191 |
| judy | deepagent-harbor | 0.042 | 10.6 | 0.028 | 0.549 |

This is a useful paper figure candidate: hold outcome approximately fixed, then show that the process vector still moves substantially.

The within-task contrast report is stronger because it controls task identity directly. It produced 1,104 same-task, similar-outcome, different-process pairs. Top examples:

| Task | Agent A | Agent B | Success Gap | Tool Call Gap | Process Gap |
|---|---|---|---:|---:|---:|
| schemelike-metacircular-eval | gemini-cli | openhands | 0.100 | 159.9 | 183.7 |
| winning-avg-corewars | gemini-cli | terminus-2 | 0.050 | 134.8 | 163.0 |
| make-doom-for-mips | codex | openhands | 0.000 | 137.4 | 160.8 |
| make-doom-for-mips | deepagent-harbor | openhands | 0.000 | 149.0 | 158.8 |
| make-doom-for-mips | gemini-cli | openhands | 0.000 | 139.6 | 156.5 |
| train-fasttext | gemini-cli | terminus-3-3 | 0.100 | 127.6 | 154.7 |

This is the cleanest empirical form of the benchmark thesis so far: even on the exact same task, and sometimes with the exact same success rate, the path through the task can differ by more than 100 tool calls.

The same-model/different-harness contrast report directly addresses harness effects. Examples:

| Model | Agent A | Agent B | Success Gap | Tool Call Gap | Process Gap |
|---|---|---|---:|---:|---:|
| gpt-5-nano@openai | mini-swe-agent | terminus-2 | 0.011 | 104.2 | 116.9 |
| gpt-5-nano@openai | codex | terminus-2 | 0.036 | 104.7 | 115.5 |
| gemini-2.5-flash@gemini | gemini-cli | openhands | 0.009 | 65.7 | 83.3 |
| claude-sonnet-4-5-20250929@anthropic | claude-code | openhands | 0.029 | 63.1 | 79.7 |
| claude-sonnet-4-5-20250929@anthropic | mini-swe-agent | openhands | 0.001 | 52.9 | 73.6 |

These rows are strong evidence that harnesses are not passive wrappers. Holding the model fixed can still leave very large movement-pattern differences.

The same-harness/different-model contrast report is messier because model changes often move both success and process. Still, it is useful for decomposition: within `terminus-2`, `gpt-5-nano@openai` produces far more tool calls than many other models, while within `openhands`, different models can shift both success and process volume substantially. This supports a two-factor framing: harness sets the action grammar and many process priors; model choice changes capability, recovery, and sometimes exploration intensity within that grammar.

The task-centered residual report gives each agent's average deviation from task baselines on traced rows:

| Agent | Success Residual | Tool Call Residual | Intent Entropy Residual | Failure Rate Residual | Duration Residual |
|---|---:|---:|---:|---:|---:|
| terminus-3-3 | 0.420 | 3.3 | 0.289 | -0.044 | -16s |
| judy | 0.405 | 5.9 | -0.715 | -0.041 | 149s |
| deepagent-harbor | 0.357 | -14.4 | -0.140 | -0.002 | 72s |
| claude-code | 0.114 | -7.5 | 0.042 | -0.014 | -131s |
| codex | 0.019 | -26.0 | -0.380 | 0.018 | -300s |
| terminus-2 | 0.003 | 2.4 | 0.125 | -0.011 | 59s |
| openhands | -0.032 | 29.0 | 0.111 | -0.037 | 197s |
| mini-swe-agent | -0.087 | -21.1 | -0.303 | 0.039 | -198s |
| gemini-cli | -0.125 | -24.8 | -0.400 | 0.242 | -169s |

This table is probably more useful than raw success rates for the paper. It says, for example, that `codex` is very compact and fast relative to task baselines but only slightly above task-centered success in the traced subset, while `deepagent-harbor` is above baseline with fewer tool calls. `openhands` is below baseline despite substantially above-baseline tool volume.

## Why Traditional Benchmarks Are Under-Resolving

A leaderboard answer like "agent A scored 45% and agent B scored 40%" is materially incomplete. It does not tell us:

- whether one system read 5 files while another read 50
- whether one system edited directly while another searched broadly first
- whether one system repeated failing commands
- whether cost and latency were spent on useful localization or loops
- whether the system respected task boundaries
- whether the path was robust enough to trust in larger repositories

The public data already supports this weaker but important claim: process variation is large enough that final success cannot be a sufficient behavioral description.

The stronger blast-radius claim requires Harbor-native file-level traces plus gold context labels. That is what this repo is set up to provide.

## Proposed Constructs And Metrics

### Focus

Does the agent stay near task-relevant context?

Metrics:

- required file recall
- target file recall
- justified file precision
- context bloat ratio
- directory dispersion
- graph distance from target files

### Curiosity

How readily does the agent expand beyond the immediate task surface?

Metrics:

- unique file reads
- unique search paths
- pre-edit tool calls
- search-to-read ratio
- entropy over directories

### Boundary Respect

Does the agent avoid explicitly irrelevant or forbidden context?

Metrics:

- forbidden read rate
- forbidden search rate
- secret/config access rate on safety tasks
- policy-violating tool calls

### Discipline

Does the agent gather enough evidence and then act, without looping?

Metrics:

- first edit step
- pre-edit tool calls
- repeated command rate
- longest same-category run
- category switch rate

### Resilience

How does the agent recover from failed actions?

Metrics:

- failed command count
- command-after-failure diversity
- verification attempts after edit
- edit-test-repair cycles
- narrowing versus broadening after failure

## Next Implementation Work

Completed in the current code pass:

1. Task-centered process correlations.
2. Aggregate agent contrast report for similar-success, different-process pairs.
3. Coarse command-intent parsing for public traces.
4. Behavior-signature CSV and dashboard sections.
5. Same-task agent contrast report.
6. First parser-tightening pass that reduced the command-intent `other` bucket.
7. First failure/recovery feature pass.
8. Same-model and same-harness contrast reports for model/harness decomposition.
9. Lightweight task-centered residual report.

The highest-value next code changes are now:

1. Continue command-intent parsing to shrink the remaining `other` bucket.
2. Refine failure-recovery features from observations, especially command failures followed by narrowing or broadening actions.
3. Add a `trace-signature` CLI command that emits one row per run and one row per agent.
4. Add bootstrap confidence intervals for contrast metrics so the figures are less brittle.
5. Add formal mixed-effects or fixed-effects regression outside the lightweight report code.

The highest-value benchmark work is:

1. Author 6 to 10 controlled Harbor-native tasks across multiple context-frontier types.
2. Run at least 3 harnesses and 3 model families on the same tasks.
3. Label required, allowed, forbidden, and target files with two annotators.
4. Use public Terminal-Bench only as the broad external validation layer until it has enough file-level context.

## Paper Shape

Working title:

> Beyond Pass/Fail: Measuring Repository Traversal Signatures in Coding Agents

Core contribution:

1. A Harbor-native replay and scoring layer for repository traversal behavior.
2. A trace-derived vocabulary for model-harness tuple behavior.
3. An empirical demonstration that coding agents with comparable outcomes can have substantially different movement patterns.

Main result to aim for:

> Success-equivalent agents are not behavior-equivalent.

That sentence is the center of the project.

# Publication Plan

## Core framing

`blastRadiusBench` should be framed as an evaluation layer on top of Harbor, not as a competing harness.

Harbor already provides:

- agent execution
- dataset integration
- task packaging
- trace export through `ATIF`

`blastRadiusBench` adds a new benchmark question:

> How much repository context did the agent consume, and how justified was that context relative to the task?

That distinction matters. The publishable contribution is not "we ran agents on coding tasks." The publishable contribution is "we make repository traversal legible and measurable across existing Harbor-compatible agents and tasks."

## Why this can be a paper

The strongest paper version has three contributions:

1. A Harbor-native replay and scoring layer for trajectory-based repository-access evaluation.
2. A task annotation protocol for `required`, `allowed`, `forbidden`, and `target` context.
3. An empirical study showing that agents with similar task success can behave very differently in repository traversal, tool usage, cost, and context bloat.

The first regex task already demonstrates the main empirical shape:

- two agents can both solve the task
- one agent can stay narrow
- another agent can explore more broadly
- success alone hides that behavioral difference

## Main research questions

The initial paper should answer a small number of sharp questions:

1. How much unnecessary repository context do coding agents consume while solving tasks?
2. How does context breadth relate to task success, latency, tool use, and cost?
3. Do stronger models or different agent scaffolds consume less context, or simply consume context differently?
4. Which task categories reliably require repository reuse, and which categories are solvable by zero-shot editing?

## Benchmark design

The benchmark should explicitly separate:

- functional outcome: did the agent solve the task?
- context behavior: what did the agent read or search?
- qualitative trajectory quality: did the exploration look justified or erratic?

That leads to three score families:

### Deterministic core

- `required_file_recall`
- `target_file_recall`
- `justified_file_precision`
- `forbidden_read_rate`
- `context_bloat_ratio`
- tool-call counts and histograms
- token, time, and cost totals

### Trajectory-shape metrics

These are strong v2 candidates:

- time to first target read
- time to first edit
- search-to-read funnel
- failed or nonexistent read attempts
- directory dispersion or entropy
- revisit ratio
- edit-to-read locality

### Judge-based metrics

Use LLM judges only for questions that deterministic parsing cannot answer cleanly:

- whether broad exploration was justified by ambiguity
- whether the agent appeared stuck
- whether the agent ignored obvious local evidence

This should remain secondary to deterministic scoring. Reviewers will trust the work more if the benchmark is mostly programmatic.

## Dataset plan

The paper should not rely on a single task type.

Use a two-tier dataset:

### Tier 1: controlled Harbor-native tasks

These tasks are authored specifically for `blastRadiusBench` so the gold context is defensible.

Recommended categories:

- single-file zero-shot-satisfiable tasks
- helper-reuse tasks where reading an internal module is clearly useful
- retrieval-required tasks where a hidden helper is necessary to pass
- distractor-rich tasks with misleading filenames or irrelevant docs
- multi-file dependency-frontier tasks
- test-led tasks where inspecting tests is either required or explicitly not useful

The current `regex-book-search` task belongs in the first bucket.

### Tier 2: adapted Harbor datasets

Once the metrics stabilize, annotate a subset of existing Harbor-runnable benchmarks such as Terminal-Bench or other repository-edit tasks.

WolfBench is now the best large-scale external corpus for this tier. Its GitHub artifacts provide repeated Terminal-Bench 2.0 runs across multiple agents, models, thinking settings, and timestamps. The run-level JSON is enough to study outcome variance, per-task reliability, duration/error metadata, and WolfBench's five-metric profile: ceiling, best-of, average, worst-of, and solid. Full blast-radius scoring still needs trajectory-level joins through the W&B Weave artifacts referenced by WolfBench.

This is where the Harbor-over-Harbor story becomes strongest:

- existing benchmarks provide realism
- Harbor provides execution and trace standardization
- `blastRadiusBench` provides the additional evaluation layer

## Annotation protocol

The paper will be weak if the gold context looks subjective.

The minimum defensible process is:

1. Two annotators independently label `required`, `allowed`, `forbidden`, and `target` files.
2. Disagreements are adjudicated and recorded.
3. A constrained oracle attempt is made using only the proposed gold context.
4. If the task can be solved without some `required` file, that file should be downgraded or the task should be re-categorized.

This is also why the benchmark should not pretend there is always one unique minimal set. The tiered labels are a feature, not a compromise.

## Experimental plan

The first serious study should look like this:

- 20 to 40 controlled tasks
- 3 to 5 agent systems
- 2 to 4 model families
- multiple seeds or repeated runs when agents are nondeterministic
- WolfBench as an external validity layer for Terminal-Bench 2.0 outcome variance before trace-level context labels are available

The key result is not one scalar leaderboard. The key result is a set of tradeoffs:

- success versus context bloat
- success versus cost
- success versus required-context recall
- task category versus exploration behavior
- reliability versus ceiling: which agents can occasionally solve broad task sets versus solve the same tasks consistently

## Figures that will matter

The spider plot is fine as a supplementary figure, but it should not be the main figure.

The main figures should probably be:

- a Pareto-style scatter of success versus context bloat
- per-agent box plots for justified precision and bloat ratio
- a category heatmap across task families
- trajectory funnel plots from search to content read to edit
- a WolfBench consistency chart showing solid, average, and ceiling scores for the same model under different agents or settings
- harness x model spider fingerprints that show behavioral skews: success, task lift, tool intensity, exploration diversity, search/read intensity, edit eagerness, verification intensity, and recovery churn

## What makes the work distinct

There are already context-oriented coding benchmarks. The distinction for `blastRadiusBench` should be:

- it is trajectory-native rather than only end-state oriented
- it is Harbor-compatible and can score existing agent runs
- it measures repository traversal behavior, not only retrieved context quality
- it preserves cost and tool-use tradeoffs alongside task outcome
- it makes harness/model behavioral style legible as a multi-axis fingerprint instead of compressing everything into a single leaderboard number

In short, the project is best described as a reusable benchmark layer for agent trajectory analysis.

## Risks

The biggest risks are straightforward:

- tasks are too easy and agents zero-shot them
- gold context is too subjective
- benchmark novelty overlaps too much with prior context benchmarks
- metrics reward narrow behavior that is not actually better

The mitigation is also straightforward:

- include both zero-shot and retrieval-required task families
- keep labels tiered instead of binary
- report multi-axis tradeoffs rather than claiming "smaller blast radius is better"
- emphasize Harbor-native trace replay as the systems contribution

## Near-term plan

The next project phase should focus on four deliverables:

1. author 6 to 10 controlled tasks across at least three task categories
2. harden the scorer for multiple Harbor agents and trace shapes
3. define an annotation rubric and adjudication workflow
4. run a first cross-agent pilot and inspect which metrics are stable
5. use WolfBench run-level imports to select high-variance Terminal-Bench tasks worth trace-level annotation

If those four pieces hold up, the paper direction is real.

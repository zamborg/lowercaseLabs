# Research Framing

Date: March 11, 2026

## One-line framing

`blastRadiusBench` is a post-hoc evaluation layer for Harbor runs that measures how coding agents traverse repository context, not just whether they eventually solve the task.

## Where this sits in the literature

The adjacent literature falls into five buckets.

### 1. Outcome-centric coding benchmarks

These works establish realistic repository-level evaluation, but they mostly score end success:

- `SWE-bench`
- `SWE-PolyBench`
- `Terminal-Bench`
- `GitTaskBench`

This line of work proves that repository-level tasks matter, but it usually does not tell us how the agent navigated the repository to get there.

### 2. Context-centric coding benchmarks

This is the closest prior art:

- `ContextBench` measures context retrieval in coding agents with gold contexts and trajectory analysis.
- `SWE-ContextBench` studies experience reuse across related software tasks.

These papers validate the general thesis that context behavior is measurable and important. They also raise the bar for us. `blastRadiusBench` should therefore not claim to have invented context-sensitive coding evaluation.

The stronger distinction is this:

- `ContextBench` is a standalone benchmark for issue-resolution context retrieval.
- `SWE-ContextBench` studies retrieval of prior experience across task sequences.
- `blastRadiusBench` is a Harbor-native scoring layer for repository traversal behavior across existing or newly authored Harbor tasks and agents.

In other words, our contribution is best framed as reusable trajectory analysis and evaluation infrastructure, plus a task suite optimized for measuring repository blast radius.

### 3. Trajectory and process evaluation

Several recent papers argue that outcome-only evaluation hides crucial behavior:

- `AgentBoard` argues for multi-faceted analytical evaluation rather than final success alone.
- `TRAJECT-Bench` measures tool trajectory quality rather than just final answers.
- `Understanding Code Agent Behaviour` analyzes successful and failed coding trajectories.
- `AgentRx` localizes failure steps from execution traces.

These are strong evidence that trajectory-native evaluation is now a live research direction. They support the methodological premise of `blastRadiusBench`: the trajectory itself is a meaningful scientific object.

### 4. Safety, trust, and boundary-respecting evaluation

Another line of work shows that process matters because agents can succeed while still behaving badly:

- `ST-WebAgentBench` measures policy-compliant completion rather than raw completion.
- `MT-AgentRisk` shows that multi-turn tool agents become substantially less safe in longer interactions.
- `AgentLeak` shows that internal agent traces reveal privacy risks missed by output-only audits.
- `ImpossibleBench` measures whether coding agents exploit tests rather than solve the intended task.

These works justify a broader notion of alignment for agents: not just “did it complete the task,” but “did it pursue the task in a bounded, policy-respecting, trustworthy way.”

### 5. Judge reliability and evaluation methodology

Because we plan to use LLM judges for ambiguous trajectory cases, we also have to inherit the judge-reliability literature:

- `Agent-as-a-Judge`
- `AgentRewardBench`
- `JudgeBench`
- `Judge Reliability Harness`
- `AI Agents That Matter`

The lesson from this bucket is straightforward:

- deterministic metrics should be primary
- judge-based scoring should be secondary
- any judge used in the benchmark needs validation against human or rule-based anchors

## Why blast radius matters

There are four strong reasons to care.

### 1. Efficiency

Repository exploration is expensive.

More file reads and broader searches usually mean:

- more prompt tokens
- more latency
- more tool calls
- more opportunities for the agent to get distracted

This matters directly for deployment cost and for practical developer experience.

### 2. Effectiveness

More context is not automatically better.

Long-context and RAG literature shows that irrelevant or poorly ordered context can hurt model performance rather than help it. In coding agents, this means a wider blast radius can reflect either useful exploration or harmful context bloat.

This makes blast radius scientifically interesting rather than merely operational.

### 3. Reliability and failure analysis

Trajectory shape often reveals whether the agent is behaving coherently:

- does it find the target file early
- does it stay near the dependency frontier
- does it recover from failed hypotheses
- does it wander across the repository without a clear plan

These are not visible in a pass/fail score.

### 4. Alignment, trust, and safety

Repository access is also a boundary-respecting behavior.

An agent that reads far beyond the task boundary may:

- expose secrets or sensitive files
- rely on irrelevant tests or hidden hints
- violate user instructions about scope
- appear “curious” in a way that is costly or risky rather than useful

So blast radius can act as a concrete, domain-specific measure of task alignment: how well the agent keeps its behavior bounded to justified context.

## The behavioral constructs we can study

The benchmark is interesting if it does not reduce everything to “smaller is better.”

What we actually care about are task-conditioned behavioral traits:

- `focus`: does the agent stay near the target and dependency frontier?
- `curiosity`: how readily does it widen its search beyond local evidence?
- `boundary respect`: does it avoid forbidden or irrelevant regions?
- `resilience`: after a failed attempt, does it recover with targeted investigation or with broad wandering?
- `discipline`: does it gather just enough evidence before editing?

These are better constructs than a single “blast radius score.” The benchmark should expose them as separate axes.

## Why this measurement approach is good

The proposed method is defensible for four reasons.

### 1. Harbor already standardizes execution

Harbor gives us:

- reproducible task packaging
- support for many agents
- standardized trajectories through `ATIF`

That means `blastRadiusBench` does not need to solve orchestration; it can focus on measurement.

### 2. Gold context can be tiered instead of pretending to be exact

The minimum context set is often ambiguous. The `required`, `allowed`, `forbidden`, and `target` split handles this directly.

That is scientifically better than pretending every task has one unique correct context footprint.

### 3. Deterministic metrics come from the trajectory itself

Most of the core benchmark can be computed mechanically from trajectory logs:

- what was searched
- what was read
- when it was read
- how often the agent wandered
- how much it cost

This keeps the main benchmark grounded in auditable measurements rather than purely subjective judging.

### 4. LLM judges can be reserved for ambiguity

Judges are useful for questions like:

- was a broad search justified?
- did the agent appear stuck?
- did it ignore obvious evidence?

But these should augment, not replace, the deterministic layer.

## The gap we should claim

The gap is not “no one measures context.”

That claim is already false because of `ContextBench` and adjacent work.

The real gap is narrower and stronger:

1. existing coding benchmarks still overwhelmingly optimize for task success
2. existing context/process benchmarks do not yet provide a Harbor-native, reusable repo-traversal scoring layer across heterogeneous agents and tasks
3. the field still lacks a clean way to study repository access as a behavioral axis alongside success, cost, and trustworthiness

That is the claim `blastRadiusBench` can support.

## Four research questions

These should be the core research questions for the paper.

### RQ1. Do agents with similar functional success exhibit systematically different repository blast radii?

Hypothesis:

Agents with comparable task success will still differ substantially in file-read breadth, search breadth, and out-of-scope exploration, with scaffold effects remaining visible even when model capability is similar.

Why it matters:

If true, benchmark leaderboards that report only success are hiding large behavioral differences relevant to cost, trust, and usability.

### RQ2. When does broader exploration help, and when is it merely context bloat?

Hypothesis:

On easy or low-frontier tasks, additional exploration mostly increases cost without improving success. On helper-required or retrieval-required tasks, targeted exploration improves success, but indiscriminate breadth still hurts efficiency.

Why it matters:

This separates productive exploration from unproductive curiosity.

### RQ3. Which trajectory patterns predict success, failure, and recovery?

Hypothesis:

Successful runs will find target files and relevant local context early, then edit with relatively local follow-up reads. Failed runs will be longer, more repetitive, and more likely to show broad search loops, nonexistent reads, or delayed localization.

Why it matters:

This turns blast radius from a descriptive metric into a diagnostic one.

### RQ4. Can repository blast radius serve as a measurable proxy for task alignment and boundary-respecting behavior?

Hypothesis:

Agents that exhibit higher forbidden or out-of-scope access rates will also show weaker instruction adherence and poorer policy-compliant behavior, even when their raw task success is high.

Why it matters:

This connects repository traversal to trustworthiness and makes the benchmark relevant to safety and alignment discussions rather than only efficiency.

## Experiments needed to support the claims

### Experiment 1. Cross-agent pilot on controlled Harbor-native tasks

Run multiple Harbor-supported agents and models on the same small task suite.

Measure:

- success
- blast-radius metrics
- cost
- latency

Goal:

Show that functional parity can mask behavioral divergence.

This directly supports `RQ1`.

### Experiment 2. Task-family stratification

Author tasks in at least four categories:

- zero-shot-satisfiable single-file tasks
- helper-reuse tasks
- retrieval-required tasks
- distractor-rich or boundary-sensitive tasks

Goal:

Show when exploration is useful and when it is wasteful.

This directly supports `RQ2`.

### Experiment 3. Trajectory-shape analysis

Add features such as:

- time to first target read
- time to first edit
- search-to-read ratio
- failed-read or nonexistent-read rate
- revisit ratio
- dispersion across directories

Goal:

Relate trajectory shape to success, failure, and recovery behavior.

This directly supports `RQ3`.

### Experiment 4. Boundary-respect tasks

Create tasks with explicitly forbidden but tempting files:

- irrelevant tests
- misleading helper modules
- docs with decoy hints
- optionally mock sensitive files or secrets

Goal:

Measure whether the agent respects scope constraints under temptation.

This directly supports `RQ4`.

### Experiment 5. Annotation and judge validation

For a subset of tasks:

- collect two independent human annotations of gold context
- adjudicate disagreements
- validate any judge-based labels against those annotations

Goal:

Demonstrate that the benchmark is not built on arbitrary labels or unreliable judges.

This is necessary for the whole paper, especially because `ContextBench` has already raised expectations around gold-context quality.

## Experimental readout

The paper should not collapse results into one scalar leaderboard.

The main readouts should be:

- success versus context-bloat scatter plots
- per-agent distributions for justified precision and bloat ratio
- category-level heatmaps
- trajectory funnel plots from search to read to edit
- boundary-respect tables reporting forbidden access and policy-compliant success

## What this project should and should not claim

### Claims we can support

- repository traversal is a meaningful independent evaluation axis
- Harbor trajectories can be post-hoc scored for blast radius
- agents with similar success can behave very differently
- broader access is sometimes useful and sometimes wasteful
- blast radius provides trust and diagnostic signals not visible in pass/fail metrics

### Claims we should avoid

- smaller blast radius is always better
- one gold context is always uniquely correct
- LLM judges are fully reliable
- our benchmark replaces existing coding benchmarks

## Bottom line

The research goal is not to crown the “smallest” agent.

The research goal is to make repository traversal measurable, interpretable, and comparable across agents, tasks, and models, so that the field can study focus, curiosity, resilience, and boundary respect as first-class properties of coding agents.

## Sources

- `ContextBench`: https://arxiv.org/abs/2602.05892
- `SWE-ContextBench`: https://arxiv.org/abs/2602.08316
- `SWE-PolyBench`: https://arxiv.org/abs/2504.08703
- `Terminal-Bench`: https://openreview.net/forum?id=a7Qa4CcHak
- `GitTaskBench`: https://arxiv.org/abs/2508.18993
- `AgentBoard`: https://arxiv.org/abs/2401.13178
- `TRAJECT-Bench`: https://arxiv.org/abs/2510.04550
- `Understanding Code Agent Behaviour`: https://arxiv.org/abs/2511.00197
- `AgentRx`: https://arxiv.org/abs/2602.02475
- `ST-WebAgentBench`: https://arxiv.org/abs/2410.06703
- `MT-AgentRisk`: https://arxiv.org/abs/2602.13379
- `AgentLeak`: https://arxiv.org/abs/2602.11510
- `ImpossibleBench`: https://arxiv.org/abs/2510.20270
- `AI Agents That Matter`: https://openreview.net/forum?id=Zy4uFzMviZ
- `Agent-as-a-Judge`: https://arxiv.org/abs/2410.10934
- `AgentRewardBench`: https://arxiv.org/abs/2504.08942
- `JudgeBench`: https://arxiv.org/abs/2410.12784
- `Judge Reliability Harness`: https://arxiv.org/abs/2603.05399
- `Lost in the Middle`: https://arxiv.org/abs/2307.03172
- `RE-RAG`: https://arxiv.org/abs/2406.05794

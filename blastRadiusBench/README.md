# blastRadiusBench

`blastRadiusBench` measures how widely coding agents traverse a repository relative to the minimum context a task appears to justify.

This repo is built around Harbor and Harbor's trajectory format, which is named `ATIF` (Agent Trajectory Interchange Format). The immediate goal is to score exported Harbor traces against task-level gold context annotations and make those scores usable for both deterministic analysis and LLM-as-a-judge review.

## MVP

The MVP in this repo does four things:

1. Parse Harbor `ATIF` trajectory JSON.
2. Normalize file-read and repo-search behavior from tool calls.
3. Score that behavior against an annotated task spec with `required`, `allowed`, and `forbidden` context.
4. Generate a compact judge prompt that can later be wrapped by Weave scorers.

The MVP is intentionally centered on small, Harbor-native repo-edit tasks where gold context is defensible. That gives us a clean place to validate the primitives before extending to broader Harbor datasets such as Terminal-Bench.

## V2

The planned v2 extends the same scoring model to:

- Harbor dataset adapters and task subsets such as Terminal-Bench and other repo-centric coding tasks.
- WolfBench run-level imports for Terminal-Bench 2.0 consistency, outcome, and variance analysis.
- Weave-based online evaluation and observability.
- Richer annotations at symbol and dependency-frontier level.
- Human and LLM adjudication for ambiguous reads.
- Aggregate visualizations such as radar/spider plots, histograms, and per-agent comparison dashboards.

## Project Layout

- `docs/benchmark-design.md`: benchmark object model, scoring axes, MVP and v2 scope.
- `docs/publication-plan.md`: paper framing, dataset strategy, and experimental plan.
- `docs/research-framing.md`: literature positioning, motivations, hypotheses, and experiments.
- `docs/research-notes.md`: current research grounding and source links.
- `src/blast_radius_bench/`: Python package for loading trajectories, scoring them, building judge prompts, and summarizing public run corpora.
- `tasks/`: Harbor-native benchmark tasks.
- `task_specs/`: gold-context specs used by `blast-radius-bench` scoring.
- `tests/`: fixtures and unit tests for the core primitives.

## First task

The first authored task is `tasks/regex-book-search`.

It is intentionally easy:

- The agent must implement `src/regex.py`.
- The real solution logic is already split across two oddly named helper files.
- There are a few irrelevant files and docs that should not be necessary.

The paired gold-context annotation lives at `task_specs/regex-book-search.yaml`.

## Quickstart

```bash
uv sync --group dev
uv run blast-radius-bench score \
  tests/fixtures/sample_trajectory.json \
  tests/fixtures/sample_task.yaml \
  --json
```

To review a Harbor job end to end:

```bash
uv run blast-radius-bench review-job \
  jobs/<job-name> \
  task_specs/<task-spec>.yaml
```

To analyze public Terminal-Bench rollout corpora and emit a static report:

```bash
uv run blast-radius-bench tb-public-report \
  yoonholee/terminalbench-trajectories \
  --output-dir reports/terminalbench-public
```

That command writes CSV summaries, PNG plots, and an `index.html` dashboard to the chosen output directory.

To analyze published WolfBench run-level artifacts:

```bash
uv run blast-radius-bench wolfbench-report \
  https://github.com/wandb/WolfBench/tree/main/wolfbench-runs \
  --output-dir reports/wolfbench \
  --max-runs 25
```

For full-corpus WolfBench reports, use a local checkout or set `GITHUB_TOKEN` to avoid GitHub API rate limits. The GitHub data contains run configs and task rewards; full trajectory-level blast-radius scoring requires joining the W&B Weave traces referenced by WolfBench.

To render heuristic harness x model spider fingerprints from an existing report:

```bash
uv run blast-radius-bench spider-report \
  reports/terminalbench-public/yoonholee-terminalbench-trajectories \
  --output-dir reports/spider/terminalbench-public \
  --min-runs 25 \
  --min-trace-coverage 0.25 \
  --top-groups 12
```

The spider report auto-detects public Terminal-Bench trace reports and WolfBench reliability reports. Trace spiders are behavioral fingerprints: high values mean more of a property such as tool intensity, search/read intensity, early edit behavior, verification intensity, or recovery churn. They are not a single quality score.

## Task Spec Shape

```yaml
task_id: sample.single-file-fix
dataset: harbor-local
repo_root_aliases:
  - /workspace
  - /app
gold_context:
  required_files:
    - src/app.py
  allowed_files:
    - src/helpers.py
  forbidden_files:
    - docs/architecture.md
  target_files:
    - src/app.py
```

# Griff/Claude Investigation Handoff

Date: June 23, 2026

## Current State

`blastRadiusBench` now has two report paths that matter for the paper direction:

1. WolfBench run-level ingestion and reliability reporting.
2. Heuristic harness x model spider fingerprints.

The pushed baseline is:

- commit: `224940f Add WolfBench and spider fingerprint reports`
- branch: `main`
- remote: `origin/main`

Important repo caveat: the parent git repo currently has unrelated dirty work under `../theVoidLocal`. Do not touch or stage those files while investigating `blastRadiusBench`.

## Core Research Question

The goal is not just to rank agents by task success. The goal is to show that the same or similar task outcomes can hide very different agent behaviors.

The key object is a harness x model behavioral fingerprint:

> Given a fixed benchmark/task corpus, how does each harness/model cell explore, read, edit, verify, recover, and vary across runs?

The current spider plots are a first heuristic version of that fingerprint.

## Current Artifacts To Inspect

Code:

- `src/blast_radius_bench/wolfbench.py`
- `src/blast_radius_bench/spider.py`
- `src/blast_radius_bench/public_terminalbench.py`
- `src/blast_radius_bench/cli.py`

Tests:

- `tests/test_wolfbench.py`
- `tests/test_spider.py`
- `tests/test_public_terminalbench.py`

Generated report:

- `reports/spider/yoonholee-terminalbench-trajectories/index.html`
- `reports/spider/yoonholee-terminalbench-trajectories/spider_summary.csv`
- `reports/spider/yoonholee-terminalbench-trajectories/spider_axes.csv`
- `reports/spider/yoonholee-terminalbench-trajectories/spider_overlay.png`
- `reports/spider/yoonholee-terminalbench-trajectories/spider_grid.png`
- `reports/spider/yoonholee-terminalbench-trajectories/spider_axis_heatmap.png`

## Commands

Run tests:

```bash
uv run pytest
```

Regenerate the current trace-backed spider report:

```bash
uv run blast-radius-bench spider-report \
  reports/terminalbench-public/yoonholee-terminalbench-trajectories \
  --output-dir reports/spider/yoonholee-terminalbench-trajectories \
  --min-runs 100 \
  --min-trace-coverage 0.25 \
  --top-groups 12
```

Generate a sampled WolfBench report:

```bash
uv run blast-radius-bench wolfbench-report \
  https://github.com/wandb/WolfBench/tree/main/wolfbench-runs \
  --output-dir reports/wolfbench \
  --max-runs 25
```

Render spider plots from a WolfBench report:

```bash
uv run blast-radius-bench spider-report \
  reports/wolfbench \
  --output-dir reports/spider/wolfbench \
  --source-type wolfbench \
  --min-runs 2
```

## Current Spider Axes

Trace mode uses eight normalized axes:

- `success`: raw success rate for the harness/model cell.
- `task_lift`: success above or below task-average baseline.
- `tool_intensity`: relative mean tool-call volume.
- `exploration_diversity`: intent/tool-category entropy.
- `search_read_intensity`: share of tool calls classified as search or read.
- `edit_eagerness`: inverted mean first-edit step; higher means earlier edits.
- `verification_intensity`: share of test/build calls.
- `recovery_churn`: failures, repeated commands, and post-failure activity.

WolfBench mode uses eight run-level reliability axes:

- `average`
- `solid`
- `ceiling`
- `best`
- `reliability`
- `variance_gap`
- `error_pressure`
- `speed`

These are fingerprints, not a scalar leaderboard. High values mean "more of this property"; some properties are favorable, while others describe behavioral skew.

## Highest-Value Investigation Steps

### 1. Audit the spider axis definitions

Check whether the current eight axes are the right eight. The likely improvement is to split the axes into two types:

- outcome axes: success, task lift, reliability
- process axes: exploration, tool intensity, search/read behavior, verification, recovery

Questions:

- Are any current axes redundant?
- Is `edit_eagerness` too coarse?
- Should `tool_intensity` be inverted or kept as a behavioral skew?
- Should `recovery_churn` distinguish useful recovery from thrashing?

Deliverable:

- A proposed v2 axis table with names, formulas, expected interpretation, and whether high is "good" or just "more".

### 2. Validate normalization

Current spider values are min-max normalized within a report. That makes shapes easy to compare inside one corpus but unsafe to compare across corpora.

Questions:

- Should axes use robust percentiles instead of min-max?
- Should we clamp to the 5th/95th percentiles?
- Should we normalize within task families, within harnesses, or globally?
- How sensitive are top spider shapes to outliers?

Deliverable:

- A small sensitivity analysis comparing min-max, percentile rank, and clipped z-score normalization.

### 3. Separate trace coverage from behavior

The first generated plot surfaced high-success groups with no embedded traces. We added `--min-trace-coverage`, but this deserves more thought.

Questions:

- Should trace coverage be its own axis?
- Should low-coverage groups be shown in a separate "outcome-only" panel?
- What threshold is defensible for a paper figure?

Deliverable:

- A recommendation for trace coverage policy and figure inclusion criteria.

### 4. Add true blast-radius axes

The current public trace report has tool and intent metrics, but not file-level context quality. The paper needs actual blast-radius behavior.

Candidate axes:

- context discipline: justified reads / total reads
- target seeking: time to first required or target file read
- context bloat: out-of-scope reads / total reads
- forbidden contact: forbidden reads / total reads
- locality: edit files near read files or target files
- directory dispersion: entropy over directories read/searched

Questions:

- Which trace sources include enough path-level detail?
- Can WolfBench Weave traces be joined back to task outcomes?
- Can Harbor ATIF and public Terminal-Bench traces share one path-read abstraction?

Deliverable:

- A concrete design for adding path-level `ReadObservation` extraction to public/Weave traces, ideally reusing the existing ATIF scorer concepts.

### 5. Use WolfBench to select tasks

WolfBench gives repeated Terminal-Bench 2.0 outcomes. Use it to choose tasks worth deeper trace annotation.

Look for:

- high ceiling, low solid tasks
- tasks with large same-model harness gaps
- tasks solved by some harnesses but never by others
- tasks with frequent timeouts or verifier failures

Deliverable:

- A shortlist of 10 to 20 Terminal-Bench tasks that are promising for trace-level context annotation.

### 6. Compare same-model harness effects

The strongest empirical story may be:

> Holding the model fixed, harness choice changes behavior shape as much as or more than model choice.

Questions:

- For the same model, which harnesses have the largest spider distance?
- Do high-success harnesses cluster by early edit, verification, or exploration?
- Are some harnesses consistently broader but not more successful?

Deliverable:

- A `same_model_spider_contrasts.csv` table and one figure that overlays harnesses for the same model.

### 7. Compare same-harness model effects

The mirror analysis:

> Holding the harness fixed, stronger models may shift the shape from exploratory/recovery-heavy to direct/verification-heavy.

Questions:

- Does model quality reduce tool intensity?
- Does model quality increase or decrease verification?
- Does model quality reduce recovery churn?

Deliverable:

- A `same_harness_model_contrasts.csv` table and one figure that overlays models inside one harness.

### 8. Improve the HTML report for paper review

The current HTML is functional. It should become a reviewer-friendly exploratory artifact.

Possible improvements:

- add text definitions for each axis next to the plots
- add sorting controls for success, task lift, spider area, and trace coverage
- add "same model" and "same harness" sections
- add downloadable CSV links
- add a warning that axes are normalized within report

Deliverable:

- A clearer `index.html` report that can be sent to collaborators without explanation.

## Recommended First Pass

Start with analysis rather than implementation.

1. Read `src/blast_radius_bench/spider.py`.
2. Open `reports/spider/yoonholee-terminalbench-trajectories/spider_summary.csv`.
3. Identify three surprising harness/model fingerprints.
4. Explain whether each surprise is likely real or a metric artifact.
5. Propose changes to the eight-axis heuristic.

The best next code change is probably not more plotting. It is likely either:

- robust normalization, or
- true path-level blast-radius axes.

## What Not To Do Yet

- Do not collapse the spider into one score.
- Do not claim the current trace spider measures file-level context discipline.
- Do not compare normalized values across unrelated report sources.
- Do not stage or modify `../theVoidLocal` files.
- Do not treat WolfBench GitHub JSON as full trajectory data; it is run-level/task-reward data. Full trace work needs Weave or another trace source.

## Expected Output From Griff/Claude

A useful investigation response should include:

1. A critique of the current eight axes.
2. A proposed v2 axis schema.
3. A list of high-variance or high-contrast tasks worth annotating.
4. A recommendation for whether the first paper figure should be:
   - spider small multiples,
   - same-model overlays,
   - an axis heatmap,
   - or a paired success-vs-behavior scatter.
5. Any concrete implementation notes for `spider.py`, `public_terminalbench.py`, or `wolfbench.py`.

# Benchmark Design

## Why this benchmark exists

Coding-agent evaluations usually optimize for task success. That misses a separate behavior: how much repository context the agent consumes while solving the task.

`blastRadiusBench` is meant to measure that behavior directly. The benchmark is not trying to prove that "less context is always better." It is trying to make repository traversal legible:

- Which files were actually read?
- Which directories were searched?
- Which reads were clearly justified by the task?
- Which reads were likely spillover?
- How much tool use, token use, and search breadth accompanied the solution attempt?

## Core object model

The benchmark operates on three inputs:

1. A Harbor `ATIF` trajectory.
2. A task spec with gold context annotations.
3. Optional LLM-as-a-judge rubrics for ambiguous cases.

The task spec uses three main context tiers:

- `required_files`: files the agent should almost certainly inspect to complete the task correctly.
- `allowed_files`: files that are plausibly justified, even if not strictly required.
- `forbidden_files` and `forbidden_globs`: files or areas that are strong negative controls for the task.

This tiering is deliberate. The "minimum seen subset" is often hard to define exactly, so the benchmark should not collapse everything into a single binary label.

## Scoring axes

The deterministic MVP in this repo scores:

- `required_file_recall`: fraction of required files that were seen.
- `target_file_recall`: fraction of target edit files that were seen.
- `justified_file_precision`: fraction of unique file reads that land in `required` or `allowed`.
- `forbidden_read_rate`: fraction of unique file reads that land in explicitly forbidden areas.
- `context_bloat_ratio`: fraction of unique file reads that were neither justified nor forbidden.
- `tool_call_histogram`: tool usage distribution across the trajectory.
- `token and cost totals`: prompt tokens, completion tokens, cached tokens, and cost when present in ATIF.

The benchmark also preserves repo-search events separately from file-content reads. That matters because wide `rg`, `find`, and `ls` use is often the first sign of context expansion even before content inspection.

## MVP scope

The MVP should stay narrow enough that annotations are defensible and tests are easy to run:

- Harbor-native tasks authored for this benchmark.
- Small repos with single-file and low-frontier multi-file edits.
- File-level gold context annotations.
- Deterministic scoring from ATIF trajectories.
- Judge prompt generation for later Weave integration.

This keeps the hard problem small: get the primitives right before scaling the dataset.

## V2 scope

V2 should add breadth only after the MVP metrics stabilize:

- Harbor dataset ingestion for Terminal-Bench subsets and other repo-edit datasets.
- Symbol-level and dependency-frontier annotations.
- Graph-distance metrics from edited files and dependency anchors.
- Weave scorers that combine deterministic metrics with rubric-based LLM judgment.
- Human adjudication workflows for disagreements between deterministic and LLM scores.
- Aggregate experiment reports and visual dashboards.

## Implementation direction

The package in `src/blast_radius_bench/` is structured around four responsibilities:

- `atif.py`: load and normalize Harbor trajectories.
- `metrics.py`: compute deterministic benchmark metrics.
- `judge.py`: build compact review prompts for LLM judges.
- `cli.py`: expose these primitives for local use and later Harbor job post-processing.

That is the right cut for now because it keeps Harbor as the runner and `blastRadiusBench` as the scorer and analysis layer.


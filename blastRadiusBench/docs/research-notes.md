# Research Notes

Date: March 10, 2026

Updated: June 19, 2026

## Current takeaways

- Harbor is the right execution harness because it already standardizes task authoring, dataset access, job orchestration, and trajectory export.
- Harbor's trajectory format is `ATIF`, not `AITF`.
- A file-level gold-context benchmark is realistic for an MVP; exact symbol-level "minimum seen subset" should wait for v2.
- Deterministic metrics should be primary, with LLM judgment used for ambiguity and qualitative analysis rather than as the only score.
- Terminal-Bench is a strong v2 target, but a small Harbor-native dataset is the better MVP because the gold context will be easier to defend.
- WolfBench is a strong external outcome-variance corpus because it publishes repeated Terminal-Bench 2.0 runs across agents and models. The GitHub data is run-level/task-reward data; repository traversal analysis still needs the associated Weave traces.
- The near-term visual target is a heuristic spider fingerprint per harness x model cell. It should show behavioral skew, not claim to be a scalar quality score.

## Sources

- Harbor evals docs: [harborframework.com/docs/evals](https://harborframework.com/docs/evals)
- Harbor task structure docs: [harborframework.com/docs/tasks](https://harborframework.com/docs/tasks)
- Harbor LLM-as-a-Judge example: [harborframework.com/docs/examples/llm-judge](https://harborframework.com/docs/examples/llm-judge)
- Harbor ATIF format reference: [harborframework.com/docs/trajectory-format](https://harborframework.com/docs/trajectory-format)
- Harbor Terminal-Bench docs: [harborframework.com/docs/running-tbench](https://harborframework.com/docs/running-tbench)
- Terminal-Bench paper and benchmark site: [terminal-bench.com](https://www.terminal-bench.com/)
- WolfBench repository and run artifacts: [github.com/wandb/WolfBench](https://github.com/wandb/WolfBench)
- WolfBench dashboard: [wolfbench.ai](https://wolfbench.ai/)
- ContextBench preprint: [arXiv:2602.09530](https://arxiv.org/abs/2602.09530)
- SWE-bench: [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- SWE-agent: [arXiv:2405.15793](https://arxiv.org/abs/2405.15793)
- Holistic Agent Leaderboard (HAL): [arXiv:2505.12111](https://arxiv.org/abs/2505.12111)
- AI Agents That Matter: [arXiv:2510.14657](https://arxiv.org/abs/2510.14657)
- Weights & Biases Weave docs: [docs.wandb.ai/weave](https://docs.wandb.ai/weave)

## Why these sources matter here

- Harbor provides the practical substrate we can build on today.
- Terminal-Bench, SWE-bench, and SWE-agent ground the benchmark in real coding-agent evaluation practice.
- WolfBench gives a large repeated-run Terminal-Bench 2.0 corpus for consistency analysis and task selection before deeper trace annotation.
- ContextBench is especially relevant because it treats context selection itself as a measurable behavior instead of only a hidden intermediate.
- HAL and AI Agents That Matter are useful for benchmark-design discipline: contamination, broad agent evaluation, and separation of execution from scoring.
- Weave is the right observability layer for the later judge path because it already supports tracing and evaluation workflows.

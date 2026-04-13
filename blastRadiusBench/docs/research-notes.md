# Research Notes

Date: March 10, 2026

## Current takeaways

- Harbor is the right execution harness because it already standardizes task authoring, dataset access, job orchestration, and trajectory export.
- Harbor's trajectory format is `ATIF`, not `AITF`.
- A file-level gold-context benchmark is realistic for an MVP; exact symbol-level "minimum seen subset" should wait for v2.
- Deterministic metrics should be primary, with LLM judgment used for ambiguity and qualitative analysis rather than as the only score.
- Terminal-Bench is a strong v2 target, but a small Harbor-native dataset is the better MVP because the gold context will be easier to defend.

## Sources

- Harbor evals docs: [harborframework.com/docs/evals](https://harborframework.com/docs/evals)
- Harbor task structure docs: [harborframework.com/docs/tasks](https://harborframework.com/docs/tasks)
- Harbor LLM-as-a-Judge example: [harborframework.com/docs/examples/llm-judge](https://harborframework.com/docs/examples/llm-judge)
- Harbor ATIF format reference: [harborframework.com/docs/trajectory-format](https://harborframework.com/docs/trajectory-format)
- Harbor Terminal-Bench docs: [harborframework.com/docs/running-tbench](https://harborframework.com/docs/running-tbench)
- Terminal-Bench paper and benchmark site: [terminal-bench.com](https://www.terminal-bench.com/)
- ContextBench preprint: [arXiv:2602.09530](https://arxiv.org/abs/2602.09530)
- SWE-bench: [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- SWE-agent: [arXiv:2405.15793](https://arxiv.org/abs/2405.15793)
- Holistic Agent Leaderboard (HAL): [arXiv:2505.12111](https://arxiv.org/abs/2505.12111)
- AI Agents That Matter: [arXiv:2510.14657](https://arxiv.org/abs/2510.14657)
- Weights & Biases Weave docs: [docs.wandb.ai/weave](https://docs.wandb.ai/weave)

## Why these sources matter here

- Harbor provides the practical substrate we can build on today.
- Terminal-Bench, SWE-bench, and SWE-agent ground the benchmark in real coding-agent evaluation practice.
- ContextBench is especially relevant because it treats context selection itself as a measurable behavior instead of only a hidden intermediate.
- HAL and AI Agents That Matter are useful for benchmark-design discipline: contamination, broad agent evaluation, and separation of execution from scoring.
- Weave is the right observability layer for the later judge path because it already supports tracing and evaluation workflows.

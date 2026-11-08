# Brief 01: tagging eval (Codex)

Build `eval/tagging/` in this directory. Python 3.11+ (use uv; system python is 3.9 and too old for mlx-lm).

Inputs: `eval/data/memos.jsonl` (lines: {id, transcript, label}) where label is one of the ids in `taxonomy.json`. Candidates listed in `eval/tagging/candidates.yaml` (HF repo ids of 4-bit MLX models, ~1-2B: e.g. LiquidAI LFM2 1.2B, Qwen3 1.7B, Llama 3.2 1B/3B, Gemma small).

Requirements:
- One shared prompt in `prompt.txt`, same for every candidate.
- Schema-constrained decoding for EVERY candidate (use outlines or mlx-lm logits processors / lm-format-enforcer) to the schema: {"title": string<=60 chars, "tags": [enum of taxonomy ids], minItems 1, maxItems 2}.
- Report per model: accuracy overall and per quadrant (confusion matrix), top-1 and any-of-tags, schema-valid rate, median/p95 latency to full output, tokens/sec, peak memory.
- Output `eval/tagging/report.md` + `results.json`. Deterministic (temperature 0).
- Include a `--limit N` flag and a fake 10-row sample so it runs without real data.
Done = `uv run python eval/tagging/run.py --limit 10` works end to end.

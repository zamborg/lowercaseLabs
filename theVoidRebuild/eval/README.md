# The Void evaluation

All source, virtual environments, model weights, API caches, inputs and reports live under `eval/`. The app and taxonomy are read only. The tagging benchmark runs on an Apple Silicon Mac with Metal; it does not use a hosted model for candidate inference.

From the repository root:

```sh
export UV_CACHE_DIR="$PWD/eval/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/eval/.python"
uv sync --project eval --frozen

# Existing cached data: no credentials, no paid calls. A cache miss fails.
uv run --project eval python eval/data/generate.py --n 300 --offline

# Ten examples from the generated set, or handwritten sample if it does not exist.
uv run --project eval python eval/tagging/run.py --limit 10 --offline --output-dir eval/tagging/quick

# Full comparison; writes report.md, results.json and timestamped per-model runs.
uv run --project eval python eval/tagging/run.py

# Explicit smoke set, without any generator/judge API calls.
uv run --project eval python eval/tagging/run.py \
  --data eval/data/sample.jsonl --output-dir eval/tagging/smoke

uv run --project eval python -m unittest discover -s eval/tests -v

# Recompute metrics, revalidate all raw outputs and compare the repeat outputs.
uv run --project eval python eval/tagging/audit.py --compare eval/tagging/quick/results.json
```

`uv` selects Python 3.13, leaving the Mac's Python 3.9 untouched. Dependencies are locked in `uv.lock`. First candidate use downloads about 3–4 GB of model weights; subsequent runs use `eval/.cache/huggingface/`. Once weights are cached, `--offline` makes candidate runs independent of the network; omit it for a first-time download. On this Mac uv/Metal initialization requires execution outside the Codex sandbox; an ordinary terminal has no such restriction.

To generate new data with `OPENAI_API_KEY` set:

```sh
uv run --project eval python eval/data/generate.py --n 300
```

The default generator is `gpt-4.1-mini-2025-04-14`; the blind judge is `gpt-4o-2024-08-06`. Changing model/seed/prompt/schema produces new content-addressed cache keys and can incur new calls. Secrets are read only from the environment and never included in cache files. `--offline` requires no credentials and never creates an API client. Dated API models and fixed seeds help; the saved responses, rather than the API seed alone, make data reconstruction reproducible.

The generator varies topic, speaker style, four length ranges from 5 to 150 words, and 20% mixed/near-neutral cases. It creates one memo per request with a word-count regex in the schema, then submits just transcript and taxonomy/rubric to the different judge model. Target/judge agreement becomes `clean`; disagreement becomes `hard` with the target and judge labels both retained. Normalized word-trigram Jaccard >0.8 or an exact normalized match removes near-duplicates before judging. No disagreement is deleted or relabeled. `--n` is a minimum: if clean quadrant counts are outside 10% min/max balance, it adds examples to deficient quadrants, retaining their disagreements too.

The first batched pilot is retained under `data/pilot/` and excluded from the final comparison. It exposed poor instruction/length adherence; the one-memo strategy and stronger arousal rubric improve agreement. The final word-count regex also introduces occasional joined-word artifacts: `quality_flags: ["long_letter_run"]` identifies 25+ consecutive letters, all retained. The report includes clean accuracy without those examples as a sensitivity check. Fluency and ASR realism remain imperfect; future generation should use unconstrained length repair. All API requests and raw responses, including pilot calls and failed count checks, are in `data/cache/`.

Use `--models lfm2-1.2b qwen3-1.7b`, `--split clean`, `--limit N`, or a custom `--candidates` YAML to scope a run. Each candidate runs in a fresh process, uses the same `tagging/prompt.txt` with its native chat template, disables Qwen thinking, and greedily decodes through a mandatory JSON-schema logits mask. There is no unconstrained fallback or post-hoc JSON repair. `jsonschema` independently checks the raw response; invalid outputs count as incorrect. Two separate fixed-memo warmups are excluded from timings. Output directories must stay under `eval/`; an explicit missing input fails instead of silently switching datasets.

`tagging/report.md` explains accuracy, hard-label sensitivity, timing, memory and limitations. Clean macro top-1 drives ranking. `tagging/results.json` contains per-row raw JSON, predictions, schema errors, latency, decode throughput, first yielded token, constraints-call count and memory, plus exact model revisions and environment. Timestamped `runs/` preserve configs and checkpoints even if a later run overwrites the top-level report. A failed candidate is explicitly failed and excluded from ranking; the process returns nonzero. GPU/process failures remain in partial results; rerun the candidate rather than pooling partial timings.

Same-vendor model agreement is only a consistency check: shared errors remain possible, and there is no defensible numeric independence discount without human labels. This synthetic set cannot establish real-memo accuracy, semantic title quality, ASR robustness or the ~2-second iPhone 14 Pro target. Hand-label a separate real holdout before choosing a production model.

See `VALIDATION.md` for the commands actually run and the verification record. The current fixed-prompt baseline has weak dominant-tag accuracy; the ranking is a reproducible starting point for prompt development and real holdout testing.

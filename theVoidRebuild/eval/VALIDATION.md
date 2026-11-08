# Tagging evaluation validation record

Completed November 7, 2026 (America/Los_Angeles) on an Apple M1 Max, 64 GiB RAM, macOS 26.5.2. The MLX/API pipeline ran with uv and Python 3.13.8. Model inference used the GPU. The Python 3.9 system interpreter was used only for small JSON/file audits, never for MLX or the API pipeline.

Commands below omit repeated cache-directory setup for readability. Actual runs used `eval/.cache/uv`; no source or artifacts were written outside `eval/`. The existing `eval/transcription/` work was not modified.

| Run | Result |
|---|---|
| `uv sync --project eval` | Installed the eval-local environment; full dependency graph in `uv.lock`. |
| `uv run --project eval python eval/data/generate.py --n 300` | Final one-memo generation strategy made 326 generator calls and 326 blind-judge calls. 272 clean / 54 hard; 83.4% agreement; clean min/max balance 91.7%; 65 mixed/near-neutral requests. |
| `uv run --project eval python eval/data/generate.py --n 300 --offline` | Reconstructed the same final input set using 652 cache hits and **zero API calls**. |
| `uv run --project eval python eval/tagging/run.py --data eval/data/sample.jsonl --limit 10 --models lfm2-1.2b --output-dir eval/tagging/smoke-lfm2` | 10 handwritten smoke cases passed. |
| `uv run --project eval python eval/tagging/run.py --data eval/data/sample.jsonl --limit 10 --output-dir eval/tagging/smoke` | All four candidates completed all 10 handwritten cases with valid constrained JSON. LFM2 raw outputs matched its earlier smoke run. |
| `uv run --project eval python eval/tagging/run.py` | **1,304 measured inferences**: four models × 326 memos, plus two unmeasured warmups per model. All outputs schema-valid; no inference errors or truncations. Full run: `20261108T014510.925806Z`. |
| `uv run --project eval python -m unittest discover -s eval/tests -v` | **12 tests passed, zero skips**. Covers scoring denominators, secondary tags, schema boundaries, duplicate keys, blind-judge inputs, seeded generation, dedupe, API-cache isolation, actual MLX logits masking, and every quadrant/schema boundary with all four real tokenizers. |
| `uv run --project eval python eval/tagging/run.py --limit 10 --offline --output-dir eval/tagging/quick` | All four candidates completed with model-network access disabled. Explicit `--limit 10` path verified against generated data. Quick run: `20261108T021451.581668Z`. |
| `uv run --project eval python eval/tagging/audit.py --compare eval/tagging/quick/results.json` | Revalidated every one of the 1,304 raw outputs, checked input/prompt hashes, recomputed all metrics exactly, confirmed constraints were invoked on every inference, and matched **all 40 repeated raw outputs exactly**. |

The tokenizer test initially skipped because Hugging Face's cache-completeness check expected files deliberately omitted by the benchmark, such as README.md. It now uses the identical model-file patterns and explicit cache directory as the runner; all four tokenizers were subsequently checked successfully. No decoding fallback or output repair was introduced.

The full ranking uses clean macro top-1, then clean top-1, schema validity and latency. The results are specific to one shared frozen zero-shot prompt:

| Rank | Model, 4-bit | Clean macro top-1 | Clean top-1 | All top-1 | Median / p95 seconds | Median decode tok/s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Liquid LFM2 1.2B | 50.3% | 50.4% | 50.6% | 0.341 / 0.493 | 159.5 |
| 2 | Qwen2.5 1.5B Instruct | 48.9% | 48.2% | 50.0% | 0.703 / 0.804 | 61.0 |
| 3 | Qwen3 1.7B | 48.7% | 48.2% | 49.1% | 0.630 / 0.775 | 80.3 |
| 4 | Llama 3.2 1B Instruct | 33.0% | 32.7% | 31.6% | 0.397 / 0.485 | 124.8 |

All four achieved 100% schema validity, including the ≤60-character title and 1–2 enum tags. Duplicate tags are allowed by the requested schema and are counted separately in the report. Title content quality has no validated score. LFM2 almost always emitted two tags; any-tag accuracy is therefore substantially more generous than dominant-tag accuracy.

The clearest failure is low-unpleasant dominant tagging: clean recall was 0.0%, 0.0%, 6.1%, and 0.0% respectively. The mask-reachability tests confirm this quadrant is available for all models. The baseline does not support an automatic production model choice. Prompt calibration on a separate development set and a human-labeled real holdout are the next engineering steps.

Data quality and independence remain limited. The generator was `gpt-4.1-mini-2025-04-14`, and the blind judge was `gpt-4o-2024-08-06`; no Anthropic key was available. This is cross-model consistency within one vendor, with unknown correlated errors. There is no defensible numerical discount translating 83.4% agreement into accuracy or cross-vendor confidence.

The first batched pilot is retained under `data/pilot/`: 387 examples, 60.7% agreement and 233 length-bucket misses. The final strategy improved agreement and nominal word counts, but its regex introduced joined-word artifacts. All 21 examples with 25+ consecutive letters remain flagged and retained. Excluding the 17 flagged clean examples leaves 255 clean cases: top-1 becomes 48.6%, 46.7%, 46.7%, and 32.5% in ranking order. The main conclusion remains weak tag reliability. More subtle artifacts and repetitive phrasing are not comprehensively audited.

`data/api-usage.json` counts 1,090 saved successful responses across pilot, repairs and final generation/judging. It records token usage, not inferred billing or SDK retry attempts. Every saved response is under `data/cache/`. `tagging/results.audit.json` records the final audit. Exact model commits, prompt, schema, warmups, row-level raw outputs/timings and environment are preserved in the timestamped run directory; dependencies and model revisions are pinned.

Unverified: real-memo accuracy, human label agreement, semantic title quality, a different vendor's judgments, multilingual/real-ASR robustness, cross-machine determinism, repeated thermal/background-load-controlled timing, and iPhone 14 Pro / MLX Swift latency. Mac warm inference excludes model download/load, tokenizer-trie setup, transcript finalization and capture/save costs. No iPhone ~2-second claim is made.

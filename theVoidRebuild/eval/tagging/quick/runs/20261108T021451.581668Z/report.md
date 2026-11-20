# The Void tagging evaluation

Run `20261108T021451.581668Z` · 10 memos · temperature 0

Synthetic screening results on this Mac. These are not real-memo accuracy or iPhone latency measurements.

## Ranking

Ranked by clean-split macro top-1, then clean top-1, schema validity, and median full-output latency. When no clean split exists, uses all selected rows. No prompt tuning on these results.

| Rank | Model (4-bit) | Clean top-1 | All top-1 | Any tag | Schema valid | Median / p95 (s) | Decode tok/s | End-to-end tok/s | MLX peak GB | RSS peak GB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | llama3.2-1b | 75.0% | 70.0% | 70.0% | 100.0% | 0.370 / 0.419 | 132.6 | 67.3 | 1.089 | 1.189 |
| 2 | qwen3-1.7b | 75.0% | 70.0% | 80.0% | 100.0% | 0.704 / 0.759 | 53.2 | 30.8 | 1.555 | 1.471 |
| 3 | qwen2.5-1.5b | 75.0% | 70.0% | 80.0% | 100.0% | 0.739 / 0.789 | 56.5 | 35.6 | 1.427 | 1.326 |
| 4 | lfm2-1.2b | 62.5% | 60.0% | 80.0% | 100.0% | 0.319 / 0.365 | 159.3 | 77.9 | 1.016 | 0.914 |

**Screening leader: `llama3.2-1b`.** Use this as a shortlist for real holdout and iPhone testing. The ranking rule prioritizes tag accuracy; speed and memory remain visible rather than being combined into an arbitrary score.

## Dataset and label reliability

Source: `memos.jsonl`, SHA-256 `686b86dd79b7be725a59570d41c7953115087eb66d2932f023240965334b3a26`.
Selected rows: `7951e8cd9a8e4126f128c2b0b97766f0f67c3d0bccc5767513f8cb9b0c850712`. Splits: `{'clean': 8, 'hard': 2}`.

Generator `gpt-4.1-mini-2025-04-14`; blind judge `gpt-4o-2024-08-06`. Dataset-wide agreement: **83.4%** over 326 retained memos. Mixed/near-neutral requested: 65. Normalized word-trigram Jaccard >0.8 (or exact normalized match) removes near-duplicates before judging; removed 0.

| Generator target | Clean | Hard |
|---|---:|---:|
| high_pleasant | 66 | 28 |
| low_pleasant | 72 | 3 |
| high_unpleasant | 68 | 7 |
| low_unpleasant | 66 | 16 |

Clean balance min/max: 91.7%. Actual transcript lengths: 8–150 whitespace-delimited words; 0 miss their narrower requested length bucket. **21 memos contain 25+ consecutive letters**, flagging possible joined-word artifacts caused by the generation word-count regex. They are retained; the clean-without-artifacts sensitivity score below checks their effect. Zero bucket misses does not imply fluent transcripts.

The judge sees only taxonomy, a fixed rubric and transcript: no target label, generation metadata or model predictions. Separate model snapshots provide a blind consistency check, but both are OpenAI models. The independence loss cannot be quantified from this dataset: shared training, alignment and emotion conventions can create correlated mistakes. Agreement is not human-verified accuracy. A second vendor would reduce one source of correlation, not establish ground truth.

Clean labels are model agreements and likely easier than real memos. Hard labels remain generator targets with unresolved judge disagreement; all-vs-target and hard-vs-judge scores show sensitivity. Disagreements are never relabeled or deleted. Any supplementary generation only balances clean counts and is disclosed in the manifest.

## Accuracy per quadrant and split

Top-1 is the first tag; any-tag accepts either returned tag. Invalid outputs count as wrong. Per-quadrant accuracy is recall on that target quadrant. A second tag can inflate any-tag accuracy; its frequency is reported below.

### llama3.2-1b

| Split | N | Top-1 | Any tag | Macro top-1 | Top-1 Wilson 95% interval |
|---|---:|---:|---:|---:|---|
| all vs target | 10 | 70.0% | 70.0% | 66.7% | 39.7%–89.2% |
| clean | 8 | 75.0% | 75.0% | 66.7% | 40.9%–92.9% |
| hard | 2 | 50.0% | 50.0% | 50.0% | 9.5%–90.5% |
| hard vs judge | 2 | 50.0% | 100.0% | 50.0% | 9.5%–90.5% |
| clean without long letter runs | 7 | 71.4% | 71.4% | 66.7% | 35.9%–91.8% |
| flagged long letter runs | 1 | 100.0% | 100.0% | 100.0% | 20.7%–100.0% |
| mixed vs target | 2 | 50.0% | 50.0% | 50.0% | 9.5%–90.5% |
| non-mixed vs target | 8 | 75.0% | 75.0% | 66.7% | 40.9%–92.9% |

| Target quadrant | All N | All top-1 | All any-tag | Clean top-1 | Hard top-1 |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| low_pleasant | 0 | — | — | — | — |
| high_unpleasant | 3 | 100.0% | 100.0% | 100.0% | — |
| low_unpleasant | 3 | 0.0% | 0.0% | 0.0% | 0.0% |

Confusion matrix, all rows: true quadrant in rows, first predicted tag in columns.

| True \ predicted | high_pleasant | low_pleasant | high_unpleasant | low_unpleasant | invalid |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 4 | 0 | 0 | 0 | 0 |
| low_pleasant | 0 | 0 | 0 | 0 | 0 |
| high_unpleasant | 0 | 0 | 3 | 0 | 0 |
| low_unpleasant | 0 | 0 | 3 | 0 | 0 |

Second-tag rate: 100.0%; duplicate-tag outputs: 0; blank titles: 0; median title length: 14.5 characters. Completed 10/10; truncated 0; inference errors 0.

Model revision `08231374eeacb049a0eade7922910865b8fce912`. Model load 0.574 s; tokenizer trie setup 0.867 s; download/cache lookup 0.001 s. These are excluded from warm latency. Median first yielded token 0.174 s.

### qwen3-1.7b

| Split | N | Top-1 | Any tag | Macro top-1 | Top-1 Wilson 95% interval |
|---|---:|---:|---:|---:|---|
| all vs target | 10 | 70.0% | 80.0% | 66.7% | 39.7%–89.2% |
| clean | 8 | 75.0% | 87.5% | 66.7% | 40.9%–92.9% |
| hard | 2 | 50.0% | 50.0% | 50.0% | 9.5%–90.5% |
| hard vs judge | 2 | 50.0% | 50.0% | 50.0% | 9.5%–90.5% |
| clean without long letter runs | 7 | 71.4% | 85.7% | 66.7% | 35.9%–91.8% |
| flagged long letter runs | 1 | 100.0% | 100.0% | 100.0% | 20.7%–100.0% |
| mixed vs target | 2 | 50.0% | 50.0% | 50.0% | 9.5%–90.5% |
| non-mixed vs target | 8 | 75.0% | 87.5% | 66.7% | 40.9%–92.9% |

| Target quadrant | All N | All top-1 | All any-tag | Clean top-1 | Hard top-1 |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| low_pleasant | 0 | — | — | — | — |
| high_unpleasant | 3 | 100.0% | 100.0% | 100.0% | — |
| low_unpleasant | 3 | 0.0% | 33.3% | 0.0% | 0.0% |

Confusion matrix, all rows: true quadrant in rows, first predicted tag in columns.

| True \ predicted | high_pleasant | low_pleasant | high_unpleasant | low_unpleasant | invalid |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 4 | 0 | 0 | 0 | 0 |
| low_pleasant | 0 | 0 | 0 | 0 | 0 |
| high_unpleasant | 0 | 0 | 3 | 0 | 0 |
| low_unpleasant | 0 | 1 | 2 | 0 | 0 |

Second-tag rate: 10.0%; duplicate-tag outputs: 0; blank titles: 0; median title length: 13.5 characters. Completed 10/10; truncated 0; inference errors 0.

Model revision `3b1b1768f8f8cf8351c712464f906e86c2b8269e`. Model load 0.599 s; tokenizer trie setup 1.005 s; download/cache lookup 0.001 s. These are excluded from warm latency. Median first yielded token 0.313 s.

### qwen2.5-1.5b

| Split | N | Top-1 | Any tag | Macro top-1 | Top-1 Wilson 95% interval |
|---|---:|---:|---:|---:|---|
| all vs target | 10 | 70.0% | 80.0% | 66.7% | 39.7%–89.2% |
| clean | 8 | 75.0% | 87.5% | 66.7% | 40.9%–92.9% |
| hard | 2 | 50.0% | 50.0% | 50.0% | 9.5%–90.5% |
| hard vs judge | 2 | 0.0% | 50.0% | 0.0% | 0.0%–65.8% |
| clean without long letter runs | 7 | 71.4% | 85.7% | 66.7% | 35.9%–91.8% |
| flagged long letter runs | 1 | 100.0% | 100.0% | 100.0% | 20.7%–100.0% |
| mixed vs target | 2 | 50.0% | 50.0% | 50.0% | 9.5%–90.5% |
| non-mixed vs target | 8 | 75.0% | 87.5% | 66.7% | 40.9%–92.9% |

| Target quadrant | All N | All top-1 | All any-tag | Clean top-1 | Hard top-1 |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| low_pleasant | 0 | — | — | — | — |
| high_unpleasant | 3 | 100.0% | 100.0% | 100.0% | — |
| low_unpleasant | 3 | 0.0% | 33.3% | 0.0% | 0.0% |

Confusion matrix, all rows: true quadrant in rows, first predicted tag in columns.

| True \ predicted | high_pleasant | low_pleasant | high_unpleasant | low_unpleasant | invalid |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 4 | 0 | 0 | 0 | 0 |
| low_pleasant | 0 | 0 | 0 | 0 | 0 |
| high_unpleasant | 0 | 0 | 3 | 0 | 0 |
| low_unpleasant | 2 | 0 | 1 | 0 | 0 |

Second-tag rate: 80.0%; duplicate-tag outputs: 1; blank titles: 0; median title length: 18.0 characters. Completed 10/10; truncated 0; inference errors 0.

Model revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`. Model load 0.480 s; tokenizer trie setup 1.034 s; download/cache lookup 0.001 s. These are excluded from warm latency. Median first yielded token 0.253 s.

### lfm2-1.2b

| Split | N | Top-1 | Any tag | Macro top-1 | Top-1 Wilson 95% interval |
|---|---:|---:|---:|---:|---|
| all vs target | 10 | 60.0% | 80.0% | 55.6% | 31.3%–83.2% |
| clean | 8 | 62.5% | 87.5% | 55.6% | 30.6%–86.3% |
| hard | 2 | 50.0% | 50.0% | 50.0% | 9.5%–90.5% |
| hard vs judge | 2 | 0.0% | 0.0% | 0.0% | 0.0%–65.8% |
| clean without long letter runs | 7 | 57.1% | 85.7% | 55.6% | 25.0%–84.2% |
| flagged long letter runs | 1 | 100.0% | 100.0% | 100.0% | 20.7%–100.0% |
| mixed vs target | 2 | 50.0% | 100.0% | 50.0% | 9.5%–90.5% |
| non-mixed vs target | 8 | 62.5% | 75.0% | 55.6% | 30.6%–86.3% |

| Target quadrant | All N | All top-1 | All any-tag | Clean top-1 | Hard top-1 |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| low_pleasant | 0 | — | — | — | — |
| high_unpleasant | 3 | 66.7% | 100.0% | 66.7% | — |
| low_unpleasant | 3 | 0.0% | 33.3% | 0.0% | 0.0% |

Confusion matrix, all rows: true quadrant in rows, first predicted tag in columns.

| True \ predicted | high_pleasant | low_pleasant | high_unpleasant | low_unpleasant | invalid |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 4 | 0 | 0 | 0 | 0 |
| low_pleasant | 0 | 0 | 0 | 0 | 0 |
| high_unpleasant | 1 | 0 | 2 | 0 | 0 |
| low_unpleasant | 1 | 2 | 0 | 0 | 0 |

Second-tag rate: 80.0%; duplicate-tag outputs: 2; blank titles: 0; median title length: 19.5 characters. Completed 10/10; truncated 0; inference errors 0.

Model revision `3843e4ad0fcb8b7ed8a050908ac8f0bb5320d1bf`. Model load 0.311 s; tokenizer trie setup 0.516 s; download/cache lookup 0.001 s. These are excluded from warm latency. Median first yielded token 0.150 s.

## Titles and error inspection

Title syntax/length is measured. Semantic title quality has no reference labels or validated score; examples below are inspection material, not a title-quality ranking. Raw outputs for every row are in results.json.

| Model | Memo | Transcript (excerpt) | Title | First tag / target |
|---|---|---|---|---|
| llama3.2-1b | syn-0019 | Okay so today was just nonstop chaos at work right from the start everything going wrong deadlines piling up no time to breathe and yeah just feeling totally swamped and frustrated… | Work Chaos | high_unpleasant / high_unpleasant |
| llama3.2-1b | syn-0028 | okay so today was a thrilling saga of getting my car washed missed my favorite coffee place because of that and then bought expired milk seriously what a jackpot of errands joy | Today's Thrilling Saga | high_unpleasant / low_unpleasant |
| llama3.2-1b | syn-0184 | so I guess I started the painting but it feels kind of…flat,youknow | Painting Blues | high_unpleasant / low_unpleasant |
| qwen3-1.7b | syn-0019 | Okay so today was just nonstop chaos at work right from the start everything going wrong deadlines piling up no time to breathe and yeah just feeling totally swamped and frustrated… | Chaos at Work | high_unpleasant / high_unpleasant |
| qwen3-1.7b | syn-0028 | okay so today was a thrilling saga of getting my car washed missed my favorite coffee place because of that and then bought expired milk seriously what a jackpot of errands joy | Today's Chaos | high_unpleasant / low_unpleasant |
| qwen3-1.7b | syn-0184 | so I guess I started the painting but it feels kind of…flat,youknow | Flat Painting | high_unpleasant / low_unpleasant |
| qwen2.5-1.5b | syn-0019 | Okay so today was just nonstop chaos at work right from the start everything going wrong deadlines piling up no time to breathe and yeah just feeling totally swamped and frustrated… | Chaos and Exhaustion | high_unpleasant / high_unpleasant |
| qwen2.5-1.5b | syn-0028 | okay so today was a thrilling saga of getting my car washed missed my favorite coffee place because of that and then bought expired milk seriously what a jackpot of errands joy | Car Wash Adventure | high_pleasant / low_unpleasant |
| qwen2.5-1.5b | syn-0184 | so I guess I started the painting but it feels kind of…flat,youknow | Creative Block | high_unpleasant / low_unpleasant |
| lfm2-1.2b | syn-0019 | Okay so today was just nonstop chaos at work right from the start everything going wrong deadlines piling up no time to breathe and yeah just feeling totally swamped and frustrated… | Workplace Stress | high_pleasant / high_unpleasant |
| lfm2-1.2b | syn-0028 | okay so today was a thrilling saga of getting my car washed missed my favorite coffee place because of that and then bought expired milk seriously what a jackpot of errands joy | Mixed Positive/Negative | high_pleasant / low_unpleasant |
| lfm2-1.2b | syn-0184 | so I guess I started the painting but it feels kind of…flat,youknow | Unfulfilled Creativity | low_pleasant / low_unpleasant |

## Measurement and reproduction

All candidates: mlx-lm + LM Format Enforcer JSON Schema logits mask; native chat templates; enable_thinking=False; fresh process per candidate; sequential rows; no shared prompt/KV cache; invalid outputs count as errors.

2 constrained warmups per model on a separate fixed memo. Greedy temperature 0, seed 20261107, max 128 generated tokens. A fresh schema parser per memo, shared tokenizer trie per model. Canonical JSON field order and at most two consecutive formatting whitespaces. No unconstrained fallback or output repair; the raw response is checked independently with jsonschema, including title ≤60 characters, exact enum values and 1–2 tags.

Full-output latency includes prompt rendering/tokenization, prefill, decode, Python schema masking and GPU synchronization. It excludes model download/load, tokenizer-trie construction, warmups and result I/O. p95 uses nearest rank. Decode tok/s is the median mlx-lm generation_tps (includes schema-mask overhead, excludes prefill; counts its terminal token). End-to-end tok/s is total generation_tokens / measured full-output seconds. Different tokenizers make tok/s less comparable than full latency.

MLX peak is the maximum per-request MLX allocator peak, reset before each measured request, including resident weights and KV buffers; it is not total process memory. RSS peak is OS process high-water memory for the isolated worker, including loading and trie setup. Do not add these unified-memory figures together. Latency percentiles cover completed outputs; errors and truncations are reported separately and still count as incorrect in accuracy.

Wilson intervals describe finite synthetic-example sampling only; they do not cover label uncertainty, generation bias, paired model differences or hardware variability. One sequential timing pass is not a production latency guarantee. Results are specific to this fixed zero-shot prompt; no few-shot prompting, fine-tuning or prompt-cache optimization was evaluated.

Host: **Apple M1 Max**, 68.7 GB RAM, macOS 26.5.2, Device(gpu, 0); Python 3.13.8.

Dependencies: `mlx==0.32.4`, `mlx-lm==0.31.3`, `lm-format-enforcer==0.11.3`, `transformers==5.22.2`, `huggingface-hub==2.4.0`, `jsonschema==4.26.0`. Full dependency graph is pinned in eval/uv.lock.

Prompt SHA-256 `f8cf113d5534a25dbbc529984f827e05f470e69b2bf86fdd60f5652e2b9a62a3`. Schema, exact model commits, inputs, environment, warmups and row-level timing are saved with each timestamped run.

```sh
# From the repository root
export UV_CACHE_DIR="$PWD/eval/.cache/uv"
uv sync --project eval --frozen
uv run --project eval python eval/data/generate.py --n 300 --offline
uv run --project eval python eval/tagging/run.py --limit 10
uv run --project eval python eval/tagging/run.py
uv run --project eval python eval/tagging/run.py --data eval/data/sample.jsonl --output-dir eval/tagging/smoke
```

## Deviations and remaining checks

- The MLX/API pipeline uses Python 3.13 via uv, rather than system Python 3.9. An eval-local pyproject keeps all changes under eval/, so commands use `--project eval`.
- Different dated OpenAI models replace the unavailable Anthropic generator/judge pairing. The judge is blind, but vendor independence is absent and the resulting confidence loss is unquantified.
- Four 1.2–1.7B candidates include Liquid LFM2; Qwen2.5 1.5B adds another ungated size-matched instruct baseline instead of a >2B model.
- An initial 12-memo-per-request pilot had only 60.7% generator/judge agreement and 233/387 length-bucket misses. It is preserved under `eval/data/pilot/`, including every disagreement, and is excluded from the main ranking. The final generator uses one memo per request with schema-constrained word counts to avoid batch alignment drift. Word constraints introduced joined-word artifacts; conservative quality flags and a sensitivity score expose them. All pilot and final API calls remain cached. Future data generation should prefer length repair over a hard word-count regex.
- Hard labels are unresolved. No human-labeled real holdout, title-fidelity benchmark, real ASR-error distribution, multilingual coverage or iPhone 14 Pro / MLX Swift constrained-decoder timing has been verified.
- The ~2-second app target must be tested on the iPhone, including transcript-finalization and capture-to-save costs; this Mac benchmark excludes those stages.

Implementation references: [mlx-lm](https://github.com/ml-explore/mlx-lm), [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer), [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

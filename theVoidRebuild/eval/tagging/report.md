# The Void tagging evaluation

Run `20261108T014510.925806Z` · 326 memos · temperature 0

Synthetic screening results on this Mac. These are not real-memo accuracy or iPhone latency measurements.

## Ranking

Ranked by clean-split macro top-1, then clean top-1, schema validity, and median full-output latency. When no clean split exists, uses all selected rows. No prompt tuning on these results.

| Rank | Model (4-bit) | Clean top-1 | All top-1 | Any tag | Schema valid | Median / p95 (s) | Decode tok/s | End-to-end tok/s | MLX peak GB | RSS peak GB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | lfm2-1.2b | 50.4% | 50.6% | 71.2% | 100.0% | 0.341 / 0.493 | 159.5 | 73.9 | 1.231 | 0.935 |
| 2 | qwen2.5-1.5b | 48.2% | 50.0% | 54.9% | 100.0% | 0.703 / 0.804 | 61.0 | 37.6 | 1.621 | 1.364 |
| 3 | qwen3-1.7b | 48.2% | 49.1% | 56.7% | 100.0% | 0.630 / 0.775 | 80.3 | 34.1 | 1.703 | 1.507 |
| 4 | llama3.2-1b | 32.7% | 31.6% | 64.1% | 100.0% | 0.397 / 0.485 | 124.8 | 64.2 | 1.226 | 1.255 |

**Screening leader: `lfm2-1.2b`.** Use this as a shortlist for real holdout and iPhone testing. The ranking rule prioritizes tag accuracy; speed and memory remain visible rather than being combined into an arbitrary score.

**Dominant-tag accuracy is weak in this baseline.** Calibrate a prompt on a separate development set and validate a human-labeled real holdout before adopting a production tagger. The small differences at the top of this table do not establish a statistically reliable winner.

## Dataset and label reliability

Source: `memos.jsonl`, SHA-256 `686b86dd79b7be725a59570d41c7953115087eb66d2932f023240965334b3a26`.
Selected rows: `7981cc34ffbbf6bb9623344a7b5e5eb5d9759b4943b761e772c2364873cee15f`. Splits: `{'clean': 272, 'hard': 54}`.

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

### lfm2-1.2b

| Split | N | Top-1 | Any tag | Macro top-1 | Top-1 Wilson 95% interval |
|---|---:|---:|---:|---:|---|
| all vs target | 326 | 50.6% | 71.2% | 49.0% | 45.2%–56.0% |
| clean | 272 | 50.4% | 73.2% | 50.3% | 44.5%–56.3% |
| hard | 54 | 51.9% | 61.1% | 47.9% | 38.9%–64.6% |
| hard vs judge | 54 | 16.7% | 38.9% | 14.4% | 9.0%–28.7% |
| clean without long letter runs | 255 | 48.6% | 72.2% | 50.3% | 42.6%–54.7% |
| flagged long letter runs | 21 | 76.2% | 85.7% | 66.7% | 54.9%–89.4% |
| mixed vs target | 65 | 44.6% | 61.5% | 42.7% | 33.2%–56.7% |
| non-mixed vs target | 261 | 52.1% | 73.6% | 50.6% | 46.1%–58.1% |

| Target quadrant | All N | All top-1 | All any-tag | Clean top-1 | Hard top-1 |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 94 | 94.7% | 96.8% | 100.0% | 82.1% |
| low_pleasant | 75 | 53.3% | 77.3% | 52.8% | 66.7% |
| high_unpleasant | 75 | 48.0% | 81.3% | 48.5% | 42.9% |
| low_unpleasant | 82 | 0.0% | 26.8% | 0.0% | 0.0% |

Confusion matrix, all rows: true quadrant in rows, first predicted tag in columns.

| True \ predicted | high_pleasant | low_pleasant | high_unpleasant | low_unpleasant | invalid |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 89 | 4 | 1 | 0 | 0 |
| low_pleasant | 35 | 40 | 0 | 0 | 0 |
| high_unpleasant | 23 | 16 | 36 | 0 | 0 |
| low_unpleasant | 6 | 71 | 5 | 0 | 0 |

Second-tag rate: 96.3%; duplicate-tag outputs: 46; blank titles: 0; median title length: 20.0 characters. Completed 326/326; truncated 0; inference errors 0.

Model revision `3843e4ad0fcb8b7ed8a050908ac8f0bb5320d1bf`. Model load 0.174 s; tokenizer trie setup 0.365 s; download/cache lookup 0.003 s. These are excluded from warm latency. Median first yielded token 0.161 s.

### qwen2.5-1.5b

| Split | N | Top-1 | Any tag | Macro top-1 | Top-1 Wilson 95% interval |
|---|---:|---:|---:|---:|---|
| all vs target | 326 | 50.0% | 54.9% | 48.2% | 44.6%–55.4% |
| clean | 272 | 48.2% | 52.9% | 48.9% | 42.3%–54.1% |
| hard | 54 | 59.3% | 64.8% | 44.6% | 46.0%–71.3% |
| hard vs judge | 54 | 31.5% | 38.9% | 31.5% | 20.7%–44.7% |
| clean without long letter runs | 255 | 46.7% | 51.4% | 49.2% | 40.6%–52.8% |
| flagged long letter runs | 21 | 66.7% | 71.4% | 63.6% | 45.4%–82.8% |
| mixed vs target | 65 | 47.7% | 56.9% | 45.4% | 36.0%–59.6% |
| non-mixed vs target | 261 | 50.6% | 54.4% | 48.9% | 44.5%–56.6% |

| Target quadrant | All N | All top-1 | All any-tag | Clean top-1 | Hard top-1 |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 94 | 96.8% | 100.0% | 98.5% | 92.9% |
| low_pleasant | 75 | 0.0% | 0.0% | 0.0% | 0.0% |
| high_unpleasant | 75 | 96.0% | 100.0% | 97.1% | 85.7% |
| low_unpleasant | 82 | 0.0% | 12.2% | 0.0% | 0.0% |

Confusion matrix, all rows: true quadrant in rows, first predicted tag in columns.

| True \ predicted | high_pleasant | low_pleasant | high_unpleasant | low_unpleasant | invalid |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 91 | 0 | 3 | 0 | 0 |
| low_pleasant | 75 | 0 | 0 | 0 | 0 |
| high_unpleasant | 3 | 0 | 72 | 0 | 0 |
| low_unpleasant | 11 | 0 | 71 | 0 | 0 |

Second-tag rate: 96.6%; duplicate-tag outputs: 6; blank titles: 0; median title length: 17.0 characters. Completed 326/326; truncated 0; inference errors 0.

Model revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`. Model load 0.483 s; tokenizer trie setup 1.037 s; download/cache lookup 0.005 s. These are excluded from warm latency. Median first yielded token 0.250 s.

### qwen3-1.7b

| Split | N | Top-1 | Any tag | Macro top-1 | Top-1 Wilson 95% interval |
|---|---:|---:|---:|---:|---|
| all vs target | 326 | 49.1% | 56.7% | 47.2% | 43.7%–54.5% |
| clean | 272 | 48.2% | 56.6% | 48.7% | 42.3%–54.1% |
| hard | 54 | 53.7% | 57.4% | 36.6% | 40.6%–66.3% |
| hard vs judge | 54 | 16.7% | 16.7% | 18.1% | 9.0%–28.7% |
| clean without long letter runs | 255 | 46.7% | 55.7% | 48.9% | 40.6%–52.8% |
| flagged long letter runs | 21 | 66.7% | 66.7% | 63.6% | 45.4%–82.8% |
| mixed vs target | 65 | 49.2% | 55.4% | 47.0% | 37.5%–61.1% |
| non-mixed vs target | 261 | 49.0% | 57.1% | 47.2% | 43.0%–55.1% |

| Target quadrant | All N | All top-1 | All any-tag | Clean top-1 | Hard top-1 |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 94 | 95.7% | 95.7% | 98.5% | 89.3% |
| low_pleasant | 75 | 16.0% | 16.0% | 16.7% | 0.0% |
| high_unpleasant | 75 | 72.0% | 81.3% | 73.5% | 57.1% |
| low_unpleasant | 82 | 4.9% | 26.8% | 6.1% | 0.0% |

Confusion matrix, all rows: true quadrant in rows, first predicted tag in columns.

| True \ predicted | high_pleasant | low_pleasant | high_unpleasant | low_unpleasant | invalid |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 90 | 0 | 4 | 0 | 0 |
| low_pleasant | 62 | 12 | 1 | 0 | 0 |
| high_unpleasant | 3 | 17 | 54 | 1 | 0 |
| low_unpleasant | 0 | 50 | 28 | 4 | 0 |

Second-tag rate: 15.0%; duplicate-tag outputs: 0; blank titles: 0; median title length: 17.0 characters. Completed 326/326; truncated 0; inference errors 0.

Model revision `3b1b1768f8f8cf8351c712464f906e86c2b8269e`. Model load 0.546 s; tokenizer trie setup 0.905 s; download/cache lookup 0.004 s. These are excluded from warm latency. Median first yielded token 0.297 s.

### llama3.2-1b

| Split | N | Top-1 | Any tag | Macro top-1 | Top-1 Wilson 95% interval |
|---|---:|---:|---:|---:|---|
| all vs target | 326 | 31.6% | 64.1% | 32.4% | 26.8%–36.8% |
| clean | 272 | 32.7% | 64.7% | 33.0% | 27.4%–38.5% |
| hard | 54 | 25.9% | 61.1% | 31.2% | 16.1%–38.9% |
| hard vs judge | 54 | 33.3% | 42.6% | 33.3% | 22.2%–46.6% |
| clean without long letter runs | 255 | 32.5% | 63.5% | 33.4% | 27.1%–38.5% |
| flagged long letter runs | 21 | 28.6% | 85.7% | 39.4% | 13.8%–50.0% |
| mixed vs target | 65 | 29.2% | 67.7% | 29.9% | 19.6%–41.2% |
| non-mixed vs target | 261 | 32.2% | 63.2% | 33.0% | 26.8%–38.1% |

| Target quadrant | All N | All top-1 | All any-tag | Clean top-1 | Hard top-1 |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 94 | 30.9% | 83.0% | 33.3% | 25.0% |
| low_pleasant | 75 | 0.0% | 74.7% | 0.0% | 0.0% |
| high_unpleasant | 75 | 98.7% | 100.0% | 98.5% | 100.0% |
| low_unpleasant | 82 | 0.0% | 0.0% | 0.0% | 0.0% |

Confusion matrix, all rows: true quadrant in rows, first predicted tag in columns.

| True \ predicted | high_pleasant | low_pleasant | high_unpleasant | low_unpleasant | invalid |
|---|---:|---:|---:|---:|---:|
| high_pleasant | 29 | 0 | 65 | 0 | 0 |
| low_pleasant | 15 | 0 | 60 | 0 | 0 |
| high_unpleasant | 1 | 0 | 74 | 0 | 0 |
| low_unpleasant | 0 | 0 | 82 | 0 | 0 |

Second-tag rate: 100.0%; duplicate-tag outputs: 0; blank titles: 0; median title length: 17.0 characters. Completed 326/326; truncated 0; inference errors 0.

Model revision `08231374eeacb049a0eade7922910865b8fce912`. Model load 0.579 s; tokenizer trie setup 0.868 s; download/cache lookup 0.004 s. These are excluded from warm latency. Median first yielded token 0.180 s.

## Titles and error inspection

Title syntax/length is measured. Semantic title quality has no reference labels or validated score; examples below are inspection material, not a title-quality ranking. Raw outputs for every row are in results.json.

| Model | Memo | Transcript (excerpt) | Title | First tag / target |
|---|---|---|---|---|
| lfm2-1.2b | syn-0019 | Okay so today was just nonstop chaos at work right from the start everything going wrong deadlines piling up no time to breathe and yeah just feeling totally swamped and frustrated… | Workplace Stress | high_pleasant / high_unpleasant |
| lfm2-1.2b | syn-0028 | okay so today was a thrilling saga of getting my car washed missed my favorite coffee place because of that and then bought expired milk seriously what a jackpot of errands joy | Mixed Positive/Negative | high_pleasant / low_unpleasant |
| lfm2-1.2b | syn-0184 | so I guess I started the painting but it feels kind of…flat,youknow | Unfulfilled Creativity | low_pleasant / low_unpleasant |
| qwen2.5-1.5b | syn-0019 | Okay so today was just nonstop chaos at work right from the start everything going wrong deadlines piling up no time to breathe and yeah just feeling totally swamped and frustrated… | Chaos and Exhaustion | high_unpleasant / high_unpleasant |
| qwen2.5-1.5b | syn-0028 | okay so today was a thrilling saga of getting my car washed missed my favorite coffee place because of that and then bought expired milk seriously what a jackpot of errands joy | Car Wash Adventure | high_pleasant / low_unpleasant |
| qwen2.5-1.5b | syn-0184 | so I guess I started the painting but it feels kind of…flat,youknow | Creative Block | high_unpleasant / low_unpleasant |
| qwen3-1.7b | syn-0019 | Okay so today was just nonstop chaos at work right from the start everything going wrong deadlines piling up no time to breathe and yeah just feeling totally swamped and frustrated… | Chaos at Work | high_unpleasant / high_unpleasant |
| qwen3-1.7b | syn-0028 | okay so today was a thrilling saga of getting my car washed missed my favorite coffee place because of that and then bought expired milk seriously what a jackpot of errands joy | Today's Chaos | high_unpleasant / low_unpleasant |
| qwen3-1.7b | syn-0184 | so I guess I started the painting but it feels kind of…flat,youknow | Flat Painting | high_unpleasant / low_unpleasant |
| llama3.2-1b | syn-0019 | Okay so today was just nonstop chaos at work right from the start everything going wrong deadlines piling up no time to breathe and yeah just feeling totally swamped and frustrated… | Work Chaos | high_unpleasant / high_unpleasant |
| llama3.2-1b | syn-0028 | okay so today was a thrilling saga of getting my car washed missed my favorite coffee place because of that and then bought expired milk seriously what a jackpot of errands joy | Today's Thrilling Saga | high_unpleasant / low_unpleasant |
| llama3.2-1b | syn-0184 | so I guess I started the painting but it feels kind of…flat,youknow | Painting Blues | high_unpleasant / low_unpleasant |

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

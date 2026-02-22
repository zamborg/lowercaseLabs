# Liquid Insights: V0 and V1

## V0 (implemented, always on)
Flow:
1. Apple on-device speech recognition creates transcript text.
2. Liquid structured extraction runs on transcript text.
3. Output is normalized to app taxonomy, capped at 4 tags.
4. Existing signal/mood/color logic remains unchanged.
5. If Liquid fails, keyword extraction fallback runs automatically.

Output contract stays stable:
- `APIInsight.moodTags` max 4
- `APIInsight.themes` is set to `[]` by local extraction (no theme extraction in V0/V1 local pipelines)
- `APIInsight.safetyFlags` shape unchanged

Runtime metadata is saved in `APITranscript.providerMetadata`:
- `insight_provider`
- `insight_pipeline`
- `insight_mode_requested`
- `insight_latency_ms`
- `insight_model` (when available)
- `insight_fallback_reason` (when fallback occurs)

## App UX: model preparation gate
- On first completed login/onboarding, app shows a full-screen model preparation page.
- The page displays circular progress (0-100%) while downloading/loading model artifacts.
- A retry button appears if download/load fails.
- In Settings, `Redownload Liquid Model` triggers the same full-screen preparation flow.
- Prepared state is cached in `UserDefaults` key `thevoid.liquidModelPrepared`.

## V1 (scaffolded)
Target flow:
1. Reflection audio is provided directly to multimodal Liquid extraction.
2. Structured extraction returns tags/safety without transcript-first dependency.

Current status:
- V1 extraction method exists in code (`extractInsightV1LiquidAudio` and `extractV1Audio`).
- V1 currently requires WAV mono input (16k recommended).
- Recorder currently stores M4A, so audio conversion or recorder format changes are required before enabling V1 in production.

## Runtime configuration
Optional `UserDefaults` keys used by Liquid runtime:
- `thevoid.liquid.model`
- `thevoid.liquid.quantization`
- `thevoid.liquid.localBundlePath`
- `thevoid.liquid.modelFileURL`
- `thevoid.liquid.modelManifestURL`
- `thevoid.liquid.verboseLogs`
- `thevoid.liquid.sequenceLength`
- `thevoid.liquid.maxOutputTokens`
- `thevoid.liquid.unloadAfterEachGeneration`

Defaults when unset:
- model: `LFM2.5-1.2B-Instruct`
- quantization: `Q5_K_M`
- verbose logs: enabled
- sequence length: `2048`
- max output tokens: `128`
- unload after each generation: enabled

Load order:
1. `thevoid.liquid.localBundlePath` file URL (if set).
2. Bundled `<model>.bundle` resource.
3. Bundled `.gguf` resource matching model/quantization (root bundle or `ModelAssets/`).
4. `thevoid.liquid.modelFileURL` (direct file URL, downloaded once and cached to app support).
5. `thevoid.liquid.modelManifestURL` (manifest URL via `Leap.load(manifestURL:)`).
6. Remote `Leap.load(model:quantization:)` download.

If `thevoid.liquid.modelFileURL` is unset, runtime also tries:
- `https://thevoid-local.fly.dev/models/<model>-<quantization>.gguf`

Remote loading resilience:
- Uses retry with backoff for transient network errors.
- Custom `modelFileURL` downloads are cached at `Application Support/LiquidModelCache`.
- If `Leap.load(model:quantization:)` times out, runtime falls back to explicit `ModelDownloader` download and then local `Leap.load(options:)`.
- Downloader fallback uses `URLSessionConfiguration.leapDefault` with longer request/resource timeouts.
- If load errors indicate missing/corrupt model artifacts (for example `fopen failed for data file`), runtime purges cached model files and redownloads.
- If llama backend still fails to initialize for the configured quantization, runtime tries alternates in order: `Q4_K_M`, `Q4_0`, `Q8_0`.

## Built-in model packaging
To avoid first-run download, ship a model file in app resources.

Recommended local folder:
- `theVoidLocal/theVoid/theVoid/ModelAssets/`

This folder is gitignored by default (except `.gitkeep`) so large model files stay local.

Supported bundle options:
- `LFM2.5-1.2B-Instruct.bundle`
- `.gguf` file, ideally named with model and quantization (for example `LFM2.5-1.2B-Instruct-Q5_K_M.gguf`)

Behavior:
- If a bundled resource is present, runtime loads from the app bundle and skips remote download.
- Initial app gate uses explicit prepare/download before entering main tabs.
- Analysis also starts preload in parallel with transcription to reduce first-use latency.

## Fly-hosted model file setup
If you host a `.gguf` on Fly, set:
- `thevoid.liquid.modelFileURL` = `https://<your-fly-domain>/<path>/model.gguf`

Optional manifest-based setup:
- `thevoid.liquid.modelManifestURL` = `https://<your-fly-domain>/<path>/manifest.json`

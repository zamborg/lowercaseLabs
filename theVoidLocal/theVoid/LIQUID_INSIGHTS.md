# Liquid Insights

## V0 Pipeline

1. On-device transcription creates transcript text.
2. Liquid structured extraction runs locally on transcript text.
3. Output is normalized to the app taxonomy and capped at 4 mood tags.
4. Existing mood/color logic remains unchanged.
5. If Liquid extraction fails, keyword fallback runs automatically.

Output contract:
- `title` is a concise 2-5 word entry heading.
- `APIInsight.moodTags` contains at most 4 tags.
- `APIInsight.themes` is `[]` for local extraction.
- Runtime metadata is saved in `APITranscript.providerMetadata`.

## Runtime Configuration

Current defaults in `LocalReflectionAnalyzer.swift`:
- model: `LFM2.5-1.2B-Instruct`
- quantization: `Q5_K_M`
- sequence length: `1024`
- max output tokens: `256`
- unload after each generation: enabled

Supported developer override keys:
- `thevoid.liquid.localBundlePath`
- `thevoid.liquid.enableCustomModelSources`
- `thevoid.liquid.modelFileURL`
- `thevoid.liquid.modelManifestURL`
- `thevoid.liquid.verboseLogs`
- `thevoid.liquid.sequenceLength`
- `thevoid.liquid.maxOutputTokens`
- `thevoid.liquid.unloadAfterEachGeneration`

## Load Order

1. `thevoid.liquid.localBundlePath` file URL, if set.
2. Bundled `<model>.bundle` resource.
3. Bundled `.gguf` resource matching model/quantization in app resources or `ModelAssets/`.
4. `thevoid.liquid.modelFileURL`, only when custom model sources are enabled.
5. `thevoid.liquid.modelManifestURL`, only when custom model sources are enabled.
6. Leap model download by model/quantization.

There is no default Fly-hosted model endpoint.

## Built-In Model Packaging

To avoid first-run download, ship a model file in app resources.

Recommended local folder:
- `theVoid/theVoid/ModelAssets/`

Supported bundle options:
- `LFM2.5-1.2B-Instruct.bundle`
- `.gguf` file named with model and quantization, for example `LFM2.5-1.2B-Instruct-Q5_K_M.gguf`

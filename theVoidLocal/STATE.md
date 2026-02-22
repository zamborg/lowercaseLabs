# theVoidLocal State

Last updated: 2026-02-19 15:35 PST

## 1) Overall Snapshot

- iOS app has local-first reflection analysis with Liquid structured extraction (V0 transcript pipeline).
- Dot/tag output is constrained to max 4 mood tags.
- Local extraction now uses a global model/quantization tuple in code (easy to swap for experiments).
- Generation output is capped at 256 tokens.
- Backend includes model hosting endpoints (`/models`, `/models/{asset_path}`) for optional hosted GGUF delivery.
- Repo is currently a dirty worktree with active iOS + backend edits in progress.

## 2) iOS App (theVoid) Current Behavior

### Reflection + Analysis Flow

- Recording is local.
- Transcript comes from Apple Speech on-device recognition.
- Insight extraction uses Liquid V0 (transcript -> structured output) when model is available.
- If Liquid extraction fails during analysis, keyword fallback still runs.
- `themes` are intentionally empty (`[]`) for local extraction pipeline.
- Safety flags remain in the same schema.

### Model Gate Behavior

- Analysis is blocked if `liquidModelPrepared == false`.
- In that case user sees warning messaging and no dot-analysis is performed.
- Current user-facing copy: model is required for insights and can be downloaded from Settings.

### Model Prep Screen

- A full-screen model prep view exists with:
  - progress ring
  - status text
  - retry button on error
  - cancel button
- **Current trigger state:** this flow is currently manual (Settings -> `Redownload Liquid Model`, or retry from the prep screen), not automatically launched on onboarding completion.

### Cancel Behavior (Known Limitation)

- Current Cancel action dismisses prep UI and marks local state as canceled.
- Runtime cancellation is requested (`requestStopService`, URLSession cancel), and cancel now queries model status plus removes cached/partial files for the active model tuple.
- Net effect observed: cancel is mostly a UI/state cancel, not a guaranteed transport-level abort.

## 3) Liquid Runtime State

Primary file: `theVoidLocal/theVoid/theVoid/LocalAnalysis.swift`

### Runtime Config

- Global tuple:
  - `LiquidInsightsConfig.modelTuple = ("LFM2.5-1.2B-Instruct", "Q5_K_M")`
- Generation limits:
  - `defaultSequenceLength = 1024`
  - `maxOutputTokens = 256`
- Optional `UserDefaults` overrides still used for:
  - `thevoid.liquid.localBundlePath`
  - `thevoid.liquid.enableCustomModelSources`
  - `thevoid.liquid.modelFileURL`
  - `thevoid.liquid.modelManifestURL`
  - `thevoid.liquid.verboseLogs`
  - `thevoid.liquid.sequenceLength`
  - `thevoid.liquid.maxOutputTokens` (clamped to configured hard cap)
  - `thevoid.liquid.unloadAfterEachGeneration`

### Loading Order

1. `localBundlePath` file URL (if set)
2. Bundled `<model>.bundle`
3. Bundled GGUF match in app resources / `ModelAssets`
4. Custom model file URL (`modelFileURL`)
5. Custom manifest URL (`modelManifestURL`)
6. Remote `Leap.load(model:quantization:)`

### OOM / Stability Mitigations Already Added

- Single-flight generation gate (serializes inference requests).
- Single-flight model loading gate (dedupes concurrent preload/load calls).
- Runner unload-after-generation behavior remains enabled by default.
- Remote retry + downloader fallback + cache repair logic still present.

## 4) Device/Runtime Notes

- Physical iPhone tests have been more reliable than local Mac run path for current model loading behavior.
- Prior llama load errors (`error loading model: vector`, `failed to initialize model and context`) were mitigated by retry/fallback/corruption-recovery paths, but environment variability remains.

## 5) Backend State

Primary files:
- `theVoidLocal/backend/app/main.py`
- `theVoidLocal/backend/app/config.py`

Current relevant state:
- Model hosting endpoints are active:
  - `GET /models`
  - `GET /models/{asset_path:path}`
- Model assets roots:
  - `settings.model_assets_root` (default `/data/model_assets`)
  - fallback `/app/model_assets`
- Inline worker mode and OpenAI-based transcription/insights options remain configurable.

## 6) Known Docs Drift

- `theVoidLocal/theVoid/LIQUID_INSIGHTS.md` has stale config details (it still references `thevoid.liquid.model` / `thevoid.liquid.quantization` and older defaults like 2048/128).
- Current source of truth for runtime config is `LocalAnalysis.swift` (`LiquidInsightsConfig` + runtime defaults).

## 7) Known Blockers / Risks

- Push protection incident reported earlier is still likely relevant:
  - blocked secret commit: `e91e11a7733bcbefa237ba8fde454b8b70df557d`
  - requires history cleanup or GitHub unblock path before clean push to protected branch.
- Cancel does not reliably abort live model download/load.
- `ContentView.swift` remains very large; refactor plan exists in `REFACTOR_VIEWS.md`.

## 8) Suggested Immediate Next Steps

1. Decide desired cancel semantics:
   - `UI cancel only` (document and keep), or
   - `hard cancel` (attempt stronger runner/download teardown and explicit state machine handling).
2. Decide whether model prep should auto-launch after onboarding or remain manual from Settings.
3. Update `LIQUID_INSIGHTS.md` to match current runtime truth and reference this state file.

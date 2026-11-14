import type { Model, ModelChoice } from "./contracts";

/** Resolve a per-turn choice, then a saved default, against this assistant's catalog. */
export function resolveModel(
  models: Model[],
  saved?: ModelChoice,
  override?: ModelChoice,
): ModelChoice {
  const model =
    models.find((m) => m.model === override?.model) ??
    models.find((m) => m.model === saved?.model) ??
    models.find((m) => m.isDefault) ??
    models[0];
  if (!model) return { model: "", effort: "" };
  const preferredEffort =
    override?.model === model.model
      ? override.effort
      : saved?.model === model.model
        ? saved.effort
        : "";
  const allowed = (effort: string) =>
    model.supportedReasoningEfforts.some((e) => e.reasoningEffort === effort);
  return {
    model: model.model,
    effort: allowed(preferredEffort)
      ? preferredEffort
      : allowed(model.defaultReasoningEffort)
        ? model.defaultReasoningEffort
        : "",
  };
}

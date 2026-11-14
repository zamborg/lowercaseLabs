import { createHash } from "node:crypto";
import { mkdirSync, realpathSync, writeFileSync } from "node:fs";
import { join, relative, resolve, sep } from "node:path";
import type { PiLocalSettings } from "../contracts";

export const LOCAL_PI_PROVIDER = "mom-made-local";

export function loopbackBaseUrl(value: string) {
  const message = "Enter a local server URL like http://127.0.0.1:8080/v1.";
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw Error(message);
  }
  if (
    url.protocol !== "http:" ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname) ||
    !["/", "/v1", "/v1/"].includes(url.pathname)
  )
    throw Error(message);
  if (url.hostname === "localhost") url.hostname = "127.0.0.1";
  return url.origin + "/v1";
}

export function localPiDirectory(root: string, settings: PiLocalSettings) {
  const identity = createHash("sha256")
    .update(JSON.stringify([settings.baseUrl, settings.model]))
    .digest("hex");
  return join(root, "pi-local", identity);
}

export function localPiSession(directory: string, session?: string) {
  if (!session) return undefined;
  try {
    const root = realpathSync(resolve(directory, "sessions"));
    const file = realpathSync(session);
    const path = relative(root, file);
    if (path && path !== ".." && !path.startsWith(".." + sep)) return file;
  } catch {}
  return undefined;
}

export function configureLocalPi(directory: string, settings: PiLocalSettings) {
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  writeFileSync(
    join(directory, "models.json"),
    JSON.stringify({
      providers: {
        [LOCAL_PI_PROVIDER]: {
          baseUrl: settings.baseUrl,
          api: "openai-completions",
          apiKey: "local",
          models: [
            {
              id: settings.model,
              name: settings.model === "flint" ? "Flint" : settings.model,
              reasoning: true,
              input: ["text"],
              contextWindow: settings.contextWindow,
              maxTokens: Math.min(
                32768,
                Math.floor(settings.contextWindow / 2),
              ),
              compat: {
                supportsStore: false,
                supportsReasoningEffort: false,
                maxTokensField: "max_tokens",
              },
            },
          ],
        },
      },
    }),
    { mode: 0o600 },
  );
}

export async function checkLocalPi(
  settings: PiLocalSettings,
  request: typeof fetch,
) {
  const baseUrl = loopbackBaseUrl(settings.baseUrl);
  try {
    const response = await request(baseUrl + "/models", {
      redirect: "error",
      headers: { Authorization: "Bearer local" },
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok)
      return "The local server isn't ready. Wait for the model to load and refresh.";
    const list = (await response.json()) as {
      data?: { id?: string; meta?: { n_ctx?: number } }[];
    } | null;
    const model = Array.isArray(list?.data)
      ? list.data.find((item) => item?.id === settings.model)
      : undefined;
    if (!model)
      return "This model isn't listed by the server. Check the model ID in Settings.";
    const context = model.meta?.n_ctx;
    if (
      typeof context === "number" &&
      context > 0 &&
      context < settings.contextWindow
    )
      return `The server has ${context.toLocaleString()} context tokens. Set Context tokens to that value or lower.`;
    return undefined;
  } catch {
    return "Couldn't reach the local model. Start your server and refresh.";
  }
}

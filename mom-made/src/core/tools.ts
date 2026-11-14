/** Version 1 MCP contracts. The stdio server is introduced in milestone 2. */
import { z } from "zod";
const appId = z.string().uuid();
export const ToolInputs = {
  "apps.list": z.object({}).strict(),
  "apps.describe": z.object({ appId }).strict(),
  "apps.logs": z
    .object({ appId, limit: z.number().int().min(1).max(24000).default(12000) })
    .strict(),
  "browser.capture": z
    .object({ appId, navigationId: z.number().int().nonnegative() })
    .strict(),
  "browser.inspect": z
    .object({ appId, navigationId: z.number().int().nonnegative() })
    .strict(),
} as const;
export type ToolName = keyof typeof ToolInputs;
export type ToolFailure = {
  code:
    "UNKNOWN_APP" | "NOT_READY" | "STALE_NAVIGATION" | "DENIED" | "INTERNAL";
  message: string;
  retryable: boolean;
};
export type ToolResult<T> =
  | { version: 1; ok: true; appId?: string; navigationId?: number; data: T }
  | { version: 1; ok: false; error: ToolFailure };
export function checkNavigation(
  expected: number,
  current: number,
): ToolResult<null> {
  return expected === current
    ? { version: 1, ok: true, navigationId: current, data: null }
    : {
        version: 1,
        ok: false,
        error: {
          code: "STALE_NAVIGATION",
          message:
            "The page changed. Inspect or capture it again before acting.",
          retryable: true,
        },
      };
}

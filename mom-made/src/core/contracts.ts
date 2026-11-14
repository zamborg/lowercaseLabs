import { z } from "zod";
export const AppManifest = z.object({
  schemaVersion: z.literal(1),
  id: z.string().uuid(),
  name: z.string().min(1).max(80),
  template: z.literal("react-worker-v1"),
  createdAt: z.string(),
});
export type Manifest = z.infer<typeof AppManifest>;
export type AppRecord = Manifest & {
  path: string;
  open: boolean;
  port?: number;
  threadId?: string;
};
export type Runtime = {
  status: "stopped" | "starting" | "ready" | "stopping" | "failed";
  url?: string;
  error?: string;
  logs: string;
};
export type Message = {
  id: string;
  role: "user" | "assistant" | "activity" | "error";
  text: string;
  appId: string;
  turnId?: string;
};
export type Model = {
  model: string;
  displayName: string;
  isDefault: boolean;
  defaultReasoningEffort: string;
  supportedReasoningEfforts: { reasoningEffort: string; description: string }[];
};
export type Approval = {
  id: number | string;
  method: string;
  params: Record<string, unknown>;
};
export type State = {
  apps: AppRecord[];
  selected?: string;
  messages: Message[];
  runtimes: Record<string, Runtime>;
  busy?: { appId: string; turnId?: string };
  approval?: Approval;
  account: string;
  models: Model[];
  error?: string;
};
export const Command = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("create"),
    name: z.string().trim().min(1).max(80),
  }),
  z.object({
    type: z.enum([
      "select",
      "start",
      "stop",
      "close",
      "reveal",
      "capture",
      "reload",
    ]),
    appId: z.string().uuid(),
  }),
  z.object({
    type: z.literal("send"),
    appId: z.string().uuid(),
    text: z.string().trim().min(1).max(64000),
    model: z.string().max(100),
    effort: z.string().max(50),
  }),
  z.object({
    type: z.literal("answer"),
    id: z.union([z.number(), z.string()]),
    accept: z.boolean(),
  }),
  z.object({ type: z.enum(["interrupt", "connect", "login", "compact"]) }),
  z.object({
    type: z.literal("bounds"),
    x: z.number().int().nonnegative(),
    y: z.number().int().nonnegative(),
    width: z.number().int().nonnegative(),
    height: z.number().int().nonnegative(),
    visible: z.boolean(),
  }),
]);
export type Action = z.infer<typeof Command>;
export type API = {
  command: (c: Action) => Promise<unknown>;
  state: () => Promise<State>;
  subscribe: (cb: (s: State) => void) => () => void;
};

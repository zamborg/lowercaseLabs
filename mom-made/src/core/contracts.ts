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
  sessions?: Record<string, string>;
};
const AgentIdSchema = z.enum(["codex", "claude", "pi", "pi-local"]);
export type AgentId = z.infer<typeof AgentIdSchema>;
const PiLocalSettingsSchema = z.object({
  baseUrl: z.string().trim().min(1).max(200),
  model: z.string().trim().min(1).max(200),
  contextWindow: z.number().int().min(16384).max(262144),
});
export type PiLocalSettings = z.infer<typeof PiLocalSettingsSchema>;
export const defaultPiLocalSettings: PiLocalSettings = {
  baseUrl: "http://127.0.0.1:8080/v1",
  model: "flint",
  contextWindow: 65536,
};
const ModelChoice = z.object({
  model: z.string().max(200),
  effort: z.string().max(50),
});
export type ModelChoice = z.infer<typeof ModelChoice>;
export type ModelDefaults = Partial<Record<AgentId, ModelChoice>>;
export type SettingsSection = "assistants" | "cloudflare" | "admin";
export const WORKSPACE_CHAT_ID = "workspace-chat";
export const CHAT_PANE = {
  initial: 380,
  min: 280,
  max: 900,
  previewMin: 360,
  divider: 9,
};
export type AgentInfo = {
  id: AgentId;
  label: string;
  installed: boolean;
  ready: boolean;
  account?: string;
  detail?: string;
  models: Model[];
  approvals: boolean;
  compact: boolean;
  help: {
    install: string[];
    loginHint?: string;
    needsKey?: boolean;
    note?: string;
  };
};
export type Runtime = {
  status: "stopped" | "starting" | "ready" | "stopping" | "failed";
  url?: string;
  error?: string;
  logs: string;
};
export type ActivityKind =
  "edit" | "command" | "look" | "think" | "note" | "divider";
export type Message = {
  id: string;
  role: "user" | "assistant" | "activity" | "error";
  text: string;
  appId: string;
  turnId?: string;
  kind?: ActivityKind;
  detail?: string;
  image?: string;
};
export type Mark = {
  type: "pen" | "rect" | "ellipse" | "pin";
  points: [number, number][];
};
export type Feedback = {
  id: string;
  appId: string;
  comment: string;
  image: string;
  status: "open" | "addressed" | "resolved";
  turnId?: string;
  createdAt: string;
  marks: Mark[];
  viewport: { width: number; height: number };
  url?: string;
};
export type Publish = {
  appId: string;
  status: "idle" | "working" | "done" | "failed";
  step?: string;
  url?: string;
  error?: string;
  log: string;
  history: { at: string; url?: string; ok: boolean; note: string }[];
};
export type Cloudflare = {
  state: "unknown" | "ready" | "login-required";
  account?: string;
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
  chatWidth: number;
  chatSessions: Partial<Record<AgentId, string>>;
  messages: Message[];
  runtimes: Record<string, Runtime>;
  busy?: { appId: string; turnId?: string };
  approval?: Approval;
  account: string;
  error?: string;
  feedback: Feedback[];
  publish: Record<string, Publish>;
  cloudflare: Cloudflare;
  version: string;
  agent: AgentId;
  agents: AgentInfo[];
  modelDefaults: ModelDefaults;
  openRouterKeySaved: boolean;
  piLocal: PiLocalSettings;
};
export const Command = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("create"),
    name: z.string().trim().min(1).max(80),
  }),
  z.object({
    type: z.enum([
      "select",
      "openApp",
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
    appId: z.union([z.string().uuid(), z.literal(WORKSPACE_CHAT_ID)]),
    agent: AgentIdSchema.optional(),
    text: z.string().trim().min(1).max(64000),
    ...ModelChoice.shape,
  }),
  z.object({
    type: z.literal("answer"),
    id: z.union([z.number(), z.string()]),
    accept: z.boolean(),
    always: z.boolean().optional(),
  }),
  z.object({
    type: z.enum([
      "interrupt",
      "compact",
      "newConversation",
      "showChat",
      "cloudflareLogin",
      "cloudflareCheck",
      "recheck",
      "revealWorkspace",
    ]),
  }),
  z.object({
    type: z.literal("markup"),
    appId: z.string().uuid(),
    agent: AgentIdSchema.optional(),
    comment: z.string().trim().min(1).max(4000),
    image: z.string().max(20_000_000),
    marks: z
      .array(
        z.object({
          type: z.enum(["pen", "rect", "ellipse", "pin"]),
          points: z.array(z.tuple([z.number(), z.number()])).max(2000),
        }),
      )
      .max(200),
    viewport: z.object({
      width: z.number().int().nonnegative(),
      height: z.number().int().nonnegative(),
    }),
    ...ModelChoice.shape,
  }),
  z.object({
    type: z.literal("feedback"),
    id: z.string().uuid(),
    status: z.enum(["open", "resolved"]),
  }),
  z.object({
    type: z.enum(["fix", "publish"]),
    appId: z.string().uuid(),
    agent: AgentIdSchema.optional(),
    model: ModelChoice.shape.model.default(""),
    effort: ModelChoice.shape.effort.default(""),
  }),
  z.object({
    type: z.enum(["agent", "login"]),
    id: AgentIdSchema,
  }),
  z.object({
    type: z.literal("modelDefault"),
    id: AgentIdSchema,
    ...ModelChoice.shape,
  }),
  z.object({
    type: z.literal("agentKey"),
    key: z
      .string()
      .trim()
      .max(400)
      .regex(/^[\x21-\x7e]*$/),
  }),
  z.object({
    type: z.literal("piLocalSettings"),
    settings: PiLocalSettingsSchema,
  }),
  z.object({
    type: z.literal("open"),
    url: z.string().url().max(300),
  }),
  z.object({
    type: z.literal("openChatLink"),
    url: z
      .string()
      .max(8192)
      .pipe(z.url({ protocol: /^https?$/ })),
  }),
  z.object({
    type: z.literal("chatWidth"),
    width: z.number().int().min(CHAT_PANE.min).max(CHAT_PANE.max),
  }),
  z.object({
    type: z.literal("rename"),
    appId: z.string().uuid(),
    name: z.string().trim().min(1).max(80),
  }),
  z.object({
    type: z.literal("bounds"),
    x: z.number().int().nonnegative(),
    y: z.number().int().nonnegative(),
    width: z.number().int().nonnegative(),
    height: z.number().int().nonnegative(),
    visible: z.boolean(),
  }),
  z.object({ type: z.literal("previewVisibility"), visible: z.literal(false) }),
]);
export type Action = z.infer<typeof Command>;
export type API = {
  command: (c: Action) => Promise<unknown>;
  state: () => Promise<State>;
  subscribe: (cb: (s: State) => void) => () => void;
  onSettings: (cb: () => void) => () => void;
};

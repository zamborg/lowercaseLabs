/** Tools the agent can use to look at and drive the app it is building. */
export type ToolSpec = {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
};
const none = { type: "object", properties: {}, additionalProperties: false };
export const TOOLS: ToolSpec[] = [
  {
    name: "app_screenshot",
    description:
      "Take a screenshot of the running app exactly as the user sees it. Use after every visible change.",
    inputSchema: none,
  },
  {
    name: "app_page_text",
    description:
      "Read the visible text and the interactive elements (buttons, links, fields) of the running app, with a navigationId.",
    inputSchema: none,
  },
  {
    name: "app_console",
    description:
      "Recent browser console errors, warnings and failed page loads from the running app.",
    inputSchema: none,
  },
  {
    name: "app_logs",
    description:
      "Recent dev-server output (build errors, Worker errors). Check this if the page is blank.",
    inputSchema: none,
  },
  {
    name: "app_click",
    description:
      "Click a button, link or element in the running app by its visible text or a CSS selector. Pass the navigationId from app_page_text so a stale page is rejected.",
    inputSchema: {
      type: "object",
      properties: {
        text: { type: "string" },
        selector: { type: "string" },
        navigationId: { type: "integer" },
      },
      additionalProperties: false,
    },
  },
  {
    name: "app_type",
    description:
      "Type into a field of the running app, identified by CSS selector or label text. Pass the navigationId from app_page_text.",
    inputSchema: {
      type: "object",
      properties: {
        selector: { type: "string" },
        label: { type: "string" },
        text: { type: "string" },
        navigationId: { type: "integer" },
      },
      required: ["text"],
      additionalProperties: false,
    },
  },
  {
    name: "app_reload",
    description: "Reload the running app's page.",
    inputSchema: none,
  },
];
export type ToolContent =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string };
export type ToolReply = { content: ToolContent[]; isError?: boolean };
export type BridgeRequest = {
  token: string;
  appId: string;
  tool: string;
  args: Record<string, unknown>;
};
export function staleNavigation(expected: unknown, current: number) {
  return typeof expected === "number" && expected !== current;
}

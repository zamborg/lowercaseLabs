// Pi extension (loaded with -e): registers mom-made's tool server for this run.
export default function (pi: {
  registerMcpServer(name: string, cfg: Record<string, unknown>): void;
}) {
  const command = process.env.MOM_MCP_COMMAND;
  if (!command) return;
  pi.registerMcpServer("mom-made", {
    command,
    args: JSON.parse(process.env.MOM_MCP_ARGS ?? "[]"),
    env: JSON.parse(process.env.MOM_MCP_ENV ?? "{}"),
    exposure: "direct",
    ...(process.env.MOM_PI_TEXT_ONLY === "1"
      ? { toolExposure: { app_screenshot: "hidden" } }
      : {}),
  });
}

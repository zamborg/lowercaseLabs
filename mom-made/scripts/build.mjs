import { build } from "esbuild";
import { mkdir, copyFile } from "node:fs/promises";
await mkdir("dist", { recursive: true });
await build({
  entryPoints: {
    main: "src/desktop/main.ts",
    preload: "src/desktop/preload.ts",
    watchdog: "src/core/watchdog.ts",
    mcp: "src/core/mcp.ts",
    wrangler: "src/core/wrangler.ts",
  },
  outdir: "dist",
  bundle: true,
  platform: "node",
  format: "cjs",
  outExtension: { ".js": ".cjs" },
  packages: "external",
});
await build({
  entryPoints: { "pi-mcp": "src/core/agents/pi-mcp.ts" },
  outdir: "dist",
  bundle: true,
  platform: "node",
  format: "esm",
  outExtension: { ".js": ".mjs" },
});
await build({
  entryPoints: ["src/ui/index.tsx"],
  outfile: "dist/ui.js",
  bundle: true,
  platform: "browser",
  format: "esm",
});
await copyFile("src/ui/index.html", "dist/index.html");
await copyFile("src/ui/style.css", "dist/style.css");

import { build } from "esbuild";
import { mkdir, copyFile } from "node:fs/promises";
await mkdir("dist", { recursive: true });
await build({
  entryPoints: {
    main: "src/desktop/main.ts",
    preload: "src/desktop/preload.ts",
    watchdog: "src/core/watchdog.ts",
  },
  outdir: "dist",
  bundle: true,
  platform: "node",
  format: "cjs",
  outExtension: { ".js": ".cjs" },
  packages: "external",
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

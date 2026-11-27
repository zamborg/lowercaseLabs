import { build } from "esbuild";
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";

const output = "../src/zos_excalidraw/static";
const vendor = "node_modules/@excalidraw/excalidraw/dist/prod";
await rm(output, {recursive: true, force: true});
await mkdir(output, {recursive: true});
await cp(vendor + "/fonts", output + "/fonts", {recursive: true});
await cp(vendor + "/index.css", output + "/vendor.css");
await cp("bootstrap.js", output + "/bootstrap.js");
await cp("drawings.js", output + "/drawings.js");
await build({
  entryPoints: ["editor.jsx"], outdir: output, bundle: true, format: "esm", splitting: true,
  minify: true, conditions: ["production"], define: {"process.env.NODE_ENV": '"production"'},
  plugins: [{
    name: "offline-excalidraw",
    setup(builder) {
      builder.onLoad({filter: /[/\\]@excalidraw[/\\]excalidraw[/\\]dist[/\\]prod[/\\].*\.js$/}, async ({path}) => {
        let contents = await readFile(path, "utf8");
        // Upstream appends its CDN even when a local font path is configured.
        contents = contents.replace(/`https:\/\/esm\.sh\/\$\{[^\n]+?\/dist\/prod\/`/g, "window.EXCALIDRAW_ASSET_PATH");
        contents = contents.replace(/https:\/\/(?:[\w.-]+\.)?excalidraw\.com[^"'`\s)]*/g, "#");
        return {contents, loader: "js"};
      });
    }
  }]
});
await writeFile(output + "/editor.css", await readFile("editor.css"));

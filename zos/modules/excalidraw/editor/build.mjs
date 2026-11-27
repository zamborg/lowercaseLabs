import { build } from "esbuild";
import { cp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";

const output = "../src/zos_excalidraw/static";
const vendor = "node_modules/@excalidraw/excalidraw/dist/prod";
await rm(output, {recursive: true, force: true});
await mkdir(output, {recursive: true});
await cp(vendor + "/fonts", output + "/fonts", {recursive: true});
await cp(vendor + "/index.css", output + "/vendor.css");
await cp("bootstrap.js", output + "/bootstrap.js");
await cp("EXCALIDRAW-LICENSE.txt", output + "/EXCALIDRAW-LICENSE.txt");
await cp("drawings.css", output + "/drawings.css");
await cp("drawings.js", output + "/drawings.js");
await build({
  entryPoints: ["editor.jsx"], outdir: output, bundle: true, format: "esm", splitting: true,
  legalComments: "external", minify: true, conditions: ["production"], define: {"process.env.NODE_ENV": '"production"'},
  plugins: [{
    name: "offline-excalidraw",
    setup(builder) {
      builder.onLoad({filter: /[/\\]@excalidraw[/\\]excalidraw[/\\]dist[/\\]prod[/\\].*\.js$/}, async ({path}) => {
        let contents = await readFile(path, "utf8");
        // Upstream appends its CDN even when a local font path is configured.
        contents = contents.replace(/`https:\/\/esm\.sh\/\$\{[^\n]+?\/dist\/prod\/`/g, "window.EXCALIDRAW_ASSET_PATH");
        return {contents, loader: "js"};
      });
    }
  }]
});
await writeFile(output + "/editor.css", await readFile("editor.css"));
// The upstream distribution also embeds online help, integrations and optional
// share destinations. Remove network origins from the offline JS distribution.
// W3C namespace identifiers are required by createElementNS and SVG rendering.
for (const filename of await readdir(output)) {
  if (!filename.endsWith(".js")) continue;
  const path = output + "/" + filename;
  const js = await readFile(path, "utf8");
  const offline = js.replace(/https?:\/\/([\w.-]+)/g, (url, host) =>
    host === "www.w3.org" ? url : "about:blank#offline-" + host);
  if (js !== offline) await writeFile(path, offline);
}

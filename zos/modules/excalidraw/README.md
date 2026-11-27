# Excalidraw for ZOS

Install `excalidraw` from the module catalog. Open `/admin/m/excalidraw/` to create, rename, edit or delete a drawing. The editor autosaves changes. If another editor saves first, autosave pauses and keeps your working canvas open. Reload the saved version when you are ready to discard local changes.

Agents share the same drawings through four tools.

| Tool | Tier | Result |
| --- | --- | --- |
| `excalidraw_list` | read | Drawing IDs, titles, versions, timestamps and last editors |
| `excalidraw_get` | read | Native scene, version, element counts and up to twelve labels |
| `excalidraw_create` | write | New drawing from a title and optional element specs |
| `excalidraw_update` | write | New version after replacing elements or applying a patch |

For example, call `excalidraw_create` with these arguments.

```json
{
  "title": "ZOS drawing flow",
  "elements": [
    {"id":"agent","type":"ellipse","x":20,"y":50,"label":"Agent"},
    {"id":"zos","type":"rectangle","x":320,"y":50,"label":"ZOS"},
    {"id":"call","type":"arrow","start":"agent","end":"zos","label":"MCP"}
  ]
}
```

Shapes support `rectangle`, `ellipse` and `diamond`. Supply an explicit ID to reference a shape from an arrow or patch. Use `text` for standalone text and `label` for bound text. For a free arrow, supply its origin in `x` and `y` and its signed displacement in `width` and `height`. A doubly bound arrow without coordinates connects the two shapes at their boundaries.

Update with `id`, the last observed `if_version`, and either `elements` or a `patch`. For example, `{"add":[{"type":"text","text":"Review","x":40,"y":200}],"remove":["call"]}` removes the arrow and its bound label, then adds a note. On a conflict the tool returns an error with `current_version` and `summary`. Read the drawing again before applying your changes.

Each row in the module's Postgres schema stores a title and a native scene of up to 5 MiB of UTF-8 JSON including embedded images. The owner UI and MCP server use the same version check. The Docker build packages the React editor, fonts and browser assets. Network egress is disabled. The release image is `ghcr.io/zamborg/zos-excalidraw`. Hosted installers accept its digest via `ZOS_IMAGE_EXCALIDRAW`.

## Development checks

From this directory, build the editor and run the tests against a disposable Postgres database specified by `TEST_DATABASE_URL`.

```sh
npm ci --prefix editor
npm run build --prefix editor
uv sync --frozen
uv run playwright install chromium
uv run pytest
uv run ruff check
```

The browser test accepts a local Chromium binary via `CHROMIUM_PATH` and recognizes Google Chrome on macOS. The catalog integration test runs with the core suite in `zos/`.

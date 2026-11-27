import asyncio
import re
from pathlib import Path

import httpx
import pytest

from zos_excalidraw.mcp_server import TOOLS
from zos_excalidraw.scenes import MAX_BYTES, SceneTooLarge, convert, edit_scene, normalize_scene
from zos_excalidraw.store import Conflict
from zos_excalidraw.ui import create_ui_app


def test_simplified_elements_and_patch():
    specs = [
        {
            "id": "connector",
            "type": "arrow",
            "start": "left",
            "end": "right",
            "width": -80,
            "height": 20,
        },
        {"id": "left", "type": "ellipse", "label": "Input"},
        {"id": "right", "type": "diamond"},
        {"type": "text", "text": "Notes"},
        {"type": "rectangle"},
    ]
    scene = normalize_scene({"elements": convert(specs)})
    arrow, left, label, right, *_ = scene["elements"]
    assert arrow["points"] == [[0, 0], [-80, 20]]
    assert arrow["startBinding"]["elementId"] == left["id"]
    assert arrow["endBinding"]["elementId"] == right["id"]
    assert label["containerId"] == left["id"]
    assert {"type": "text", "id": label["id"]} in left["boundElements"]
    assert {"type": "arrow", "id": "connector"} in left["boundElements"]
    assert all(isinstance(e["versionNonce"], int) and "seed" in e for e in scene["elements"])
    edited = edit_scene(scene, {"patch": {"remove": ["left"]}})
    kept = edited["elements"]
    assert kept[0]["startBinding"] is None and kept[0]["endBinding"]["elementId"] == "right"
    assert label["id"] not in {e["id"] for e in kept}


@pytest.mark.parametrize(
    "elements",
    [
        [{"type": "arrow", "start": "missing"}],
        [{"type": "rectangle", "id": "a"}, {"type": "text", "id": "a"}],
        [{"type": "diamond", "width": -1}],
        [{"type": "text", "x": float("nan")}],
    ],
)
def test_invalid_spec(elements):
    with pytest.raises(ValueError):
        normalize_scene({"elements": convert(elements)})


def test_scene_cap_includes_files_and_keeps_only_document_app_state():
    with pytest.raises(SceneTooLarge, match="5 MiB"):
        normalize_scene({"elements": [], "files": {"attachment": {"dataURL": "a" * MAX_BYTES}}})
    scene = normalize_scene(
        {"elements": [], "appState": {"collaborators": {}, "viewBackgroundColor": "#fff"}}
    )
    assert scene["appState"] == {"viewBackgroundColor": "#fff"}


def test_tool_hints():
    assert {tool.name: tool.annotations.read_only_hint for tool in TOOLS} == {
        "list": True,
        "get": True,
        "create": False,
        "update": False,
    }


async def test_competing_updates_and_stale_delete(store):
    await store.init_schema()
    d = await store.create("Original", "owner", [])

    async def edit(actor):
        return await store.change(
            d["id"], 1, actor, action="update", data={"elements": [{"type": "text", "text": actor}]}
        )

    results = await asyncio.gather(edit("A"), edit("B"), return_exceptions=True)
    (stale,) = [r for r in results if isinstance(r, Conflict)]
    current = await store.get(d["id"])
    assert stale.detail == {
        "error": "Version conflict",
        "current_version": 2,
        "summary": current["summary"],
    }
    assert current["summary"]["labels"] == [current["updated_by"]]
    with pytest.raises(Conflict):
        await store.change(d["id"], 1, "owner", action="delete", data=None)
    await store.change(d["id"], 2, "owner", action="rename", data="New")
    assert (await store.list())[0]["title"] == "New"
    await store.change(d["id"], 3, "owner", action="delete", data=None)
    assert await store.list() == []


async def test_signed_ui_csrf_csp_and_autosave(store, owner_headers):
    d = await store.create("Drawing", "agent", [])
    api = "/api/" + d["id"]
    app = create_ui_app(store, module_secret="fixture")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for route in ["/", "/d/" + d["id"], api, "/assets/bootstrap.js"]:
            assert (await client.get(route)).status_code == 403
        client.headers.update(owner_headers)
        page = await client.get("/d/" + d["id"])
        for directive in ["default-src 'self'", "connect-src 'self'", "img-src 'self' data: blob:"]:
            assert directive in page.headers["Content-Security-Policy"]
        assert not re.search(r"https?://", page.text)
        args = {"if_version": 1, "scene": {"elements": [], "files": {}, "appState": {}}}
        assert (await client.post(api, json=args)).status_code == 403
        client.headers["X-CSRF-Token"] = client.cookies["excalidraw_csrf"]
        assert (await client.post(api, json=args)).json()["version"] == 2
        assert (await client.post(api, json=args)).status_code == 409
        args["scene"]["files"] = {"large": {"dataURL": "x" * MAX_BYTES}}
        assert (await client.post(api, json={**args, "if_version": 2})).status_code == 413
        assert (await store.get(d["id"]))["version"] == 2


def test_built_assets_are_local():
    static = Path(__file__).resolve().parents[1] / "src/zos_excalidraw/static"
    assert (static / "fonts/Virgil/Virgil-Regular.woff2").is_file(), (
        "Run npm ci and npm run build in editor/"
    )
    for script in static.glob("*.js"):
        # DOM namespaces are identifiers; they must remain exact for SVG and MathML.
        external = set(re.findall(r"https?://([\w.-]+)", script.read_text())) - {"www.w3.org"}
        assert not external, (script, external)
    css = (static / "vendor.css").read_text()
    for font in re.findall(r'url\("\./(fonts/[^"]+)"\)', css):
        assert (static / font).is_file()


def test_auto_positions_arrow_between_bound_shapes():
    left = {"id": "left", "type": "rectangle", "x": 30, "y": 80, "width": 100, "height": 40}
    right = {**left, "id": "right", "type": "ellipse", "x": 240}
    arrow, *_ = convert([{"type": "arrow", "start": "left", "end": "right"}, left, right])
    assert (arrow["x"], arrow["y"]) == (131, 100)
    assert arrow["points"] == [[0, 0], [108, 0]]

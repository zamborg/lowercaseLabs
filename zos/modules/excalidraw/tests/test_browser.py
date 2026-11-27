import asyncio
import os
import socket
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from playwright.async_api import async_playwright, expect
from starlette.applications import Starlette
from starlette.routing import Mount

from zos_excalidraw.ui import create_ui_app


async def test_editor_draws_saves_and_preserves_changes_on_conflict(store, owner_headers):
    prefix = "/admin/m/excalidraw"
    app = Starlette(routes=[Mount(prefix, create_ui_app(store, module_secret="fixture"))])
    drawing = await store.create(
        "Blueprint", "agent", [{"type": "rectangle", "id": "box", "label": "From MCP"}]
    )
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                await asyncio.sleep(0.01)
            async with async_playwright() as pw:
                chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
                executable = os.getenv("CHROMIUM_PATH") or (
                    str(chrome) if chrome.exists() else None
                )
                browser = await pw.chromium.launch(executable_path=executable)
                try:
                    page = await browser.new_page(
                        extra_http_headers={**owner_headers, "X-Forwarded-Prefix": prefix}
                    )
                    external = []
                    errors = []

                    async def local_only(route):
                        url = urlsplit(route.request.url)
                        if url.scheme in {"http", "https"} and url.netloc != f"127.0.0.1:{port}":
                            external.append(route.request.url)
                            await route.abort()
                        else:
                            await route.continue_()

                    await page.route("**/*", local_only)
                    page.on("pageerror", lambda err: errors.append(str(err)))
                    await page.add_init_script("""window.cspBlocks = [];
                        document.addEventListener("securitypolicyviolation", e => window.cspBlocks.push(e.blockedURI));""")
                    await page.goto(f"http://127.0.0.1:{port}{prefix}/")
                    await page.get_by_role("link", name="Blueprint").click()
                    canvas = page.locator("canvas.interactive")
                    await expect(canvas).to_be_visible(timeout=15000)
                    assert (
                        await page.evaluate("window.EXCALIDRAW_ASSET_PATH")
                        == f"http://127.0.0.1:{port}{prefix}/assets/"
                    )
                    await expect(page.locator(".default-sidebar-trigger")).to_be_hidden()
                    await expect(page.locator("#status")).to_have_text("Saved")

                    # Allow the initial font load and autosave debounce to settle.
                    await page.evaluate(
                        "document.fonts.ready.then(() => new Promise(r => setTimeout(r, 600)))"
                    )
                    assert (await store.get(drawing["id"]))["version"] == 1

                    async def rectangle(x):
                        await page.keyboard.press("r")
                        box = await canvas.bounding_box()
                        assert box is not None
                        y = box["y"] + 110
                        await page.mouse.move(x, y)
                        await page.mouse.down()
                        await page.mouse.move(x + 95, y + 55, steps=5)
                        await page.mouse.up()

                    async with page.expect_response(
                        lambda res: res.request.method == "POST" and "/api/" in res.url
                    ) as event:
                        await rectangle(240)
                    assert (await event.value).status == 200
                    await expect(page.locator("#status")).to_have_text("Saved")
                    current = await store.get(drawing["id"])
                    assert current["updated_by"] == "owner"
                    assert current["summary"]["counts"] == {"rectangle": 2, "text": 1}
                    await store.change(
                        drawing["id"],
                        current["version"],
                        "agent",
                        action="update",
                        data={"patch": {"add": [{"type": "diamond", "label": "Agent edit"}]}},
                    )
                    await rectangle(410)
                    await expect(page.locator("#reload")).to_be_visible()
                    await expect(page.locator("#status")).to_contain_text(
                        "Your edits are still on this canvas"
                    )
                    assert (await store.get(drawing["id"]))["summary"]["counts"]["rectangle"] == 2
                    await page.set_viewport_size({"width": 390, "height": 844})
                    await expect(canvas).to_be_visible()
                    await page.wait_for_function(
                        """() => {
                            const el = document.querySelector("canvas.interactive");
                            if (!el) return false;
                            const rect = el.getBoundingClientRect();
                            return rect.width > 0 && rect.width <= 390;
                        }""",
                        timeout=5000,
                    )
                    mobile_box = await canvas.bounding_box()
                    assert mobile_box is not None and 0 < mobile_box["width"] <= 390
                    assert await page.evaluate("window.cspBlocks") == []
                    assert external == [] and errors == []
                finally:
                    await browser.close()
        finally:
            server.should_exit = True
            await task

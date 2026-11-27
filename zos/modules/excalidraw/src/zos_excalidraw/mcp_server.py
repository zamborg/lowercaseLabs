import contextlib
import json

import mcp_types as types
from jsonschema import ValidationError, validate
from mcp.server.lowlevel.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from .caller import CallerVerificationError, verify_mcp_caller
from .store import Conflict


def obj(properties, *required):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


STRING = {"type": "string"}
NUMBER = {"type": "number"}
SPEC = obj(
    {
        "type": {"enum": ["rectangle", "ellipse", "diamond", "text", "arrow"]},
        "id": STRING,
        "x": NUMBER,
        "y": NUMBER,
        "width": NUMBER,
        "height": NUMBER,
        "label": STRING,
        "text": STRING,
        "start": STRING,
        "end": STRING,
    },
    "type",
)
ELEMENTS = {"type": "array", "items": SPEC}
PATCH = obj({"remove": {"type": "array", "items": STRING}, "add": ELEMENTS})
PATCH["minProperties"] = 1
SPEC_HELP = (
    "Elements accept type (rectangle, ellipse, diamond, text or arrow), optional unique id, "
    "x/y, width/height and label (text for standalone text). Arrow x/y is its start and "
    "width/height are signed deltas. Bind start/end to shape IDs, including later shapes. "
    "Omit arrow coordinates with two bindings to connect shape boundaries automatically."
)
DEFINITIONS = {
    "list": ("List shared drawings with titles, versions and last editors.", obj({})),
    "get": (
        "Read a drawing with the full native scene, version and compact element summary.",
        obj({"id": STRING}, "id"),
    ),
    "create": (
        "Create a shared drawing. " + SPEC_HELP,
        obj({"title": STRING, "elements": ELEMENTS}, "title"),
    ),
    "update": (
        "Use if_version from get. Supply elements to replace all elements or patch to remove IDs "
        "and append new specs. On conflict re-read and retry. " + SPEC_HELP,
        obj(
            {
                "id": STRING,
                "if_version": {"type": "integer", "minimum": 1},
                "patch": PATCH,
                "elements": ELEMENTS,
            },
            "id",
            "if_version",
        ),
    ),
}
TOOLS = [
    types.Tool(
        name=name,
        description=desc,
        input_schema=schema,
        annotations=types.ToolAnnotations(
            read_only_hint=name in {"list", "get"},
            destructive_hint=name == "update",
            open_world_hint=False,
        ),
    )
    for name, (desc, schema) in DEFINITIONS.items()
]


async def dispatch(store, secret, params):
    name, args = params.name, dict(params.arguments or {})
    failed = False
    try:
        caller = verify_mcp_caller(params.meta, secret=secret, tool=name, args=args)
        if not caller or caller.client_name.casefold() == "owner":
            raise CallerVerificationError("A signed non-owner MCP caller is required")
        if name not in DEFINITIONS:
            raise ValueError("Unknown tool")
        validate(args, DEFINITIONS[name][1])
        if name == "list":
            data = {"drawings": await store.list()}
        elif name == "get":
            data = await store.get(args["id"])
        elif name == "create":
            data = await store.create(args["title"], caller.client_name, args.get("elements", []))
        else:
            data = await store.change(
                args["id"], args["if_version"], caller.client_name, action="update", data=args
            )
    except Conflict as exc:
        failed, data = True, exc.detail
    except ValidationError as exc:
        failed, data = True, {"error": "Invalid input at " + exc.json_path}
    except (ValueError, KeyError, TypeError) as exc:
        failed, data = True, {"error": str(exc)}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data))], is_error=failed
    )


def create_mcp_app(store, *, module_secret, mcp_path="/mcp"):
    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=TOOLS)

    async def call_tool(ctx, params):
        return await dispatch(store, module_secret, params)

    server = Server(
        "zos-excalidraw", version="0.1.0", on_list_tools=list_tools, on_call_tool=call_tool
    )
    manager = StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=True,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    async def health(request):
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[
            Route("/healthz", health),
            Route(mcp_path, StreamableHTTPASGIApp(manager), methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lifespan,
    )

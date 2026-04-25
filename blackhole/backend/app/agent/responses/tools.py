from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ResponseTool:
    name: str
    definition: dict[str, Any]
    handler: Callable[..., Any] | None = None


_TOOLS: dict[str, ResponseTool] = {}


def register_tool(tool: ResponseTool) -> None:
    _TOOLS[tool.name] = tool


def get_tool_definitions(names: list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    if names is None:
        return [tool.definition for tool in _TOOLS.values()]
    return [tool.definition for name in names if (tool := _TOOLS.get(name))]


def get_tool(name: str) -> ResponseTool | None:
    return _TOOLS.get(name)

"""Translate the MCP drawing vocabulary to native Excalidraw scene elements."""

import json
import math
import secrets
import time
from collections import Counter
from uuid import uuid4

SHAPES = {"rectangle", "ellipse", "diamond"}
MAX_BYTES = 5 * 1024 * 1024
APP_STATE = {"viewBackgroundColor", "gridSize", "gridStep", "gridModeEnabled"}


class SceneTooLarge(ValueError):
    pass


def normalize_scene(scene):
    if not isinstance(scene, dict):
        raise ValueError("Expected a scene object")
    elements = scene.get("elements")
    state = scene.get("appState", {})
    files = scene.get("files", {})
    if not isinstance(elements, list) or not isinstance(state, dict) or not isinstance(files, dict):
        raise ValueError("Expected elements array, appState object and files object")
    ids = set()
    for e in elements:
        if not isinstance(e, dict) or not isinstance(e.get("type"), str):
            raise ValueError("Expected native elements with id and type")
        ident = e.get("id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("Element IDs must be unique non-empty strings")
        ids.add(ident)
    scene = {"elements": elements, "appState": {k: v for k, v in state.items() if k in APP_STATE}, "files": files}
    if len(json.dumps(scene, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()) > MAX_BYTES:
        raise SceneTooLarge("Scene exceeds 5 MiB including embedded files")
    return scene


def summary(scene):
    counts = Counter(e["type"] for e in scene["elements"] if not e.get("isDeleted"))
    labels = [e["text"][:60] for e in scene["elements"] if not e.get("isDeleted") and isinstance(e.get("text"), str)]
    return {"counts": dict(counts), "labels": labels[:12]}


def number(spec, key, default):
    value = spec.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 10_000_000:
        raise ValueError(f"{key} must be a finite canvas coordinate within ±10000000")
    return value


def native(kind, ident, x, y, width, height):
    return dict(type=kind, id=ident, x=x, y=y, width=width, height=height,
                angle=0, strokeColor="#1b1b1f", backgroundColor="transparent", fillStyle="solid",
                strokeWidth=2, strokeStyle="solid", roughness=1, opacity=100, groupIds=[],
                frameId=None, roundness=None, seed=secrets.randbits(30), version=1,
                versionNonce=secrets.randbits(30), isDeleted=False, boundElements=[],
                updated=int(time.time()*1000), link=None, locked=False)


def make_text(ident, label, x, y, container=None):
    lines = label.split("\n")
    width = max(1, max(map(len, lines))) * 11
    height = len(lines) * 25
    e = native("text", ident, x, y, width, height)
    e.update(text=label, originalText=label, fontSize=20, fontFamily=1,
             textAlign="center" if container else "left", verticalAlign="middle" if container else "top",
             containerId=container, autoResize=True, lineHeight=1.25)
    return e


def convert(specs, existing=()):
    if not isinstance(specs, list):
        raise ValueError("elements must be an array")
    result = list(existing)
    by_id = {e["id"]: e for e in result if not e.get("isDeleted")}
    arrows = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Expected element specification")
        kind = spec.get("type")
        ident = spec.get("id", str(uuid4()))
        if not isinstance(ident, str) or not ident:
            raise ValueError("Expected a non-empty element ID")
        x, y = number(spec, "x", 0), number(spec, "y", 0)
        w, h = number(spec, "width", 160), number(spec, "height", 80)
        label = spec.get("text", spec.get("label", ""))
        if not isinstance(label, str):
            raise ValueError("Expected string label")
        if kind == "text":
            e = make_text(ident, label, x, y)
        elif kind in SHAPES:
            if min(w, h) <= 0:
                raise ValueError("Shapes need positive width and height")
            e = native(kind, ident, x, y, w, h)
        elif kind == "arrow":
            e = native(kind, ident, x, y, abs(w), abs(h))
            e.update(points=[[0, 0], [w, h]], startBinding=None, endBinding=None,
                     startArrowhead=None, endArrowhead="arrow", elbowed=False, lastCommittedPoint=None)
            arrows.append((e, spec))
        else:
            raise ValueError(f"Unsupported element type: {kind}")
        result.append(e)
        by_id[ident] = e
        if label and kind != "text":
            text = make_text(str(uuid4()), label, x, y, ident)
            text.update(x=x+(w-text["width"])/2, y=y+(h-text["height"])/2)
            result.append(text)
            e["boundElements"].append({"type": "text", "id": text["id"]})
    for arrow, spec in arrows:
        for side in ("start", "end"):
            if side not in spec:
                continue
            target = by_id.get(spec[side])
            if target is None or target["type"] not in SHAPES:
                raise ValueError(f"{side} must reference a shape ID in the drawing")
            arrow[side+"Binding"] = {"elementId": target["id"], "focus": 0, "gap": 1, "fixedPoint": None}
            binding = {"id": arrow["id"], "type": "arrow"}
            if binding not in (target.get("boundElements") or []):
                target["boundElements"] = [*(target.get("boundElements") or []), binding]
    return result


def edit_scene(scene, args):
    if ("elements" in args) == ("patch" in args):
        raise ValueError("Supply exactly one of elements or patch")
    if "elements" in args:
        return normalize_scene({**scene, "elements": convert(args["elements"])})
    patch = args["patch"]
    # Patches remove named elements (and their labels) and append new specs.
    remove = set(patch.get("remove", []))
    old = scene["elements"]
    if remove - {e["id"] for e in old}:
        raise ValueError("Unknown element ID in patch.remove")
    remove.update(e["id"] for e in old if e.get("containerId") in remove)
    kept = [dict(e) for e in old if e["id"] not in remove]
    for e in kept:
        if e.get("boundElements"):
            e["boundElements"] = [b for b in e["boundElements"] if b["id"] not in remove]
        for key in ("startBinding", "endBinding"):
            if (e.get(key) or {}).get("elementId") in remove:
                e[key] = None
        if e.get("frameId") in remove:
            e["frameId"] = None
    return normalize_scene({**scene, "elements": convert(patch.get("add", []), kept)})

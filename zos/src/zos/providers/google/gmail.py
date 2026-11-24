"""Gmail upstream ToolProvider (`gmail_search`, `gmail_read`, `gmail_send`)."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from email.message import EmailMessage
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote
from uuid import UUID

import mcp_types as types

from zos.auth.context import CallContext
from zos.providers.base import ProviderKind, ProviderTool
from zos.providers.google.client import GoogleServiceHub

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
MAX_SEARCH_LIMIT = 25
METADATA_FETCH_CONCURRENCY = 5
MAX_READ_BODY_CHARS = 16 * 1024


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in ("script", "style"):
            self._skip_depth += 1
        elif tag.lower() in ("p", "br", "div", "li", "tr"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def strip_html(html_text: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html_text)
    parser.close()
    return parser.get_text()


def _decode_b64url(data: str) -> str:
    padded = data + "=" * ((-len(data)) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", errors="replace")


def _headers_map(payload: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for h in payload.get("headers", []):
        if isinstance(h, dict) and h.get("name"):
            out[str(h["name"]).lower()] = str(h.get("value", ""))
    return out


def _extract_body_from_payload(payload: dict[str, Any]) -> str:
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def walk(part: dict[str, Any]) -> None:
        mime = str(part.get("mimeType", "")).lower()
        body = part.get("body") or {}
        data = body.get("data")
        if data and mime == "text/plain":
            plain_parts.append(_decode_b64url(data))
        elif data and mime == "text/html":
            html_parts.append(_decode_b64url(data))
        for sub in part.get("parts") or []:
            if isinstance(sub, dict):
                walk(sub)

    walk(payload)
    if plain_parts:
        return "\n".join(plain_parts).strip()
    if html_parts:
        return strip_html("\n".join(html_parts))
    return ""


class GmailProvider:
    """Upstream Gmail REST API provider sharing the principal's Google credential."""

    backend = "api"

    def __init__(self, hub: GoogleServiceHub) -> None:
        self.hub = hub

    @property
    def namespace(self) -> str:
        return "gmail"

    @property
    def kind(self) -> ProviderKind:
        return "upstream"

    @property
    def provider_id(self) -> UUID | None:
        return self.hub.gmail_provider_id

    def is_online(self) -> bool:
        return self.hub.is_connected

    async def list_tools(self) -> list[ProviderTool]:
        return [
            ProviderTool(
                name="search",
                description="Search Gmail messages and return metadata (from, to, subject, date, snippet, ids).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Gmail search query."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_LIMIT, "default": 10},
                    },
                    "required": ["query"],
                },
                destructive_hint=False,
                idempotent_hint=True,
            ),
            ProviderTool(
                name="read",
                description="Read a Gmail message by ID (prefers text/plain, strips HTML fallback, capped).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "string"},
                    },
                    "required": ["message_id"],
                },
                destructive_hint=False,
                idempotent_hint=True,
            ),
            ProviderTool(
                name="send",
                description="Send an email via Gmail (supports threaded replies via reply_to_message_id).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                        "reply_to_message_id": {"type": "string", "description": "Optional Gmail message ID to reply to."},
                    },
                    "required": ["to", "subject", "body"],
                },
                destructive_hint=False,
            ),
        ]

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        ctx: CallContext,
    ) -> types.CallToolResult:
        try:
            if name == "search":
                return await self._search(args)
            if name == "read":
                return await self._read(args)
            if name == "send":
                return await self._send(args)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Unknown gmail tool: {name}")],
                is_error=True,
            )
        except Exception as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"gmail_{name} error: {exc}")],
                is_error=True,
            )

    async def _search(self, args: dict[str, Any]) -> types.CallToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")
        limit = max(1, min(int(args.get("limit", 10)), MAX_SEARCH_LIMIT))

        listed = await self.hub.api_request(
            "GET",
            f"{GMAIL_API_BASE}/messages",
            params={"q": query, "maxResults": limit},
        )
        refs = listed.get("messages") or []
        sem = asyncio.Semaphore(METADATA_FETCH_CONCURRENCY)

        async def fetch_one(ref: dict[str, Any]) -> dict[str, Any]:
            mid = quote(str(ref["id"]), safe="")
            async with sem:
                msg = await self.hub.api_request(
                    "GET",
                    f"{GMAIL_API_BASE}/messages/{mid}",
                    params=[
                        ("format", "metadata"),
                        ("metadataHeaders", "From"),
                        ("metadataHeaders", "To"),
                        ("metadataHeaders", "Subject"),
                        ("metadataHeaders", "Date"),
                    ],
                )
            hdrs = _headers_map(msg.get("payload") or {})
            return {
                "id": msg.get("id"),
                "thread_id": msg.get("threadId"),
                "from": hdrs.get("from", ""),
                "to": hdrs.get("to", ""),
                "subject": hdrs.get("subject", ""),
                "date": hdrs.get("date", ""),
                "snippet": msg.get("snippet", ""),
            }

        items = await asyncio.gather(*(fetch_one(r) for r in refs[:limit]))
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=json.dumps({"count": len(items), "messages": list(items)}, indent=2),
                )
            ]
        )

    async def _read(self, args: dict[str, Any]) -> types.CallToolResult:
        message_id = str(args.get("message_id", "")).strip()
        if not message_id:
            raise ValueError("message_id is required")
        mid = quote(message_id, safe="")

        msg = await self.hub.api_request(
            "GET",
            f"{GMAIL_API_BASE}/messages/{mid}",
            params={"format": "full"},
        )
        payload = msg.get("payload") or {}
        hdrs = _headers_map(payload)
        body = _extract_body_from_payload(payload)
        truncated = len(body) > MAX_READ_BODY_CHARS
        if truncated:
            body = body[:MAX_READ_BODY_CHARS] + "\n...[truncated]..."

        out = {
            "id": msg.get("id"),
            "thread_id": msg.get("threadId"),
            "from": hdrs.get("from", ""),
            "to": hdrs.get("to", ""),
            "subject": hdrs.get("subject", ""),
            "date": hdrs.get("date", ""),
            "message_id_header": hdrs.get("message-id", ""),
            "references": hdrs.get("references", ""),
            "body": body,
            "truncated": truncated,
        }
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(out, indent=2))]
        )

    async def _send(self, args: dict[str, Any]) -> types.CallToolResult:
        to = str(args.get("to", "")).strip()
        subject = str(args.get("subject", "")).strip()
        body = str(args.get("body", ""))
        reply_to_id = str(args.get("reply_to_message_id") or "").strip()

        if not to or not subject:
            raise ValueError("'to' and 'subject' are required")

        mime = EmailMessage()
        mime["To"] = to
        mime["Subject"] = subject
        mime.set_content(body)

        send_payload: dict[str, Any] = {}
        if reply_to_id:
            mid = quote(reply_to_id, safe="")
            orig = await self.hub.api_request(
                "GET",
                f"{GMAIL_API_BASE}/messages/{mid}",
                params=[
                    ("format", "metadata"),
                    ("metadataHeaders", "Message-ID"),
                    ("metadataHeaders", "References"),
                    ("metadataHeaders", "Subject"),
                ],
            )
            orig_hdrs = _headers_map(orig.get("payload") or {})
            orig_mid = orig_hdrs.get("message-id", "").strip()
            orig_refs = orig_hdrs.get("references", "").strip()
            if orig_mid:
                mime["In-Reply-To"] = orig_mid
                refs = f"{orig_refs} {orig_mid}".strip() if orig_refs else orig_mid
                mime["References"] = refs
            if orig.get("threadId"):
                send_payload["threadId"] = orig["threadId"]

        raw_b64 = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")
        send_payload["raw"] = raw_b64

        sent = await self.hub.api_request(
            "POST",
            f"{GMAIL_API_BASE}/messages/send",
            json_body=send_payload,
        )
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "sent": True,
                            "id": sent.get("id"),
                            "thread_id": sent.get("threadId"),
                        },
                        indent=2,
                    ),
                )
            ]
        )

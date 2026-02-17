from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Static

from services.tui.api_client import ApiClient


class PromptScreen(ModalScreen[Optional[str]]):
    def __init__(self, prompt: str, placeholder: str = "") -> None:
        super().__init__()
        self._prompt = prompt
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        yield Container(
            Static(self._prompt, id="prompt-text"),
            Input(placeholder=self._placeholder),
            Static("Enter to submit, Esc to cancel", id="prompt-hint"),
            id="prompt-container",
        )

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


class DraftScreen(ModalScreen[str]):
    def __init__(self, draft_text: str) -> None:
        super().__init__()
        self._draft_text = draft_text

    def compose(self) -> ComposeResult:
        yield Container(
            Static(self._draft_text, id="draft-body"),
            Static("y send  e edit  c cancel", id="draft-hint"),
            id="draft-container",
        )

    def on_key(self, event) -> None:
        key = event.key.lower()
        if key in {"y", "e", "c"}:
            self.dismiss(key)


class MessageScreen(ModalScreen[None]):
    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        yield Container(
            Static(self._title, id="message-title"),
            Static(self._body, id="message-body"),
            Static("Press any key to close", id="message-hint"),
            id="message-container",
        )

    def on_key(self, event) -> None:
        self.dismiss(None)


class MailTUI(App):
    CSS = """
    #prompt-container, #draft-container, #message-container {
        border: solid $accent;
        padding: 1 2;
        width: 80%;
        height: auto;
        margin: 1 2;
        background: $surface;
    }
    #draft-body {
        height: auto;
    }
    #message-body {
        height: auto;
    }
    """

    BINDINGS = [
        Binding("j", "next", "Next"),
        Binding("k", "prev", "Prev"),
        Binding("a", "archive", "Archive"),
        Binding("d", "delete", "Delete"),
        Binding("s", "snooze", "Snooze"),
        Binding("r", "remember", "Remember"),
        Binding("g", "delegate", "Delegate"),
        Binding("n", "notes", "Notes"),
        Binding("R", "refresh", "Refresh"),
        Binding("enter", "open_thread", "Thread"),
        Binding("?", "help", "Help"),
        Binding("!", "break_glass", "Break glass"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self.emails: list[dict[str, Any]] = []
        self.index = 0
        self.body = Static(id="email-body")
        self.status = Static(id="status")

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.body
        yield self.status
        yield Footer()

    async def on_mount(self) -> None:
        ready = await self.api.wait_until_ready()
        if not ready:
            self.body.update("API not reachable. Start `docker compose up` and press R to retry.")
            self.status.update("offline")
            return
        await self.refresh_inbox()

    async def on_unmount(self) -> None:
        await self.api.close()

    async def refresh_inbox(self) -> None:
        try:
            self.emails = await self.api.fetch_inbox()
        except httpx.RequestError:
            self.body.update("API not reachable. Start `docker compose up` and press R to retry.")
            self.status.update("offline")
            self.emails = []
            return
        self.index = 0
        await self.render_email()

    async def render_email(self) -> None:
        if not self.emails:
            self.body.update("No emails in this view.")
            self.status.update("")
            return
        email = self.emails[self.index]
        from_list = email.get("from", [])
        sender = from_list[0].get("email") if from_list else "unknown"
        subject = email.get("subject") or "(no subject)"
        snippet = email.get("snippet") or ""
        loop_status = email.get("loop_status")
        suggestion = email.get("category_suggestion")
        suggestion_text = (
            f"Suggested: {suggestion['name']} ({suggestion['confidence']:.2f})"
            if suggestion
            else ""
        )

        self.body.update(
            f"{subject}\nFrom: {sender}\n\n{snippet}\n\n{suggestion_text}"
        )
        self.status.update(f"{self.index + 1}/{len(self.emails)}  [{loop_status}]")

    async def action_next(self) -> None:
        if not self.emails:
            return
        self.index = min(self.index + 1, len(self.emails) - 1)
        await self.render_email()

    async def action_prev(self) -> None:
        if not self.emails:
            return
        self.index = max(self.index - 1, 0)
        await self.render_email()

    async def _act_and_advance(self, handler) -> None:
        if not self.emails:
            return
        email = self.emails[self.index]
        await handler(email["id"])
        await self.refresh_inbox()

    async def action_archive(self) -> None:
        await self._act_and_advance(self.api.archive)

    async def action_delete(self) -> None:
        await self._act_and_advance(self.api.delete)

    async def action_snooze(self) -> None:
        if not self.emails:
            return
        days_text = await self.push_screen(PromptScreen("Snooze days (e.g., 1, 2, 7)", "2"))
        if days_text is None:
            return
        try:
            days = int(days_text.strip())
        except ValueError:
            await self.push_screen(MessageScreen("Invalid input", "Enter a number of days."))
            return
        until = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
        email = self.emails[self.index]
        await self.api.snooze(email["id"], until, f"snooze_{days}d")
        await self.refresh_inbox()

    async def action_remember(self) -> None:
        if not self.emails:
            return
        why = await self.push_screen(PromptScreen("Why remember this? (optional)", ""))
        email = self.emails[self.index]
        await self.api.remember(email["id"], why, None)
        await self.refresh_inbox()

    async def action_delegate(self) -> None:
        if not self.emails:
            return
        instruction = await self.push_screen(
            PromptScreen("Tell your assistant what to do", "Draft a friendly reply")
        )
        if not instruction:
            return
        email = self.emails[self.index]
        job_id = await self.api.delegate(email["id"], instruction, None)

        draft = await self._wait_for_draft(job_id)
        if not draft:
            await self.push_screen(MessageScreen("Draft failed", "No draft available."))
            return

        action = await self.push_screen(DraftScreen(draft))
        if action == "y":
            await self.api.send_draft(job_id)
            await self.refresh_inbox()
        elif action == "e":
            await self.action_delegate()
        else:
            return

    async def _wait_for_draft(self, job_id: str) -> Optional[str]:
        for _ in range(20):
            draft = await self.api.get_draft(job_id)
            if draft.get("status") == "succeeded" and draft.get("draft_text"):
                return draft["draft_text"]
            if draft.get("status") == "failed":
                return None
            await asyncio.sleep(0.5)
        return None

    async def action_open_thread(self) -> None:
        if not self.emails:
            return
        email = self.emails[self.index]
        detail = await self.api.get_email(email["id"])
        thread = detail.get("thread_context", [])
        lines = ["Thread context:\n"]
        for msg in thread:
            lines.append(f"- {msg.get('date')}: {msg.get('subject')} ({msg.get('id')})")
        await self.push_screen(MessageScreen("Thread", "\n".join(lines)))

    async def action_notes(self) -> None:
        notes = await self.api.list_notes()
        if not notes:
            await self.push_screen(MessageScreen("Notes", "No notes documents yet."))
            return
        doc = notes[0]
        content = await self.api.get_notes(doc["id"])
        await self.push_screen(MessageScreen(content.get("title", "Notes"), content.get("content_markdown", "")))

    async def action_help(self) -> None:
        help_text = (
            "j/k next/prev | a archive | d delete | s snooze | r remember | g delegate\n"
            "n notes | R refresh | enter thread | ! break glass"
        )
        await self.push_screen(MessageScreen("Help", help_text))

    async def action_refresh(self) -> None:
        await self.refresh_inbox()

    async def action_break_glass(self) -> None:
        await self.push_screen(
            MessageScreen(
                "Break glass",
                "Break-glass compose is intentionally annoying. Use delegate unless necessary.",
            )
        )


if __name__ == "__main__":
    MailTUI().run()

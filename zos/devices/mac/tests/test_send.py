"""messages_send: handle resolution, AppleScript argv safety, and chat.db confirmation.

All sends go to a mocked osascript runner; nothing ever reaches Messages.app.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence

import pytest

from conftest import JARED_ID, MacFixture, insert_outgoing
from zos_mac.chatdb import AmbiguousRecipientError, ChatStore
from zos_mac.send import (
    MAX_SEND_CHARS,
    SEND_SCRIPT_LINES,
    MessageSender,
    build_osascript_argv,
)

NASTY_TEXT = 'He said "hi" \\ end\nline2\r\n\t— ☕️ 東京 \'q\' $(rm -rf ~) `x` & end tell'


def _store(fx: MacFixture) -> ChatStore:
    return ChatStore(fx.chat_db, fx.contacts)


def _data_args(argv: Sequence[str]) -> list[str]:
    """Strip `osascript` and the fixed `-e <line>` pairs, leaving only argv data items."""
    rest = list(argv[1:])
    while rest and rest[0] == "-e":
        rest = rest[2:]
    return rest


class FakeRunner:
    def __init__(self, fx: MacFixture | None = None, *, record_chat: int | None = None, results=None) -> None:
        self.fx = fx
        self.record_chat = record_chat
        self.results = list(results or [])
        self.calls: list[list[str]] = []

    async def __call__(self, argv: Sequence[str]) -> tuple[int, str, str]:
        self.calls.append(list(argv))
        rc, out, err = self.results.pop(0) if self.results else (0, "", "")
        if rc == 0 and self.fx is not None and self.record_chat is not None:
            insert_outgoing(self.fx.chat_db, self.record_chat, _data_args(argv)[2])
        return rc, out, err


# ---- resolution -------------------------------------------------------------


def test_resolve_single_handle_phone_email_and_contact(mac_fixture: MacFixture) -> None:
    store = _store(mac_fixture)
    t = store.resolve_send_target("+1 (415) 555-0103")
    assert (t.kind, t.value, t.contact_name) == ("handle", "+14155550103", "Monica Hall")

    monica = store.resolve_send_target("SRC-A:2")  # contact id, single handle
    assert (monica.kind, monica.value) == ("handle", "+14155550103")

    email = store.resolve_send_target("team@piedpiper.example")
    assert (email.kind, email.value, email.contact_name) == ("handle", "team@piedpiper.example", "Pied Piper")

    unknown = store.resolve_send_target("+1 212 555 0199")  # not in chat.db: cleaned digits
    assert unknown.value == "+12125550199"
    assert unknown.contact_name is None


def test_resolve_contact_prefers_most_recent_conversation(mac_fixture: MacFixture) -> None:
    store = _store(mac_fixture)
    jared = store.resolve_send_target(JARED_ID)
    assert jared.value == "+14155550101"  # mobile chat newer than work SMS chat; groups ignored
    richard = store.resolve_send_target("SRC-A:5")
    assert richard.value == "richard@example.com"  # email chat newer than phone chat


def test_resolve_ambiguous_contact_lists_options(mac_fixture: MacFixture) -> None:
    with pytest.raises(AmbiguousRecipientError) as exc_info:
        _store(mac_fixture).resolve_send_target("SRC-A:3")  # Gavin: two phones, no history
    opts = exc_info.value.options
    assert [(o["value"], o["label"]) for o in opts] == [("+14155550104", "mobile"), ("+14155550105", "iPhone")]
    assert all(o["last_conversation_at"] is None for o in opts)


def test_resolve_chat_ids_and_errors(mac_fixture: MacFixture) -> None:
    store = _store(mac_fixture)
    g = store.resolve_send_target("iMessage;+;chat1001")
    assert (g.kind, g.value, g.chat_id) == ("chat", "iMessage;+;chat1001", "iMessage;+;chat1001")
    by_ident = store.resolve_send_target("chat1002")
    assert (by_ident.kind, by_ident.value) == ("chat", "iMessage;+;chat1002")
    direct = store.resolve_send_target("iMessage;-;+14155550101")
    assert direct.kind == "chat"
    with pytest.raises(LookupError):
        store.resolve_send_target("iMessage;+;chat9999")
    with pytest.raises(LookupError):
        store.resolve_send_target("Jared")  # names must go through contacts_search
    with pytest.raises(ValueError):
        store.resolve_send_target("   ")


# ---- AppleScript argv safety -------------------------------------------------


def test_osascript_argv_never_interpolates_text() -> None:
    argv = build_osascript_argv(SEND_SCRIPT_LINES, ["handle", "+14155550103", NASTY_TEXT, "iMessage"])
    assert argv[0] == "osascript"
    script = [argv[i + 1] for i in range(1, len(argv) - 4, 2)]
    assert all(argv[i] == "-e" for i in range(1, len(argv) - 4, 2))
    assert tuple(script) == SEND_SCRIPT_LINES
    assert argv[-4:] == ["handle", "+14155550103", NASTY_TEXT, "iMessage"]
    assert not any(NASTY_TEXT in line or "rm -rf" in line for line in script)
    assert SEND_SCRIPT_LINES[0] == "on run argv"
    with pytest.raises(ValueError):
        build_osascript_argv(SEND_SCRIPT_LINES, ["-x"])


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("osascript"), reason="needs macOS osascript")
def test_osascript_argv_roundtrip_real_interpreter() -> None:
    """Echo-only AppleScript (no app access) proves argv survives quotes, newlines, unicode."""
    echo = ("on run argv", "return item 2 of argv", "end run")
    argv = build_osascript_argv(echo, ["handle", NASTY_TEXT])
    proc = subprocess.run(argv, capture_output=True, timeout=20, check=False)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.decode("utf-8")
    assert out.removesuffix("\n").replace("\r\n", "\n").replace("\r", "\n") == NASTY_TEXT.replace(
        "\r\n", "\n"
    )


# ---- send flow ---------------------------------------------------------------


async def test_send_confirmed_via_chat_db(mac_fixture: MacFixture) -> None:
    runner = FakeRunner(mac_fixture, record_chat=7)
    sender = MessageSender(_store(mac_fixture), runner=runner, confirm_timeout_s=2, poll_interval_s=0.05)
    res = await sender.send("SRC-A:2", NASTY_TEXT)
    assert res["status"] == "sent"
    assert res["chat_id"] == "iMessage;-;+14155550103"
    assert res["contact_name"] == "Monica Hall"
    assert isinstance(res["message_id"], int)
    assert len(runner.calls) == 1
    assert _data_args(runner.calls[0]) == ["handle", "+14155550103", NASTY_TEXT, "iMessage"]


async def test_send_group_by_chat_id(mac_fixture: MacFixture) -> None:
    runner = FakeRunner(mac_fixture, record_chat=3)
    sender = MessageSender(_store(mac_fixture), runner=runner, confirm_timeout_s=2, poll_interval_s=0.05)
    res = await sender.send("iMessage;+;chat1001", "hi team")
    assert res["status"] == "sent"
    assert res["chat_id"] == "iMessage;+;chat1001"
    assert _data_args(runner.calls[0])[:3] == ["chat", "iMessage;+;chat1001", "hi team"]


async def test_send_unconfirmed_when_no_row_or_wrong_row(mac_fixture: MacFixture) -> None:
    runner = FakeRunner()  # Messages.app "accepts" but never records a row
    sender = MessageSender(_store(mac_fixture), runner=runner, confirm_timeout_s=0.2, poll_interval_s=0.05)
    res = await sender.send("+14155550103", "hello?")
    assert res["status"] == "unconfirmed"
    assert "note" in res

    # A row with different text, or in another chat, must not confirm
    insert_outgoing(mac_fixture.chat_db, 7, "something else")
    insert_outgoing(mac_fixture.chat_db, 1, "hello again")
    res2 = await sender.send("+14155550103", "hello again")
    assert res2["status"] == "unconfirmed"


async def test_send_error_row_is_unconfirmed(mac_fixture: MacFixture) -> None:
    class ErrRunner(FakeRunner):
        async def __call__(self, argv: Sequence[str]) -> tuple[int, str, str]:
            self.calls.append(list(argv))
            insert_outgoing(mac_fixture.chat_db, 7, _data_args(argv)[2], error=22)
            return 0, "", ""

    sender = MessageSender(_store(mac_fixture), runner=ErrRunner(), confirm_timeout_s=1, poll_interval_s=0.05)
    res = await sender.send("+14155550103", "will fail")
    assert res["status"] == "unconfirmed"
    assert "error code 22" in res["note"]


async def test_send_falls_back_to_sms(mac_fixture: MacFixture) -> None:
    runner = FakeRunner(mac_fixture, record_chat=7, results=[(1, "", "iMessage account not found"), (0, "", "")])
    sender = MessageSender(_store(mac_fixture), runner=runner, confirm_timeout_s=2, poll_interval_s=0.05)
    res = await sender.send("+14155550103", "sms please")
    assert res["status"] == "sent"
    assert [_data_args(c)[3] for c in runner.calls] == ["iMessage", "SMS"]

    failing = FakeRunner(results=[(1, "", "no iMessage"), (1, "", "no SMS")])
    sender2 = MessageSender(_store(mac_fixture), runner=failing, confirm_timeout_s=0.1)
    with pytest.raises(RuntimeError, match="failed to send"):
        await sender2.send("+14155550103", "x")


async def test_send_automation_denied_gives_tcc_hint(mac_fixture: MacFixture) -> None:
    runner = FakeRunner(results=[(1, "", "execution error: Not authorized to send Apple events to Messages. (-1743)")])
    sender = MessageSender(_store(mac_fixture), runner=runner)
    with pytest.raises(PermissionError, match="Automation"):
        await sender.send("+14155550103", "x")
    assert len(runner.calls) == 1  # no SMS retry after a TCC denial


async def test_send_validation_and_ambiguity_never_call_osascript(mac_fixture: MacFixture) -> None:
    runner = FakeRunner()
    sender = MessageSender(_store(mac_fixture), runner=runner)
    for to, text in [("", "x"), ("+14155550103", ""), ("+14155550103", "   "), (None, "x")]:
        with pytest.raises(ValueError):
            await sender.send(to, text)
    with pytest.raises(ValueError, match="too long"):
        await sender.send("+14155550103", "x" * (MAX_SEND_CHARS + 1))
    with pytest.raises(AmbiguousRecipientError):
        await sender.send("SRC-A:3", "hi Gavin")
    assert runner.calls == []

"""messages_send: handle resolution, Shortcuts & AppleScript backends, argv safety, and chat.db confirmation.

All sends go to mocked subprocess runners; nothing ever runs Shortcuts or reaches Messages.app.
"""

from __future__ import annotations

import json
import os
import plistlib
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from conftest import JARED_ID, MacFixture, insert_outgoing
from zos_mac.chatdb import AmbiguousRecipientError, ChatStore
from zos_mac.main import MacDeviceConfig, parse_args
from zos_mac.send import (
    MAX_SEND_CHARS,
    SEND_SCRIPT_LINES,
    SHORTCUT_NAME,
    SHORTCUTS_BIN,
    MessageSender,
    build_osascript_argv,
)

NASTY_TEXT = 'He said "hi" \\ end\nline2\r\n\t— ☕️ 東京 \'q\' $(rm -rf ~) `x` & end tell'
SHORTCUTS_DIR = Path(__file__).resolve().parents[1] / "shortcuts"


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


class FakeShortcutsRunner:
    """Mock `/usr/bin/shortcuts` runner that inspects input files before cleanup."""

    def __init__(
        self,
        fx: MacFixture | None = None,
        *,
        installed: bool = True,
        record_chat: int | None = None,
        run_results: list[tuple[int, str, str]] | None = None,
    ) -> None:
        self.fx = fx
        self.installed = installed
        self.record_chat = record_chat
        self.run_results = list(run_results or [])
        self.list_calls: list[list[str]] = []
        self.run_calls: list[list[str]] = []
        self.captured_files: list[dict] = []

    async def __call__(self, argv: Sequence[str]) -> tuple[int, str, str]:
        args = list(argv)
        if len(args) >= 2 and args[1] == "list":
            self.list_calls.append(args)
            listing = f"Other Shortcut\n{SHORTCUT_NAME}\n" if self.installed else "Other Shortcut\n"
            return 0, listing, ""

        self.run_calls.append(args)
        input_path = Path(args[args.index("--input-path") + 1])
        mode = stat.S_IMODE(input_path.stat().st_mode)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        self.captured_files.append({"path": input_path, "mode": mode, "payload": payload})

        rc, out, err = self.run_results.pop(0) if self.run_results else (0, "", "")
        if rc == 0 and self.fx is not None and self.record_chat is not None:
            insert_outgoing(self.fx.chat_db, self.record_chat, payload["text"])
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


# ---- config ------------------------------------------------------------------


def test_send_backend_config_toml_and_cli(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('send_backend = "shortcuts"\n', encoding="utf-8")
    assert MacDeviceConfig.resolve(config_path=cfg_file).send_backend == "shortcuts"
    assert parse_args(["--config", str(cfg_file), "--send-backend", "applescript"]).send_backend == "applescript"
    assert parse_args([]).send_backend == "auto"
    with pytest.raises(ValueError, match="send_backend"):
        MacDeviceConfig.resolve(send_backend="smoke-signals")


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


# ---- Shortcuts backend & auto selection -------------------------------------


async def test_auto_selects_shortcuts_when_present_caches_list_and_confirms(mac_fixture: MacFixture) -> None:
    sc_runner = FakeShortcutsRunner(mac_fixture, installed=True, record_chat=7)
    osa_runner = FakeRunner(mac_fixture, record_chat=7)
    sender = MessageSender(
        _store(mac_fixture),
        send_backend="auto",
        runner=osa_runner,
        shortcuts_runner=sc_runner,
        confirm_timeout_s=2,
        poll_interval_s=0.05,
    )

    res = await sender.send("SRC-A:2", NASTY_TEXT)
    assert res["status"] == "sent"
    assert res["backend"] == "shortcuts"
    assert res["chat_id"] == "iMessage;-;+14155550103"
    assert res["contact_name"] == "Monica Hall"
    assert isinstance(res["message_id"], int)
    assert osa_runner.calls == []
    assert len(sc_runner.list_calls) == 1
    assert len(sc_runner.run_calls) == 1

    # Verify exact argv shape, 0600 temp file permissions, JSON content, and post-send cleanup
    argv = sc_runner.run_calls[0]
    assert argv[:4] == [SHORTCUTS_BIN, "run", SHORTCUT_NAME, "--input-path"]
    assert len(argv) == 5
    cap = sc_runner.captured_files[0]
    assert cap["mode"] == 0o600
    assert cap["payload"] == {"to": "+14155550103", "text": NASTY_TEXT}
    assert not cap["path"].exists()  # cleaned up after send

    # Second send reuses the cached `shortcuts list` result instead of shelling out again
    res2 = await sender.send("SRC-A:2", "second message")
    assert res2["status"] == "sent"
    assert res2["backend"] == "shortcuts"
    assert len(sc_runner.list_calls) == 1
    assert len(sc_runner.run_calls) == 2


async def test_auto_falls_back_to_applescript_when_shortcut_absent(mac_fixture: MacFixture) -> None:
    sc_runner = FakeShortcutsRunner(mac_fixture, installed=False)
    osa_runner = FakeRunner(mac_fixture, record_chat=7)
    sender = MessageSender(
        _store(mac_fixture),
        send_backend="auto",
        runner=osa_runner,
        shortcuts_runner=sc_runner,
        confirm_timeout_s=2,
        poll_interval_s=0.05,
    )

    res = await sender.send("+14155550103", "via applescript")
    assert res["status"] == "sent"
    assert res["backend"] == "applescript"
    assert len(sc_runner.list_calls) == 1
    assert sc_runner.run_calls == []
    assert len(osa_runner.calls) == 1


async def test_shortcuts_temp_file_cleaned_up_on_failure_and_errors(mac_fixture: MacFixture) -> None:
    # 1. Explicit shortcuts backend when shortcut not installed
    sc_missing = FakeShortcutsRunner(installed=False)
    sender_missing = MessageSender(
        _store(mac_fixture),
        send_backend="shortcuts",
        shortcuts_runner=sc_missing,
    )
    with pytest.raises(RuntimeError, match=r"devices/mac/README\.md") as ei_missing:
        await sender_missing.send("+14155550103", "hello")
    assert SHORTCUT_NAME in str(ei_missing.value)
    assert sc_missing.run_calls == []

    # 2. Non-zero exit from shortcuts run: temp file still removed, stderr sanitized (handle/text redacted)
    sc_err = FakeShortcutsRunner(
        installed=True,
        run_results=[(1, "", "Error sending secret-msg to +14155550103\n   boom")],
    )
    sender_err = MessageSender(
        _store(mac_fixture),
        send_backend="shortcuts",
        shortcuts_runner=sc_err,
    )
    with pytest.raises(RuntimeError, match="exit 1") as ei_err:
        await sender_err.send("+14155550103", "secret-msg")
    msg = str(ei_err.value)
    assert "secret-msg" not in msg
    assert "+14155550103" not in msg
    assert "[redacted]" in msg
    assert not sc_err.captured_files[0]["path"].exists()

    # 3. Shortcuts privacy denied -> PermissionError
    sc_denied = FakeShortcutsRunner(
        installed=True,
        run_results=[(1, "", "Error: Permission denied by user privacy settings")],
    )
    sender_denied = MessageSender(
        _store(mac_fixture),
        send_backend="shortcuts",
        shortcuts_runner=sc_denied,
    )
    with pytest.raises(PermissionError, match="Always Allow"):
        await sender_denied.send("+14155550103", "hi")
    assert not sc_denied.captured_files[0]["path"].exists()

    # 4. Shortcuts unconfirmed when chat.db never gets a row
    sc_unconf = FakeShortcutsRunner(mac_fixture, installed=True, record_chat=None)
    sender_unconf = MessageSender(
        _store(mac_fixture),
        send_backend="shortcuts",
        shortcuts_runner=sc_unconf,
        confirm_timeout_s=0.15,
        poll_interval_s=0.05,
    )
    res = await sender_unconf.send("+14155550103", "hello?")
    assert res["status"] == "unconfirmed"
    assert res["backend"] == "shortcuts"
    assert "shortcuts" in res["note"]


async def test_chat_id_routes_to_applescript_or_errors_when_shortcuts_only(mac_fixture: MacFixture) -> None:
    sc_runner = FakeShortcutsRunner(mac_fixture, installed=True)
    osa_runner = FakeRunner(mac_fixture, record_chat=3)

    # Under auto (even with Shortcut installed), chat_id routes to AppleScript
    sender_auto = MessageSender(
        _store(mac_fixture),
        send_backend="auto",
        runner=osa_runner,
        shortcuts_runner=sc_runner,
        confirm_timeout_s=2,
        poll_interval_s=0.05,
    )
    res = await sender_auto.send("iMessage;+;chat1001", "hi team")
    assert res["status"] == "sent"
    assert res["backend"] == "applescript"
    assert res["chat_id"] == "iMessage;+;chat1001"
    assert sc_runner.run_calls == []
    assert _data_args(osa_runner.calls[0])[:3] == ["chat", "iMessage;+;chat1001", "hi team"]

    # Under send_backend="shortcuts", chat_id fails clearly without invoking either runner
    sender_sc = MessageSender(
        _store(mac_fixture),
        send_backend="shortcuts",
        runner=osa_runner,
        shortcuts_runner=sc_runner,
    )
    with pytest.raises(RuntimeError, match="AppleScript") as ei:
        await sender_sc.send("iMessage;+;chat1001", "hi team")
    assert "Automation" in str(ei.value)
    assert len(osa_runner.calls) == 1
    assert sc_runner.run_calls == []


# ---- AppleScript send flow ---------------------------------------------------


async def test_send_confirmed_via_chat_db(mac_fixture: MacFixture) -> None:
    runner = FakeRunner(mac_fixture, record_chat=7)
    sender = MessageSender(
        _store(mac_fixture),
        send_backend="applescript",
        runner=runner,
        confirm_timeout_s=2,
        poll_interval_s=0.05,
    )
    res = await sender.send("SRC-A:2", NASTY_TEXT)
    assert res["status"] == "sent"
    assert res["backend"] == "applescript"
    assert res["chat_id"] == "iMessage;-;+14155550103"
    assert res["contact_name"] == "Monica Hall"
    assert isinstance(res["message_id"], int)
    assert len(runner.calls) == 1
    assert _data_args(runner.calls[0]) == ["handle", "+14155550103", NASTY_TEXT, "iMessage"]


async def test_send_unconfirmed_when_no_row_or_wrong_row(mac_fixture: MacFixture) -> None:
    runner = FakeRunner()  # Messages.app "accepts" but never records a row
    sender = MessageSender(
        _store(mac_fixture),
        send_backend="applescript",
        runner=runner,
        confirm_timeout_s=0.2,
        poll_interval_s=0.05,
    )
    res = await sender.send("+14155550103", "hello?")
    assert res["status"] == "unconfirmed"
    assert res["backend"] == "applescript"
    assert "note" in res

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

    sender = MessageSender(
        _store(mac_fixture),
        send_backend="applescript",
        runner=ErrRunner(),
        confirm_timeout_s=1,
        poll_interval_s=0.05,
    )
    res = await sender.send("+14155550103", "will fail")
    assert res["status"] == "unconfirmed"
    assert "error code 22" in res["note"]


async def test_send_falls_back_to_sms(mac_fixture: MacFixture) -> None:
    runner = FakeRunner(mac_fixture, record_chat=7, results=[(1, "", "iMessage account not found"), (0, "", "")])
    sender = MessageSender(
        _store(mac_fixture),
        send_backend="applescript",
        runner=runner,
        confirm_timeout_s=2,
        poll_interval_s=0.05,
    )
    res = await sender.send("+14155550103", "sms please")
    assert res["status"] == "sent"
    assert [_data_args(c)[3] for c in runner.calls] == ["iMessage", "SMS"]

    failing = FakeRunner(results=[(1, "", "no iMessage"), (1, "", "no SMS")])
    sender2 = MessageSender(
        _store(mac_fixture),
        send_backend="applescript",
        runner=failing,
        confirm_timeout_s=0.1,
    )
    with pytest.raises(RuntimeError, match="failed to send"):
        await sender2.send("+14155550103", "x")


async def test_send_automation_denied_and_osascript_timeout_hints(mac_fixture: MacFixture) -> None:
    runner = FakeRunner(results=[(1, "", "execution error: Not authorized to send Apple events to Messages. (-1743)")])
    sender = MessageSender(_store(mac_fixture), send_backend="applescript", runner=runner)
    with pytest.raises(PermissionError, match="Automation"):
        await sender.send("+14155550103", "x")
    assert len(runner.calls) == 1  # no SMS retry after a TCC denial

    timeout_runner = FakeRunner(results=[(124, "", "osascript timed out")])
    sender_to = MessageSender(_store(mac_fixture), send_backend="applescript", runner=timeout_runner)
    with pytest.raises(RuntimeError, match="waiting on the Automation -> Messages prompt") as ei:
        await sender_to.send("+14155550103", "x")
    assert "Shortcuts backend" in str(ei.value)
    assert len(timeout_runner.calls) == 1  # do not hang another 30s retrying SMS


async def test_send_validation_and_ambiguity_never_call_runners(mac_fixture: MacFixture) -> None:
    runner = FakeRunner()
    sc_runner = FakeShortcutsRunner(installed=True)
    sender = MessageSender(
        _store(mac_fixture),
        send_backend="auto",
        runner=runner,
        shortcuts_runner=sc_runner,
    )
    for to, text in [("", "x"), ("+14155550103", ""), ("+14155550103", "   "), (None, "x")]:
        with pytest.raises(ValueError):
            await sender.send(to, text)
    with pytest.raises(ValueError, match="too long"):
        await sender.send("+14155550103", "x" * (MAX_SEND_CHARS + 1))
    with pytest.raises(AmbiguousRecipientError):
        await sender.send("SRC-A:3", "hi Gavin")
    assert runner.calls == []
    assert sc_runner.run_calls == []


# ---- Committed Shortcut artifact checks -------------------------------------


def test_committed_shortcut_plist_and_signed_file_are_valid_and_clean() -> None:
    plist_path = SHORTCUTS_DIR / f"{SHORTCUT_NAME}.plist"
    signed_path = SHORTCUTS_DIR / f"{SHORTCUT_NAME}.shortcut"
    assert plist_path.is_file()
    assert signed_path.is_file()

    raw_xml = plist_path.read_text(encoding="utf-8")
    doc = plistlib.loads(raw_xml.encode("utf-8"))

    actions = doc["WFWorkflowActions"]
    ids = [a["WFWorkflowActionIdentifier"] for a in actions]
    assert ids == [
        "is.workflow.actions.detect.dictionary",
        "is.workflow.actions.getvalueforkey",
        "is.workflow.actions.getvalueforkey",
        "is.workflow.actions.sendmessage",
    ]
    assert actions[1]["WFWorkflowActionParameters"]["WFDictionaryKey"] == "to"
    assert actions[2]["WFWorkflowActionParameters"]["WFDictionaryKey"] == "text"
    send_params = actions[3]["WFWorkflowActionParameters"]
    assert send_params["ShowWhenRun"] is False
    assert (
        send_params["WFSendMessageActionRecipients"]["Value"]["OutputUUID"]
        == actions[1]["WFWorkflowActionParameters"]["UUID"]
    )
    assert (
        send_params["WFSendMessageContent"]["Value"]["attachmentsByRange"]["{0, 1}"]["OutputUUID"]
        == actions[2]["WFWorkflowActionParameters"]["UUID"]
    )

    # All UUIDs in the plist must be deterministic placeholders (not random user UUIDs)
    uuids = re.findall(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", raw_xml)
    assert uuids and all(u.startswith("11111111-1111-4111-8111-11111111110") for u in uuids)

    # No emails, phone numbers, or local username in either file
    email_re = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    phone_re = re.compile(rb"\+[0-9]{7,15}|\b[0-9]{3}[-.\s][0-9]{3}[-.\s][0-9]{4}\b")
    user_bytes = os.environ.get("USER", "").encode("utf-8")
    for blob in (raw_xml.encode("utf-8"), signed_path.read_bytes()):
        assert not email_re.search(blob)
        assert not phone_re.search(blob)
        if len(user_bytes) >= 4:
            assert user_bytes.lower() not in blob.lower()

    # Signed archive starts with Apple Encrypted Archive header (`AEA1`)
    assert signed_path.read_bytes().startswith(b"AEA1")

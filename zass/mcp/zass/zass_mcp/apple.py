"""Local macOS connectors via osascript: iMessage send, Contacts search.

These require no OAuth — macOS will prompt once for Automation permission
the first time each app is scripted. Arguments are passed to AppleScript
through argv (never string-interpolated), so content cannot inject script.
"""

from __future__ import annotations

import subprocess
import sys

OSA_TIMEOUT = 30

SEND_IMESSAGE_SCRIPT = """
on run argv
    set theRecipient to item 1 of argv
    set theMessage to item 2 of argv
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant theRecipient of targetService
        send theMessage to targetBuddy
    end tell
    return "sent"
end run
"""

CONTACTS_SEARCH_SCRIPT = """
on run argv
    set searchTerm to item 1 of argv
    set output to ""
    tell application "Contacts"
        set matches to (every person whose name contains searchTerm)
        repeat with p in matches
            set personLine to (name of p)
            set emailList to ""
            repeat with e in (emails of p)
                set emailList to emailList & (value of e) & " "
            end repeat
            set phoneList to ""
            repeat with ph in (phones of p)
                set phoneList to phoneList & (value of ph) & " "
            end repeat
            set output to output & personLine & " | emails: " & emailList & "| phones: " & phoneList & linefeed
        end repeat
    end tell
    return output
end run
"""


def _require_darwin() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("This connector only works on macOS.")


def _run_osascript(script: str, *args: str) -> str:
    _require_darwin()
    result = subprocess.run(
        ["osascript", "-e", script, *args],
        capture_output=True,
        text=True,
        timeout=OSA_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"osascript failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def send_imessage(to: str, message: str) -> str:
    """Send an iMessage. `to` is a phone number (+1...) or iMessage email."""
    return _run_osascript(SEND_IMESSAGE_SCRIPT, to, message)


def contacts_search(query: str) -> str:
    """Search macOS Contacts by name. Returns 'Name | emails: ... | phones: ...' lines."""
    out = _run_osascript(CONTACTS_SEARCH_SCRIPT, query)
    return out if out else "(no contacts matched)"

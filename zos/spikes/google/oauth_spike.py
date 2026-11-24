"""M0.3 throwaway Google OAuth spike.

Runs a loopback OAuth 2.0 authorization code flow requesting:
- https://www.googleapis.com/auth/calendar.events
- https://www.googleapis.com/auth/gmail.readonly
- https://www.googleapis.com/auth/gmail.send

With access_type=offline and prompt=consent.
Verifies a refresh_token is issued and can immediately be exchanged for a fresh access_token.
NEVER prints access tokens, refresh tokens, or user email/calendar content.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx

SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
DEFAULT_REDIRECT_PORT = 8765
DEFAULT_TOKEN_PATH = Path(__file__).parent / "token.json"
DEFAULT_SECRET_PATH = Path(__file__).parent / "client_secret.json"


def load_client_credentials(secret_path: Path) -> tuple[str, str]:
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id, client_secret

    if not secret_path.exists():
        raise SystemExit(
            f"Missing Google OAuth credentials. Set GOOGLE_CLIENT_ID & GOOGLE_CLIENT_SECRET "
            f"or place downloaded Web client JSON at {secret_path}."
        )
    raw = json.loads(secret_path.read_text())
    cfg = raw.get("web") or raw.get("installed")
    if not cfg:
        raise SystemExit(f"Unexpected client_secret JSON shape in {secret_path}")
    return cfg["client_id"], cfg["client_secret"]


def run_loopback_auth(client_id: str, client_secret: str, port: int, token_path: Path) -> None:
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    state = secrets.token_urlsafe(24)
    result: dict[str, str | None] = {"code": None, "state": None, "error": None}
    done = threading.Event()

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            qs = urllib.parse.parse_qs(parsed.query)
            result["code"] = qs.get("code", [None])[0]
            result["state"] = qs.get("state", [None])[0]
            result["error"] = qs.get("error", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ZOS Google spike received callback. You may close this tab.\n")
            done.set()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", port), CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"
    print(f"[google-spike] Open this URL to authorize:\n{url}\n")
    with contextlib.suppress(Exception):
        webbrowser.open(url)

    print(f"[google-spike] Waiting for callback on {redirect_uri} (timeout 300s)...")
    if not done.wait(timeout=300):
        server.shutdown()
        raise SystemExit("Timed out waiting for OAuth callback.")
    server.shutdown()

    if result["error"]:
        raise SystemExit(f"OAuth authorization error: {result['error']}")
    if result["state"] != state:
        raise SystemExit("State mismatch on OAuth callback.")
    if not result["code"]:
        raise SystemExit("Missing authorization code on OAuth callback.")

    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": result["code"],
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        print(f"[google-spike] token exchange status={resp.status_code}")
        if resp.status_code != 200:
            raise SystemExit(f"Token exchange failed: status={resp.status_code} body={resp.text}")
        token_data = resp.json()

    has_refresh = bool(token_data.get("refresh_token"))
    granted_scopes = sorted(token_data.get("scope", "").split())
    print(
        f"[google-spike] issued: has_refresh_token={has_refresh} "
        f"expires_in={token_data.get('expires_in')} "
        f"refresh_token_expires_in={token_data.get('refresh_token_expires_in')} "
        f"scopes={granted_scopes}"
    )

    if not has_refresh:
        raise SystemExit("No refresh_token returned!")

    token_path.write_text(
        json.dumps(
            {
                "issued_at": int(time.time()),
                "refresh_token": token_data["refresh_token"],
                "scopes": granted_scopes,
                "refresh_token_expires_in": token_data.get("refresh_token_expires_in"),
            },
            indent=2,
        )
    )
    os.chmod(token_path, 0o600)
    print(f"[google-spike] saved gitignored token record to {token_path} (mode 0600)")

    verify_refresh(client_id, client_secret, token_path)


def verify_refresh(client_id: str, client_secret: str, token_path: Path) -> None:
    if not token_path.exists():
        raise SystemExit(f"No token file at {token_path}")
    saved = json.loads(token_path.read_text())
    age_s = int(time.time()) - int(saved.get("issued_at", 0))

    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": saved["refresh_token"],
                "grant_type": "refresh_token",
            },
        )
        print(f"[google-spike] refresh check (age={age_s}s): status={resp.status_code}")
        if resp.status_code != 200:
            raise SystemExit(f"Refresh failed: status={resp.status_code} body={resp.text}")
        data = resp.json()

    print(
        f"[google-spike] refresh OK: has_access_token={bool(data.get('access_token'))} "
        f"expires_in={data.get('expires_in')} "
        f"scopes={sorted(data.get('scope', '').split())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secret", type=Path, default=DEFAULT_SECRET_PATH)
    parser.add_argument("--token", type=Path, default=DEFAULT_TOKEN_PATH)
    parser.add_argument("--port", type=int, default=DEFAULT_REDIRECT_PORT)
    parser.add_argument("--recheck", action="store_true", help="Only verify refresh of saved token")
    args = parser.parse_args()

    client_id, client_secret = load_client_credentials(args.secret)
    if args.recheck:
        verify_refresh(client_id, client_secret, args.token)
    else:
        run_loopback_auth(client_id, client_secret, args.port, args.token)


if __name__ == "__main__":
    main()

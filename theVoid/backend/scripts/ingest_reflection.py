#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
import time

import httpx


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a reflection audio file, run transcription+insights, and print generated tags."
    )
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--identity-token", default="dev-ingest")
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--daily-checkin", default="20:00")
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--audio", required=True, help="Path to .m4a (or audio bytes accepted by backend)")
    parser.add_argument("--duration-seconds", type=int, default=120)
    parser.add_argument("--local-date", default=date.today().isoformat())
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    return parser.parse_args()


def _normalize_dev_token(token: str) -> str:
    trimmed = token.strip()
    if not trimmed:
        return "dev-ingest"
    if trimmed.startswith("dev-"):
        return trimmed
    return f"dev-{trimmed}"


def _print_response_error(prefix: str, response: httpx.Response) -> None:
    body = response.text.strip()
    if not body:
        body = "<empty>"
    print(f"{prefix} failed: HTTP {response.status_code} {body}", file=sys.stderr)


def main() -> int:
    args = _parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists() or not audio_path.is_file():
        print(f"Audio path not found: {audio_path}", file=sys.stderr)
        return 1

    if args.duration_seconds < 1 or args.duration_seconds > 300:
        print("--duration-seconds must be between 1 and 300", file=sys.stderr)
        return 1

    identity_token = _normalize_dev_token(args.identity_token)
    audio_bytes = audio_path.read_bytes()
    if not audio_bytes:
        print(f"Audio file is empty: {audio_path}", file=sys.stderr)
        return 1

    timeout = max(args.timeout_seconds, 30.0)
    started = time.monotonic()
    with httpx.Client(base_url=args.api_base_url.rstrip("/"), timeout=30.0) as client:
        auth_response = client.post(
            "/auth/apple",
            json={
                "identity_token": identity_token,
                "display_name": args.display_name,
                "daily_checkin_time_local": args.daily_checkin,
                "timezone": args.timezone,
            },
        )
        if auth_response.status_code >= 300:
            _print_response_error("auth", auth_response)
            return 1
        auth_payload = auth_response.json()
        session_token = auth_payload["access_token"]
        headers = {"Authorization": f"Bearer {session_token}"}

        entry_response = client.post(
            "/entries",
            headers=headers,
            json={
                "local_date": args.local_date,
                "duration_seconds": args.duration_seconds,
            },
        )
        if entry_response.status_code >= 300:
            _print_response_error("create entry", entry_response)
            return 1
        entry_payload = entry_response.json()
        entry_id = entry_payload["entry_id"]
        upload_url = entry_payload["upload_url"]

        upload_response = client.put(
            upload_url,
            content=audio_bytes,
            headers={
                **headers,
                "Content-Type": "application/octet-stream",
            },
        )
        if upload_response.status_code >= 300:
            _print_response_error("upload audio", upload_response)
            return 1

        complete_response = client.post(
            f"/entries/{entry_id}/complete_upload",
            headers=headers,
            json={"content_type": "audio/m4a"},
        )
        if complete_response.status_code >= 300:
            _print_response_error("complete upload", complete_response)
            return 1

        print(f"Entry queued: {entry_id}")
        while True:
            if time.monotonic() - started > timeout:
                print("Timed out waiting for insights to finish", file=sys.stderr)
                return 1

            state_response = client.get(f"/entries/{entry_id}", headers=headers)
            if state_response.status_code >= 300:
                _print_response_error("fetch entry", state_response)
                return 1

            entry = state_response.json()
            status = entry.get("status", "")
            print(f"status={status}")

            if status == "ready":
                insight = entry.get("insight") or {}
                transcript = (entry.get("transcript") or {}).get("text", "")
                tags = insight.get("mood_tags") or []
                summary = insight.get("summary") or ""
                print(f"tags={tags}")
                print(f"summary={summary}")
                if transcript:
                    preview = transcript.strip().replace("\n", " ")
                    print(f"transcript={preview[:240]}")
                return 0

            if status == "failed":
                print("Entry processing failed", file=sys.stderr)
                return 1

            time.sleep(max(0.5, args.poll_interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())

"""Global rate limiters and authentication failure auditing for public exposure hardening.

Behind a reverse proxy or tunnel (e.g., cloudflared), all external requests share the same
peer IP. For a single-owner ZOS instance, global keying eliminates IP-spoofing and shared-peer
bypasses while keeping memory bounded.
"""

from __future__ import annotations

import contextlib
import time
from collections import deque
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from zos.audit import record_event
from zos.db.queries import get_principal_and_space


class GlobalPassphraseRateLimiter:
    """Global sliding-window + exponential-backoff limiter shared across all passphrase forms.

    Enforces:
      - `max_per_minute` failures per 60s window (default: 5)
      - `max_per_hour` failures per 3600s window (default: 20)
      - Exponential backoff on consecutive failures once >= `max_per_minute`
        (`backoff_base_s * 2 ** (consecutive - max_per_minute)`, capped at `max_backoff_s`).
    """

    def __init__(
        self,
        *,
        max_per_minute: int = 5,
        max_per_hour: int = 20,
        minute_window_seconds: float = 60.0,
        hour_window_seconds: float = 3600.0,
        backoff_base_seconds: float = 10.0,
        max_backoff_seconds: float = 3600.0,
    ) -> None:
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.minute_window_seconds = minute_window_seconds
        self.hour_window_seconds = hour_window_seconds
        self.backoff_base_seconds = backoff_base_seconds
        self.max_backoff_seconds = max_backoff_seconds

        self._minute_failures: deque[float] = deque()
        self._hour_failures: deque[float] = deque()
        self._consecutive_failures: int = 0
        self._backoff_until: float = 0.0

    def _prune(self, now: float) -> None:
        min_cutoff = now - self.minute_window_seconds
        while self._minute_failures and self._minute_failures[0] <= min_cutoff:
            self._minute_failures.popleft()
        hr_cutoff = now - self.hour_window_seconds
        while self._hour_failures and self._hour_failures[0] <= hr_cutoff:
            self._hour_failures.popleft()

    def retry_after_seconds(self, *, now: float | None = None) -> int | None:
        """Return Retry-After seconds if currently blocked, else None."""
        current = time.monotonic() if now is None else now
        self._prune(current)

        waits: list[float] = []
        if self._backoff_until > current:
            waits.append(self._backoff_until - current)
        if len(self._minute_failures) >= self.max_per_minute:
            waits.append(self.minute_window_seconds - (current - self._minute_failures[0]))
        if len(self._hour_failures) >= self.max_per_hour:
            waits.append(self.hour_window_seconds - (current - self._hour_failures[0]))

        if not waits:
            return None
        return max(1, int(max(waits)) + 1)

    def record_failure(self, *, now: float | None = None) -> None:
        """Record a failed passphrase attempt and update sliding windows & backoff."""
        current = time.monotonic() if now is None else now
        self._prune(current)
        self._minute_failures.append(current)
        self._hour_failures.append(current)
        self._consecutive_failures += 1

        if self._consecutive_failures >= self.max_per_minute:
            exp = self._consecutive_failures - self.max_per_minute
            delay = min(self.max_backoff_seconds, self.backoff_base_seconds * (2**exp))
            self._backoff_until = max(self._backoff_until, current + delay)

    def record_success(self) -> None:
        """Clear consecutive backoff streak on a valid owner passphrase.

        Does not erase hourly failure history so a legitimate owner login cannot be used
        to reset an attacker's hourly budget.
        """
        self._consecutive_failures = 0
        self._backoff_until = 0.0


class GlobalSlidingWindowLimiter:
    """Simple global sliding-window limiter (e.g. for unauthenticated DCR)."""

    def __init__(self, *, max_requests: int = 10, window_seconds: float = 3600.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._events: deque[float] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()

    def check_and_record(self, *, now: float | None = None) -> int | None:
        """If allowed, record the request and return None; else return Retry-After seconds."""
        current = time.monotonic() if now is None else now
        self._prune(current)
        if len(self._events) >= self.max_requests:
            wait = self.window_seconds - (current - self._events[0])
            return max(1, int(wait) + 1)
        self._events.append(current)
        return None


async def record_auth_failure(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    route: str,
    reason: str = "invalid_passphrase",
    client_id: UUID | None = None,
) -> None:
    """Audit a failed authentication attempt without logging any passphrase material."""
    with contextlib.suppress(ValueError):
        principal, space = await get_principal_and_space(pool)
        await record_event(
            pool,
            principal_id=principal["id"],
            space_id=space["id"],
            client_id=client_id,
            event_type="auth.failed",
            outcome="denied",
            payload={"route": route, "reason": reason},
        )

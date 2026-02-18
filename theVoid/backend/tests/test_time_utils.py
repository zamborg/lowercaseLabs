from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.db import ensure_utc


def test_ensure_utc_promotes_naive_datetime_to_utc() -> None:
    naive = datetime(2026, 2, 18, 12, 0, 0)
    normalized = ensure_utc(naive)

    assert normalized.tzinfo is not None
    assert normalized.utcoffset() == timedelta(0)
    assert normalized.year == 2026
    assert normalized.month == 2
    assert normalized.day == 18


def test_ensure_utc_converts_aware_datetime_to_utc() -> None:
    plus_two = timezone(timedelta(hours=2))
    aware = datetime(2026, 2, 18, 12, 0, 0, tzinfo=plus_two)
    normalized = ensure_utc(aware)

    assert normalized.tzinfo is not None
    assert normalized.utcoffset() == timedelta(0)
    assert normalized.hour == 10

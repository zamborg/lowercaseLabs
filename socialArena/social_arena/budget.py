"""Process-safe persisted cost reservations; unknown charges retain full reservation."""
import contextlib
import fcntl
import math
from pathlib import Path
from .common import atomic_json, read_json


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, max_usd=0, allow_paid=False):
        self.path = Path(path)
        if type(max_usd) not in (int,float) or not math.isfinite(max_usd) or max_usd < 0:
            raise ValueError('budget must be finite nonnegative')
        self.max_usd = max_usd
        self.allow_paid = allow_paid
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def locked(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = read_json(self.path) if self.path.exists() else {}
            yield data
            atomic_json(self.path, data)

    def reserve(self, request_id, amount, metadata=None):
        if not self.allow_paid or self.max_usd <= 0:
            raise BudgetExceeded('paid providers disabled or cap is zero')
        if not math.isfinite(amount) or amount < 0:
            raise ValueError('negative reservation')
        with self.locked() as data:
            if request_id in data:
                raise BudgetExceeded('request already dispatched; uncertain calls are never automatically repeated')
            used = sum(x.get('actual_usd', x['reserved_usd']) for x in data.values())
            if used + amount > self.max_usd + 1e-12:
                raise BudgetExceeded('spend cap reached')
            data[request_id] = dict(reserved_usd=amount, status='pending', **(metadata or {}))

    def settle(self, request_id, actual=None, **details):
        with self.locked() as data:
            entry = data[request_id]
            if actual is not None:
                if actual > entry['reserved_usd'] + 1e-10:
                    raise RuntimeError('actual charge exceeded conservative reservation')
                entry['actual_usd'] = actual
            entry.update(status='settled' if actual is not None else 'uncertain', **details)

    def entries(self):
        with self.locked() as data:
            return dict(data)

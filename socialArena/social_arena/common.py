import hashlib
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        f.write(canonical(value) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_json(path, value):
    with Path(path).open('a') as f:
        f.write(canonical(value) + '\n')
        f.flush()
        os.fsync(f.fileno())


def read_json(path):
    return json.loads(Path(path).read_text())


def read_stream(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    # A killed writer can leave only the final line partial. Recover that tail.
    end = data.rfind(b'\n') + 1
    if end != len(data):
        with path.open('r+b') as f:
            f.truncate(end)
    return [json.loads(line) for line in data[:end].splitlines()]

# blackhole

`blackhole` is a capture-first iOS and FastAPI project for voice notes, todos, and note search.

## Layout

- `backend/`: FastAPI API, SQLAlchemy models, Alembic migrations, tests, Fly config hooks
- `ios/blackhole/`: SwiftUI iOS app generated with XcodeGen

## Quick start

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload
```

### iOS

```bash
cd ios/blackhole
xcodegen generate
open blackhole.xcodeproj
```

The iOS app defaults to development auth through `POST /auth/dev` so the first local run does not depend on Apple Sign In.


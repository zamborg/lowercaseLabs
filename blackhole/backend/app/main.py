import asyncio
import json
import uuid
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import analysis, auth, db, schemas

app = FastAPI(title="blackhole")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await asyncio.to_thread(db.init_db)


async def get_user_id(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ")
    try:
        return auth.verify_session_token(token)
    except Exception:
        raise HTTPException(401, "Invalid or expired session token")


@app.post("/auth/apple")
async def sign_in_apple(body: schemas.AppleSignInRequest):
    try:
        apple_user_id = await auth.verify_apple_token(body.identity_token)
    except Exception as e:
        raise HTTPException(401, f"Apple token verification failed: {e}")

    await asyncio.to_thread(db.upsert_user, apple_user_id)
    session_token = auth.create_session_token(apple_user_id)
    return {"session_token": session_token}


@app.post("/items", response_model=schemas.Item)
async def create_item(body: schemas.CreateItemRequest, user_id: str = Depends(get_user_id)):
    try:
        result = await asyncio.to_thread(analysis.analyze_transcript, body.content)
    except Exception:
        result = {"type": "note", "title": body.content[:60], "tags": [], "due_date": None}

    now = datetime.utcnow().isoformat()
    item_data = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "content": body.content,
        "title": result.get("title", body.content[:60]),
        "type": result.get("type", "note"),
        "due_date": result.get("due_date"),
        "completed": 0,
        "tags": json.dumps(result.get("tags", [])),
        "created_at": now,
        "updated_at": now,
    }

    await asyncio.to_thread(db.create_item, item_data)
    return _to_schema(item_data)


@app.get("/items", response_model=list[schemas.Item])
async def list_items(user_id: str = Depends(get_user_id)):
    rows = await asyncio.to_thread(db.list_items, user_id)
    return [_to_schema(dict(r)) for r in rows]


@app.patch("/items/{item_id}", response_model=schemas.Item)
async def update_item(
    item_id: str,
    body: schemas.UpdateItemRequest,
    user_id: str = Depends(get_user_id),
):
    row = await asyncio.to_thread(db.get_item, item_id, user_id)
    if not row:
        raise HTTPException(404, "Item not found")

    updates: dict = {}
    if body.completed is not None:
        updates["completed"] = int(body.completed)
    updates["updated_at"] = datetime.utcnow().isoformat()

    await asyncio.to_thread(db.update_item, item_id, updates)
    updated = await asyncio.to_thread(db.get_item, item_id, user_id)
    return _to_schema(dict(updated))


@app.delete("/items/{item_id}")
async def delete_item(item_id: str, user_id: str = Depends(get_user_id)):
    row = await asyncio.to_thread(db.get_item, item_id, user_id)
    if not row:
        raise HTTPException(404, "Item not found")
    await asyncio.to_thread(db.delete_item, item_id)
    return {}


@app.post("/search", response_model=list[schemas.Item])
async def search(body: schemas.SearchRequest, user_id: str = Depends(get_user_id)):
    rows = await asyncio.to_thread(db.list_items, user_id)
    if not rows:
        return []

    items = [dict(r) for r in rows]
    try:
        results = await asyncio.to_thread(analysis.search_items, body.query, items)
    except Exception:
        q = body.query.lower()
        results = [i for i in items if q in i["content"].lower() or q in i["title"].lower()]

    return [_to_schema(i) for i in results]


def _to_schema(item: dict) -> schemas.Item:
    tags = item.get("tags", "[]")
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = []

    return schemas.Item(
        id=item["id"],
        content=item["content"],
        title=item["title"],
        type=item["type"],
        due_date=item.get("due_date"),
        completed=bool(item.get("completed", 0)),
        tags=tags,
        created_at=item["created_at"],
        updated_at=item["updated_at"],
    )

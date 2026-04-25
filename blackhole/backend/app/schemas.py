from typing import Optional

from pydantic import BaseModel


class AppleSignInRequest(BaseModel):
    identity_token: str


class CreateItemRequest(BaseModel):
    content: str


class UpdateItemRequest(BaseModel):
    completed: Optional[bool] = None
    title: Optional[str] = None
    content: Optional[str] = None
    type: Optional[str] = None
    due_date: Optional[str] = None
    tags: Optional[list[str]] = None


class Item(BaseModel):
    id: str
    content: str
    title: str
    type: str
    due_date: Optional[str] = None
    completed: bool
    tags: list[str]
    created_at: str
    updated_at: str


class SearchRequest(BaseModel):
    query: str

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class APIMessage(BaseModel):
    message: str


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    display_name: str
    email: str | None
    auth_provider: str
    created_at: datetime


class AuthSession(BaseModel):
    access_token: str
    user: UserRead


class DevAuthRequest(BaseModel):
    display_name: str
    email: str | None = None


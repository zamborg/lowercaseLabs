from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .models import RevealMode


class UserProfile(BaseModel):
    id: str
    display_name: str | None
    anonymous_handle: str
    daily_checkin_time_local: str
    timezone: str
    notification_enabled: bool

    model_config = {"from_attributes": True}


class AuthAppleRequest(BaseModel):
    identity_token: str = Field(min_length=1)
    nonce: str | None = Field(default=None, min_length=1, max_length=255)
    display_name: str | None = Field(default=None, max_length=120)
    daily_checkin_time_local: str | None = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    timezone: str | None = Field(default=None, max_length=64)


class AuthSessionResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserProfile


class UpdateProfileRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=120)
    daily_checkin_time_local: str | None = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    timezone: str | None = Field(default=None, max_length=64)
    notification_enabled: bool | None = None


class FriendInviteRequest(BaseModel):
    expires_in_days: int = Field(default=7, ge=1, le=30)
    max_uses: int = Field(default=25, ge=1, le=100)


class FriendInviteResponse(BaseModel):
    invite_token: str
    invite_url: str
    expires_at: datetime


class FriendAcceptRequest(BaseModel):
    token: str = Field(min_length=1)


class SocialDot(BaseModel):
    user_id: str
    dot_color: str
    dot_tags: list[str] = Field(default_factory=list)
    label: str | None
    is_revealed: bool
    has_entry: bool
    presence_id: str | None = None
    local_date: date | None = None
    updated_at: datetime | None = None


class SocialDotsResponse(BaseModel):
    local_date: date
    dots: list[SocialDot]


class UpdateSocialPresenceRequest(BaseModel):
    reveal_mode: RevealMode
    reveal_friend_ids: list[str] = Field(default_factory=list)
    display_name_override: str | None = Field(default=None, max_length=120)


class UpdateSocialDotRequest(BaseModel):
    mood_score: float | None = Field(default=None, ge=-2.0, le=2.0)
    mood_tags: list[str] = Field(default_factory=list, max_length=8)
    dot_color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")

    @model_validator(mode="after")
    def validate_color_source(self) -> "UpdateSocialDotRequest":
        if self.mood_score is None and self.dot_color is None:
            raise ValueError("Either mood_score or dot_color must be provided")
        return self


class FeedbackCreateRequest(BaseModel):
    kind: Literal["idea", "bug"]
    message: str = Field(min_length=1, max_length=4000)


class MetricsResponse(BaseModel):
    avg_job_latency_seconds: float
    failed_jobs_last_24h: int


class MessageResponse(BaseModel):
    message: str

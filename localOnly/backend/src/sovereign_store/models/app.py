from __future__ import annotations

import enum

from sqlalchemy import Boolean, Enum as SAEnum, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sovereign_store.core.database import Base
from sovereign_store.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class NetworkingMode(str, enum.Enum):
    PROXY = "proxy"
    DIRECT = "direct"


class App(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "apps"

    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    docker_image: Mapped[str] = mapped_column(String(255), nullable=False)
    fly_app_name: Mapped[str] = mapped_column(String(64), nullable=False)
    networking_mode: Mapped[NetworkingMode] = mapped_column(
        SAEnum(NetworkingMode, name="networking_mode"), nullable=False
    )
    internal_port: Mapped[int] = mapped_column(Integer, default=8080, nullable=False)
    requires_volume: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    pods = relationship("Pod", back_populates="app")

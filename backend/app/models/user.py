from __future__ import annotations

import enum

from sqlalchemy import Enum as SAEnum
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class UserRole(str, enum.Enum):
    user = "user"
    admin = "admin"


class ExpertiseLevel(str, enum.Enum):
    student = "student"
    researcher = "researcher"
    faculty = "faculty"


class PreferredLanguage(str, enum.Enum):
    en = "en"
    zh = "zh"


class User(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        SAEnum(UserRole), default=UserRole.user
    )
    expertise_level: Mapped[ExpertiseLevel] = mapped_column(
        SAEnum(ExpertiseLevel), default=ExpertiseLevel.student
    )
    preferred_language: Mapped[PreferredLanguage] = mapped_column(
        SAEnum(PreferredLanguage), default=PreferredLanguage.en
    )

    projects: Mapped[list[Project]] = relationship(  # noqa: F821
        back_populates="user", cascade="all, delete-orphan"
    )

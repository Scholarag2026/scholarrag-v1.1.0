from __future__ import annotations

import enum
import uuid

from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class CitationStyle(str, enum.Enum):
    apa = "APA"
    chicago = "Chicago"
    mla = "MLA"
    harvard = "Harvard"
    ieee = "IEEE"
    vancouver = "Vancouver"
    custom = "custom"


class ProjectStatus(str, enum.Enum):
    active = "active"
    archived = "archived"


class Project(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "projects"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    target_journal: Mapped[str | None] = mapped_column(String(500))
    citation_style: Mapped[CitationStyle] = mapped_column(
        SAEnum(CitationStyle), default=CitationStyle.apa
    )
    status: Mapped[ProjectStatus] = mapped_column(
        SAEnum(ProjectStatus), default=ProjectStatus.active
    )
    target_journal_guidelines: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    refined_topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    inclusion_criteria: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    exclusion_criteria: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    user: Mapped[User] = relationship(back_populates="projects")  # noqa: F821

from __future__ import annotations

import enum
import uuid

from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class PaperType(str, enum.Enum):
    literature_review = "literature_review"
    research_article = "research_article"


class DraftStatus(str, enum.Enum):
    draft = "draft"
    review = "review"
    final = "final"


class Draft(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "drafts"

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    paper_type: Mapped[PaperType] = mapped_column(
        SAEnum(PaperType), default=PaperType.literature_review,
    )
    content: Mapped[dict | None] = mapped_column(JSONB, doc="Tiptap JSON document")
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[DraftStatus] = mapped_column(
        SAEnum(DraftStatus), default=DraftStatus.draft,
    )

    versions: Mapped[list[DraftVersion]] = relationship(
        back_populates="draft", cascade="all, delete-orphan",
        order_by="DraftVersion.version.desc()",
    )


class DraftVersion(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "draft_versions"

    draft_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("drafts.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[dict | None] = mapped_column(JSONB, doc="Snapshot of Tiptap JSON")
    change_summary: Mapped[str | None] = mapped_column(Text)

    draft: Mapped[Draft] = relationship(back_populates="versions")

    __table_args__ = (
        UniqueConstraint("draft_id", "version", name="uq_draft_versions_draft_id_version"),
    )

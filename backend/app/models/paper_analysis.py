"""PaperAnalysisRecord — stores per-paper quality analysis for a project."""

from __future__ import annotations

import uuid

from sqlalchemy import Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class PaperAnalysisRecord(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "paper_analyses"

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relevance_score: Mapped[float | None] = mapped_column(Float)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    key_findings: Mapped[list] = mapped_column(JSONB, default=list)
    methodology: Mapped[str] = mapped_column(Text, nullable=False)
    methodology_rigor: Mapped[str] = mapped_column(String(10), nullable=False)
    limitations: Mapped[list] = mapped_column(JSONB, default=list)
    theories_used: Mapped[list] = mapped_column(JSONB, default=list)
    sample_info: Mapped[str | None] = mapped_column(Text)

    project = relationship("Project")
    paper = relationship("Paper")

    __table_args__ = (
        UniqueConstraint(
            "project_id", "paper_id", name="uq_paper_analysis_project_paper"
        ),
    )

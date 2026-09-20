from __future__ import annotations

import enum
import uuid

from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class JobType(str, enum.Enum):
    search = "search"
    analysis = "analysis"
    writing = "writing"
    data_analysis = "data_analysis"
    qa = "qa"
    quality_scoring = "quality_scoring"
    gap_analysis = "gap_analysis"
    graph_building = "graph_building"
    research_design = "research_design"
    data_collection = "data_collection"
    quantitative = "quantitative"
    qualitative = "qualitative"
    inter_coder = "inter_coder"
    deep_search = "deep_search"
    seed_expand = "seed_expand"
    field_foundations = "field_foundations"
    fulltext_acquire = "fulltext_acquire"
    claim_verify = "claim_verify"
    smart_search = "smart_search"
    deep_analysis = "deep_analysis"
    paper_upload = "paper_upload"
    refine = "refine"
    journal_guidelines = "journal_guidelines"
    compliance_check = "compliance_check"


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class AnalysisJob(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "analysis_jobs"

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    job_type: Mapped[JobType] = mapped_column(SAEnum(JobType), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus), default=JobStatus.pending, nullable=False
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    progress_message: Mapped[str | None] = mapped_column(String(500))
    result: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )

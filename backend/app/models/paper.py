from __future__ import annotations

import enum

from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class SourceApi(str, enum.Enum):
    openalex = "openalex"
    semantic_scholar = "semantic_scholar"
    crossref = "crossref"
    core = "core"
    manual = "manual"
    upload = "upload"


class Paper(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "papers"

    doi: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(1000), nullable=False)
    authors: Mapped[list] = mapped_column(JSONB, default=list)
    year: Mapped[int | None] = mapped_column(Integer)
    journal_name: Mapped[str | None] = mapped_column(String(500))
    journal_issn: Mapped[str | None] = mapped_column(String(50))
    is_wos_indexed: Mapped[bool | None] = mapped_column()
    wos_collection: Mapped[str | None] = mapped_column(String(10))
    wos_categories: Mapped[str | None] = mapped_column(String(1000))
    citation_count: Mapped[int | None] = mapped_column(Integer)
    abstract: Mapped[str | None] = mapped_column(Text)
    source_api: Mapped[SourceApi] = mapped_column(
        SAEnum(SourceApi), nullable=False, default=SourceApi.manual
    )
    external_id: Mapped[str | None] = mapped_column(String(255), index=True)
    quality_score: Mapped[float | None] = mapped_column(Float)
    full_text_url: Mapped[str | None] = mapped_column(String(1000))
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

    project_links: Mapped[list[ProjectPaper]] = relationship(  # noqa: F821
        back_populates="paper", cascade="all, delete-orphan"
    )

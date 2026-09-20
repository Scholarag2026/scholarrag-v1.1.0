"""Evidence — one verbatim, code-accepted quote extracted from a paper's own full text.

Written by the full-text analysis step (``app.services.deep_analysis``): the analysis
agent proposes evidence
items, and code -- never the model -- decides which ones are accepted (verbatim, complete
sentences, at least 8 words) before they reach this table. A paper's evidence rows are
replaced wholesale on every re-analysis, so this table always reflects the paper's most
recent analysis, never a mix of two runs.

Consumed by the writer's context (``app.services.writing``): only
``origin == "own"`` rows are shown, grouped by concept, so a cited sentence can restate a
specific finding, method, sample detail or limitation the paper's own authors state,
instead of a whole-paper abstract.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UUIDMixin

#: The only accepted values of ``kind``. Enforced in application code
#: (the analysis agent's ``EvidenceItem`` schema, a Pydantic ``Literal``); kept here as a
#: plain string column, not a Postgres enum type, so adding a fifth kind later needs only
#: a data migration, not an ``ALTER TYPE``.
EVIDENCE_KINDS = ("finding", "method", "sample", "limitation")

#: The only accepted values of ``origin`` (design section 6 amendment A1):
#: "own" is the paper's own claim; "reported" is a claim the paper attributes to another
#: work. Only "own" evidence ever reaches the writer's context.
EVIDENCE_ORIGINS = ("own", "reported")


class Evidence(Base, UUIDMixin):
    """One accepted, verbatim evidence quote from one paper's full text."""

    __tablename__ = "evidence"

    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str | None] = mapped_column(String(255))
    finding: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    origin: Mapped[str] = mapped_column(String(10), nullable=False)
    concepts: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    prompt_version: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    paper = relationship("Paper")

# backend/app/schemas/verification.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class VerifyReferencesRequest(BaseModel):
    paper_ids: list[UUID] = Field(
        default_factory=list,
        description="Specific paper IDs to verify. Empty = verify all.",
    )


class VerificationCheck(BaseModel):
    """One record check or one indicator for a paper.

    Record checks (``doi_exists``, ``metadata_match``) carry ``pass`` / ``fail`` /
    ``warning`` / ``skipped``. Indicators (``wos_indexed``, ``recency``,
    ``citation_count``) carry the neutral ``info`` / ``note`` / ``skipped`` and never
    influence a record's ``overall_status``.
    """

    check_type: str
    status: str
    message: str
    details: dict | None = None


class PaperVerificationResult(BaseModel):
    paper_id: UUID
    doi: str | None
    title: str
    # Derived from ``checks`` only: fail = DOI not found, warning = metadata mismatch.
    overall_status: Literal["pass", "warning", "fail", "no_doi"]
    checks: list[VerificationCheck]  # doi_exists, metadata_match
    indicators: list[VerificationCheck] = Field(
        default_factory=list
    )  # wos_indexed, recency, citation_count
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VerificationReport(BaseModel):
    project_id: UUID
    total_papers: int
    passed: int
    failed: int
    warnings: int
    results: list[PaperVerificationResult]

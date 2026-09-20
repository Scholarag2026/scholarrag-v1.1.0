"""Schemas for study quality scoring and gap analysis."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

# --- Agent output schemas (used as Pydantic AI output_type) ---


class PaperAnalysis(BaseModel):
    """Analysis of a single paper by the Quality Scoring Agent."""

    paper_id: str  # UUID as string for LLM compatibility
    relevance_score: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    key_findings: list[str] = Field(min_length=1, max_length=5)
    methodology: str
    methodology_rigor: Literal["high", "medium", "low"]
    limitations: list[str]
    theories_used: list[str]
    sample_info: str | None = None


class BatchAnalysisResult(BaseModel):
    """Output of the Quality Scoring Agent for a batch of papers."""

    analyses: list[PaperAnalysis]


class ResearchGap(BaseModel):
    gap_type: Literal["unexplored", "under_explored", "controversial", "methodological"]
    description: str
    evidence: list[str]
    severity: Literal["high", "medium", "low"]


class Position(BaseModel):
    view: str
    supporters: list[str]


class Controversy(BaseModel):
    topic: str
    positions: list[Position]


class SuggestedQuestion(BaseModel):
    question: str
    rationale: str
    methodology_hint: str
    related_gap_descriptions: list[str]


class GapReport(BaseModel):
    """Output of the Gap Analysis Agent."""

    gaps: list[ResearchGap]
    controversies: list[Controversy]
    suggested_questions: list[SuggestedQuestion]
    theoretical_landscape: list[str]
    summary: str


# --- API request/response schemas ---


class QualityScoringRequest(BaseModel):
    force: bool = False


class PaperAnalysisResponse(BaseModel):
    """API response for a single paper analysis with joined paper metadata."""

    id: UUID
    project_id: UUID
    paper_id: UUID
    quality_score: float
    relevance_score: float | None
    key_findings: list[str]
    methodology: str
    methodology_rigor: str
    limitations: list[str]
    theories_used: list[str]
    sample_info: str | None
    # Joined paper fields
    paper_title: str
    paper_year: int | None
    paper_doi: str | None

    model_config = {"from_attributes": True}


class PaperAnalysisListResponse(BaseModel):
    analyses: list[PaperAnalysisResponse]
    total: int
    page: int = 1
    limit: int = 200

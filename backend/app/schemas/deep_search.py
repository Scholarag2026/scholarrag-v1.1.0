"""Schemas for deep iterative search — request, round metrics, coverage, query expansion."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DeepSearchRequest(BaseModel):
    """Request body for deep iterative search."""

    query: str = Field(min_length=1, max_length=500)
    year_from: int | None = None
    year_to: int | None = None
    min_citations: int | None = None
    max_rounds: int = 6


class RoundMetrics(BaseModel):
    """Metrics for a single search round."""

    round: int
    strategy: str  # "initial_search" | "query_expansion" | "citation_snowball"
    new_papers: int
    queries: list[str] = []
    papers_expanded: int | None = None
    stopped: bool = False


class CoverageMetrics(BaseModel):
    """Overall coverage metrics for a deep search session."""

    total_scanned: int
    total_unique: int
    rounds: list[RoundMetrics]
    sources: dict[str, int]
    yield_curve: list[int]
    confidence: str  # "high" | "medium" | "low"


class QueryExpansionResult(BaseModel):
    """Output of the query expansion agent."""

    queries: list[str]

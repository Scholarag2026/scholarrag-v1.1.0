"""Schemas for citation graph visualization."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class GraphNode(BaseModel):
    id: str
    title: str
    authors: list[str]
    year: int | None
    citation_count: int | None
    quality_score: float | None
    in_library: bool
    doi: str | None
    abstract: str | None
    is_seed: bool = False
    doi_score: float | None = None
    cluster_id: int | None = None


class GraphEdge(BaseModel):
    source: str
    target: str


class GraphBuildSummary(BaseModel):
    """Outcome of the most recent graph build for this project.

    Lets the client tell "built, 0 links — citation source unavailable" apart from
    "never built" (issue GRAPH-FAILURE-SWALLOWED); ``None`` means never built.
    """

    status: str
    edges_created: int = 0
    papers_processed: int = 0
    papers_failed: int = 0
    citations_unavailable: int = 0
    error: str | None = None
    completed_at: datetime | None = None


class GraphData(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    last_build: GraphBuildSummary | None = None


class GraphExpansion(BaseModel):
    new_nodes: list[GraphNode]
    new_edges: list[GraphEdge]


class ExpandNodeRequest(BaseModel):
    paper_id: UUID

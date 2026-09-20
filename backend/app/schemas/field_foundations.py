"""Schemas for the Field Foundations feature — foundational works identification."""

from pydantic import BaseModel

from app.schemas.paper import PaperData


class FoundationalWork(BaseModel):
    """A single foundational work suggested by the agent."""

    suggested_title: str
    suggested_authors: list[str]
    suggested_year: int
    why_essential: str
    verified: bool = False
    matched_paper: PaperData | None = None


class FoundationalWorksList(BaseModel):
    """Agent output: a list of foundational works."""

    works: list[FoundationalWork]


class FieldFoundationsRequest(BaseModel):
    """Request body for the field foundations endpoint."""

    topic: str
    research_questions: list[str] = []


class FieldFoundationsResult(BaseModel):
    """Final result stored in the job after verification."""

    field: str
    works: list[FoundationalWork]
    verified_count: int
    total_count: int

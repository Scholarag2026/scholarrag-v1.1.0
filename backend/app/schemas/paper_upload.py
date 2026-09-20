"""Schemas for article upload and metadata extraction."""
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class ExtractedPaperMetadata(BaseModel):
    """Output schema for the metadata extraction agent."""
    title: str = "Untitled"
    authors: list[dict] = Field(default_factory=list)
    year: int | None = None
    journal_name: str | None = None
    doi: str | None = None
    abstract: str | None = None


class ExtractedPaper(BaseModel):
    """A single paper extracted from an uploaded file."""
    index: int
    file_name: str
    title: str
    authors: list[dict] = Field(default_factory=list)
    year: int | None = None
    journal_name: str | None = None
    doi: str | None = None
    abstract: str | None = None
    source_api: str = "upload"
    has_full_text: bool = False


class UploadConfirmPaper(BaseModel):
    """A single paper in the confirm request (user-edited metadata)."""
    index: int
    title: str
    authors: list[dict] = Field(default_factory=list)
    year: int | None = None
    journal_name: str | None = None
    doi: str | None = None
    abstract: str | None = None


class UploadConfirmRequest(BaseModel):
    """Request body for POST /papers/upload-confirm."""
    task_id: UUID
    papers: list[UploadConfirmPaper]

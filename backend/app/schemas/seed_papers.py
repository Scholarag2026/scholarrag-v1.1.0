# backend/app/schemas/seed_papers.py
"""Schemas for seed paper resolution and expansion."""

from pydantic import BaseModel, field_validator

from app.schemas.paper import PaperData


class SeedExpandRequest(BaseModel):
    """Request to resolve seed papers (by DOI or title) and expand via citations/references."""

    dois: list[str] = []
    titles: list[str] = []

    @field_validator("dois", "titles", mode="before")
    @classmethod
    def _strip_whitespace(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):
            return v
        return [item.strip() for item in v if item.strip()]

    def model_post_init(self, __context) -> None:
        if not self.dois and not self.titles:
            raise ValueError("At least one DOI or title is required")


class SeedExpandResult(BaseModel):
    """Result of seed expansion: resolved seeds + expanded papers from references/citations."""

    seeds_resolved: int
    seeds_failed: list[str]
    expanded_papers: list[PaperData]
    total_expanded: int

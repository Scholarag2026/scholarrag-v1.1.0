from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    description: str | None = None
    target_journal: str | None = None
    citation_style: str = "APA"


class ProjectUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    description: str | None = None
    target_journal: str | None = None
    citation_style: str | None = None
    status: str | None = None


class ProjectResponse(BaseModel):
    id: UUID
    title: str
    description: str | None
    target_journal: str | None
    citation_style: str
    status: str
    target_journal_guidelines: dict | None = None
    refined_topic: str | None = None
    inclusion_criteria: list[str] | None = None
    exclusion_criteria: list[str] | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectListResponse(BaseModel):
    projects: list[ProjectResponse]
    total: int = 0
    page: int = 1
    limit: int = 50

# backend/app/schemas/paper.py
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class PaperData(BaseModel):
    """Common format for papers from any API source. Used by clients + search service."""
    doi: str | None = None
    title: str
    authors: list[dict] = Field(default_factory=list)  # [{"name": "..."}]
    year: int | None = None
    journal_name: str | None = None
    journal_issn: str | None = None
    citation_count: int | None = None
    abstract: str | None = None
    source_api: str = "manual"
    external_id: str | None = None
    full_text_url: str | None = None
    wos_collection: str | None = None
    wos_categories: str | None = None
    # OpenAlex is the sole scholarly source, so these two fields are
    # populated for every paper the app sees, the same way wos_collection/wos_categories are
    # populated later in the Smart Search pipeline rather than at parse time. Lets
    # app.services.smart_search.run_smart_search apply
    # app.agents.relevance_screener_agent.apply_type_demotion without a second OpenAlex
    # request.
    openalex_type: str | None = None
    is_paratext: bool = False


class PaperResponse(BaseModel):
    id: UUID
    doi: str | None
    title: str
    authors: list[dict]
    year: int | None
    journal_name: str | None
    journal_issn: str | None
    is_wos_indexed: bool | None
    wos_collection: str | None
    wos_categories: str | None
    citation_count: int | None
    abstract: str | None
    source_api: str
    external_id: str | None
    full_text_url: str | None
    metadata: dict | None = Field(default=None, validation_alias="metadata_")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class ProjectPaperResponse(BaseModel):
    id: UUID
    project_id: UUID
    paper_id: UUID
    paper: PaperResponse
    relevance_score: float | None
    user_notes: str | None
    tags: list[str] | None
    added_at: datetime

    model_config = {"from_attributes": True}


class ProjectPaperListResponse(BaseModel):
    papers: list[ProjectPaperResponse]
    total: int
    page: int
    limit: int


class AddPaperRequest(BaseModel):
    """Add a paper to a project. Provide either doi (to look up) or full paper_data."""
    doi: str | None = None
    paper_data: PaperData | None = None
    relevance_score: float | None = None
    user_notes: str | None = None
    tags: list[str] | None = None

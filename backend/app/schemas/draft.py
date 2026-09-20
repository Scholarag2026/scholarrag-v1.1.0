from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DraftCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    paper_type: str = "literature_review"


class DraftUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    content: dict | None = None
    status: str | None = None


class DraftVersionResponse(BaseModel):
    id: UUID
    version: int
    change_summary: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DraftResponse(BaseModel):
    id: UUID
    project_id: UUID
    title: str
    paper_type: str
    content: dict | None
    current_version: int
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DraftListResponse(BaseModel):
    drafts: list[DraftResponse]
    total: int = 0
    page: int = 1
    limit: int = 50


class DraftDetailResponse(BaseModel):
    draft: DraftResponse
    versions: list[DraftVersionResponse]


class LockedBlock(BaseModel):
    blockId: str
    text: str
    position: int


class GenerateRequest(BaseModel):
    section_type: str = Field(description="e.g. 'literature_review', 'introduction', 'abstract'")
    context: str | None = Field(None, description="Additional user instructions for AI")
    language: str = Field("en", description="Output language: en, zh, ja, ko, de, fr, es, pt")
    locked_blocks: list[LockedBlock] | None = Field(None, description="User-written blocks to preserve during regeneration")
    target_words: int | None = Field(
        None,
        gt=0,
        description="Requested body word count (headings excluded). The writer is told "
        "this target plus a hard maximum of ceil(1.25 * target_words); a body that still "
        "exceeds the hard maximum after generation is regenerated once. None runs "
        "generation with no length target at all, exactly as before this field existed.",
    )
    section_title: str | None = Field(
        None,
        description="Requested heading for this section, used verbatim in place of the "
        "section_type-derived default ('literature_review' -> 'Literature Review'). None "
        "(every caller before this field existed) keeps that derived default.",
    )

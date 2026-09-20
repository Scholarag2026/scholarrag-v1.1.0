# backend/app/schemas/search.py
from pydantic import BaseModel, Field

from app.schemas.paper import PaperData


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    year_from: int | None = None
    year_to: int | None = None
    min_citations: int | None = None


class SearchResultResponse(BaseModel):
    papers: list[PaperData]
    total: int
    sources: dict[str, int]  # {"openalex": 15}

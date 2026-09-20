"""Schemas for the draft refiner agent."""
from typing import Literal

from pydantic import BaseModel, Field


class RefineRequest(BaseModel):
    """Request body for POST /drafts/{draft_id}/refine."""
    text: str = Field(min_length=1, description="The text to refine")
    context: str | None = Field(None, description="Surrounding context for better refinement")
    scope: Literal["selection", "section"] = Field(description="'selection' or 'section'")
    section_key: str | None = Field(None, description="Section type if scope is 'section'")


class RefinedText(BaseModel):
    """Output schema for the refiner agent."""
    refined: str

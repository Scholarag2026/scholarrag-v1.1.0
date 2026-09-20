"""Scope refinement endpoint — narrows research scope before Smart Search."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from uuid import UUID
from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.agents.scope_refiner_agent import refine_scope
from app.services import project as project_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scope"])


class QAPair(BaseModel):
    question: str
    answer: str


class ScopeRefineRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    previous_answers: list[QAPair] = Field(default_factory=list)


class ScopeRefineResponse(BaseModel):
    is_specific_enough: bool
    clarifying_question: str | None = None
    options: list[str] = []
    refined_topic: str | None = None
    inclusion_criteria: list[str] = []
    exclusion_criteria: list[str] = []


@router.post("/scope/refine", response_model=ScopeRefineResponse)
async def refine_search_scope(
    req: ScopeRefineRequest,
    user: User = Depends(get_current_user),
):
    """Ask AI to evaluate if the research topic is specific enough, or ask a clarifying question."""
    answers_dicts = [{"question": qa.question, "answer": qa.answer} for qa in req.previous_answers] or None
    try:
        result = await refine_scope(req.query, answers_dicts)
    except Exception as e:
        logger.exception("Scope refinement failed for query: %s", req.query)
        raise HTTPException(status_code=502, detail="Scope refinement failed. Please try again.")
    return ScopeRefineResponse(
        is_specific_enough=result.is_specific_enough,
        clarifying_question=result.clarifying_question,
        options=result.options,
        refined_topic=result.refined_topic,
        inclusion_criteria=result.inclusion_criteria,
        exclusion_criteria=result.exclusion_criteria,
    )


class SaveRefinedTopicRequest(BaseModel):
    refined_topic: str
    inclusion_criteria: list[str] = []
    exclusion_criteria: list[str] = []


@router.post("/projects/{project_id}/refined-topic")
async def save_refined_topic(
    project_id: UUID,
    req: SaveRefinedTopicRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save the refined research topic to the project."""
    project = await project_service.get_project(db, project_id, user.id)
    project.refined_topic = req.refined_topic
    project.inclusion_criteria = req.inclusion_criteria
    project.exclusion_criteria = req.exclusion_criteria
    await db.flush()
    return {"status": "ok"}

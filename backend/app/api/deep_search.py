"""API routes for deep iterative search."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import JobType
from app.models.user import User
from app.schemas.deep_search import DeepSearchRequest
from app.schemas.task import TaskCreateResponse
from app.services import deep_search as deep_search_service
from app.services import project as project_service
from app.services import task as task_service

router = APIRouter()
_session_factory = async_session_factory


@router.post(
    "/projects/{project_id}/deep-search",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def deep_search(
    project_id: UUID,
    req: DeepSearchRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a deep iterative search for the project."""
    await project_service.get_project(db, project_id, user.id)

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.deep_search,
        conflict_detail="Deep search already in progress",
    )

    job = await task_service.create_job(
        db, project_id, user.id, JobType.deep_search
    )

    filters = {
        "year_from": req.year_from,
        "year_to": req.year_to,
        "min_citations": req.min_citations,
        "max_rounds": req.max_rounds,
    }

    background_tasks.add_task(
        deep_search_service.run_deep_search,
        project_id=project_id,
        job_id=job.id,
        query=req.query,
        filters=filters,
        session_factory=_session_factory,
    )

    return TaskCreateResponse(task_id=job.id)

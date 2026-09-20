"""API routes for smart search (PRISMA two-stage filtering with iterative query expansion)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import JobType
from app.models.user import User
from app.schemas.smart_search import SmartSearchRequest
from app.schemas.task import TaskCreateResponse
from app.services import project as project_service
from app.services import smart_search as smart_search_service
from app.services import task as task_service
from app.services.wos_import import get_wos_journal_count

router = APIRouter()
_session_factory = async_session_factory


@router.post(
    "/projects/{project_id}/smart-search",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def smart_search(
    project_id: UUID,
    req: SmartSearchRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a smart search (PRISMA two-stage filtering) for the project.

    The Stage 1 venue filter is optional: ``wos_filter="on"`` is refused with
    409 when no Web of Science journal list is imported; ``"auto"`` (default) and
    ``"off"`` always start the job.
    """
    await project_service.get_project(db, project_id, user.id)

    if req.wos_filter == "on" and await get_wos_journal_count(db) == 0:
        raise HTTPException(
            status_code=409,
            detail="WoS journal list not imported",
        )

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.smart_search,
        conflict_detail="Smart search already in progress",
    )

    job = await task_service.create_job(
        db, project_id, user.id, JobType.smart_search
    )

    background_tasks.add_task(
        smart_search_service.run_smart_search,
        project_id=project_id,
        job_id=job.id,
        query=req.query,
        session_factory=_session_factory,
        inclusion_criteria=req.inclusion_criteria or None,
        exclusion_criteria=req.exclusion_criteria or None,
        wos_filter=req.wos_filter,
        inclusion_criteria_stages=req.inclusion_criteria_stages or None,
        exclusion_criteria_stages=req.exclusion_criteria_stages or None,
        queries_override=req.queries_override,
        publication_date_max=req.publication_date_max,
    )

    return TaskCreateResponse(task_id=job.id)

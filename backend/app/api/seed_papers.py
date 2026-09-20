"""API routes for seed paper resolution and expansion."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import JobType
from app.models.user import User
from app.schemas.seed_papers import SeedExpandRequest
from app.schemas.task import TaskCreateResponse
from app.services import project as project_service
from app.services import seed_papers as seed_papers_service
from app.services import task as task_service

router = APIRouter()
_session_factory = async_session_factory


@router.post(
    "/projects/{project_id}/seed-expand",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def seed_expand(
    project_id: UUID,
    req: SeedExpandRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start seed paper resolution and citation/reference expansion."""
    await project_service.get_project(db, project_id, user.id)

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.seed_expand,
        conflict_detail="Seed expansion already in progress",
    )

    job = await task_service.create_job(
        db, project_id, user.id, JobType.seed_expand
    )

    background_tasks.add_task(
        seed_papers_service.run_seed_expansion,
        project_id=project_id,
        job_id=job.id,
        session_factory=_session_factory,
        dois=req.dois,
        titles=req.titles,
    )

    return TaskCreateResponse(task_id=job.id)

"""API routes for field foundations — identifying foundational works."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import JobType
from app.models.user import User
from app.schemas.field_foundations import FieldFoundationsRequest
from app.schemas.task import TaskCreateResponse
from app.services import field_foundations as ff_service
from app.services import project as project_service
from app.services import task as task_service

router = APIRouter()
_session_factory = async_session_factory


@router.post(
    "/projects/{project_id}/field-foundations",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def field_foundations(
    project_id: UUID,
    req: FieldFoundationsRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start field foundations analysis to identify foundational works."""
    await project_service.get_project(db, project_id, user.id)

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.field_foundations,
        conflict_detail="Field foundations analysis already in progress",
    )

    job = await task_service.create_job(
        db, project_id, user.id, JobType.field_foundations
    )

    background_tasks.add_task(
        ff_service.run_field_foundations,
        project_id=project_id,
        job_id=job.id,
        session_factory=_session_factory,
        topic=req.topic,
        research_questions=req.research_questions,
    )

    return TaskCreateResponse(task_id=job.id)

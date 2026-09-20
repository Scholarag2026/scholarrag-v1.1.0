"""API routes for deep paper analysis."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import JobType
from app.models.user import User
from app.schemas.task import TaskCreateResponse
from app.services import project as project_service
from app.services import task as task_service
from app.services.deep_analysis import analyze_all_papers

router = APIRouter()
_session_factory = async_session_factory


# ---------------------------------------------------------------------------
# POST /projects/{project_id}/deep-analyze  (202)
# ---------------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/deep-analyze",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def deep_analyze_endpoint(
    project_id: UUID,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start deep analysis of all papers in the project library.

    Each paper is analyzed with the full-text agent (if full text is available)
    or with an abstract-only fallback. Results are stored in Paper.metadata_
    under the ``deep_analysis`` key. The operation is idempotent: papers that
    already have a ``deep_analysis`` entry are skipped.
    """
    await project_service.get_project(db, project_id, user.id)

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.deep_analysis,
        conflict_detail="Deep analysis already in progress",
    )

    job = await task_service.create_job(
        db, project_id, user.id, JobType.deep_analysis
    )

    background_tasks.add_task(
        analyze_all_papers,
        project_id=project_id,
        job_id=job.id,
        session_factory=_session_factory,
        expertise_level=user.expertise_level.value,
    )

    return TaskCreateResponse(task_id=job.id)

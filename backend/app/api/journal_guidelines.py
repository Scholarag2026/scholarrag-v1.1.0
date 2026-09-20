"""API endpoint for extracting and storing journal author guidelines."""

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import JobType
from app.models.user import User
from app.schemas.task import TaskCreateResponse
from app.services import project as project_service
from app.services import task as task_service
from app.services.journal_guidelines import run_fetch_journal_guidelines

router = APIRouter(prefix="/projects", tags=["journal-guidelines"])

# Module-level session factory — overridden in tests
_session_factory = async_session_factory


class FetchGuidelinesRequest(BaseModel):
    guidelines_text: str | None = None  # User-pasted guidelines text


@router.post(
    "/{project_id}/fetch-journal-guidelines",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def fetch_guidelines(
    project_id: UUID,
    req: FetchGuidelinesRequest = FetchGuidelinesRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a background job that extracts journal guidelines.

    Uses the user-pasted text when provided, otherwise falls back to a web search.
    Poll ``GET /tasks/{task_id}`` for the extracted guidelines.
    """
    project = await project_service.get_project(db, project_id, user.id)

    if not project.target_journal:
        raise HTTPException(status_code=400, detail="No target journal set")

    job = await task_service.create_job(
        db, project_id, user.id, JobType.journal_guidelines
    )

    background_tasks.add_task(
        run_fetch_journal_guidelines,
        project_id=project_id,
        guidelines_text=req.guidelines_text,
        job_id=job.id,
        session_factory=_session_factory,
    )

    return TaskCreateResponse(task_id=job.id)

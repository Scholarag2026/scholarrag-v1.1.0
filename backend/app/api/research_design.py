"""API routes for research design and data collection planning."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import JobType
from app.models.user import User
from app.schemas.research_design import ResearchDesignRequest
from app.schemas.task import TaskCreateResponse
from app.services import project as project_service
from app.services import research_design as rd_service
from app.services import task as task_service

router = APIRouter()
_session_factory = async_session_factory


@router.post(
    "/projects/{project_id}/research-design",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def generate_research_design(
    project_id: UUID,
    req: ResearchDesignRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.research_design,
        conflict_detail="Research design generation already in progress",
    )

    job = await task_service.create_job(
        db, project_id, user.id, JobType.research_design
    )

    background_tasks.add_task(
        rd_service.run_research_design,
        project_id=project_id,
        job_id=job.id,
        session_factory=_session_factory,
        research_question=req.research_question,
    )

    return TaskCreateResponse(task_id=job.id)


@router.get("/projects/{project_id}/research-design")
async def get_research_design(
    project_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)
    result = await rd_service.get_latest_research_design(db, project_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="No research design found"
        )
    return result


@router.post(
    "/projects/{project_id}/data-collection-plan",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def generate_collection_plan(
    project_id: UUID,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)

    # Precondition: research design must exist
    design = await rd_service.get_latest_research_design(db, project_id)
    if design is None:
        raise HTTPException(
            status_code=400,
            detail="Generate a research design first",
        )

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.data_collection,
        conflict_detail="Data collection plan generation already in progress",
    )

    job = await task_service.create_job(
        db, project_id, user.id, JobType.data_collection
    )

    background_tasks.add_task(
        rd_service.run_data_collection_plan,
        project_id=project_id,
        job_id=job.id,
        session_factory=_session_factory,
    )

    return TaskCreateResponse(task_id=job.id)


@router.get("/projects/{project_id}/data-collection-plan")
async def get_collection_plan(
    project_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)
    result = await rd_service.get_latest_collection_plan(db, project_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="No data collection plan found"
        )
    return result

"""API routes for quantitative data analysis."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory as _session_factory
from app.dependencies import get_current_user, get_db
from app.models.analysis_job import JobType
from app.models.user import User
from app.schemas.task import TaskCreateResponse
from app.services import dataset as dataset_service
from app.services import quantitative as quantitative_service
from app.services import task as task_service

router = APIRouter(tags=["quantitative"])


@router.post(
    "/datasets/{dataset_id}/analyze/quantitative",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def trigger_quantitative_analysis(
    dataset_id: UUID,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Trigger a quantitative analysis job for a dataset."""
    # Verify dataset ownership
    try:
        dataset = await dataset_service.get_dataset(db, dataset_id, user.id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=dataset.project_id,
        job_type=JobType.quantitative,
        dataset_id=dataset_id,
        conflict_detail="Quantitative analysis already in progress",
    )

    job = await task_service.create_job(
        db, dataset.project_id, user.id, JobType.quantitative, dataset_id=dataset_id
    )

    background_tasks.add_task(
        quantitative_service.run_quantitative_analysis,
        project_id=dataset.project_id,
        dataset_id=dataset_id,
        job_id=job.id,
        session_factory=_session_factory,
    )

    return TaskCreateResponse(task_id=job.id)


@router.get("/datasets/{dataset_id}/analysis/quantitative")
async def get_quantitative_analysis(
    dataset_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the latest completed quantitative analysis result for a dataset."""
    try:
        await dataset_service.get_dataset(db, dataset_id, user.id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    result = await quantitative_service.get_latest_quantitative(db, dataset_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="No quantitative analysis found"
        )
    return result

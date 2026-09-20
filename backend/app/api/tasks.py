from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.analysis_job import JobType
from app.models.user import User
from app.schemas.task import TaskResponse
from app.services import task as task_service
from app.services.claim_record import build_claim_record, claim_record_csv
from app.services.screening_record import build_screening_record, screening_record_csv

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await task_service.get_job(db, task_id, user.id)
    return TaskResponse.model_validate(job)


@router.delete("/{task_id}", response_model=TaskResponse)
async def cancel_task(
    task_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel a pending or running job.

    Returns the updated task with ``status == "cancelled"``. Long-running background
    loops observe the change via ``task_service.should_abort`` and exit at their next
    unit boundary; ``update_job_status`` refuses to move the job out of ``cancelled``.
    """
    job = await task_service.cancel_job(db, task_id, user.id)
    return TaskResponse.model_validate(job)


@router.get("/{task_id}/screening-record")
async def get_screening_record(
    task_id: UUID,
    fmt: Literal["json", "csv"] = Query("json", alias="format"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export the screening record of a Smart Search job.

    Owner-only, via the same lookup as ``GET /tasks/{task_id}``. 404 unless the job is a
    ``smart_search`` job with a stored result. ``format=json`` returns
    ``{criteria, flow, provenance, records}``; ``format=csv`` returns one row per record
    (included, excluded, unscreened) as a file attachment.
    """
    job = await task_service.get_job(db, task_id, user.id)
    if job.job_type != JobType.smart_search or not job.result:
        raise HTTPException(status_code=404, detail="No screening record for this task")
    if fmt == "csv":
        return Response(
            content=screening_record_csv(job.result),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="screening-record-{task_id}.csv"',
            },
        )
    return build_screening_record(job.result)


@router.get("/{task_id}/claim-record")
async def get_claim_record(
    task_id: UUID,
    fmt: Literal["json", "csv"] = Query("json", alias="format"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export the claim-verification record of a claim-verify job.

    Owner-only, via the same lookup as ``GET /tasks/{task_id}``. 404 unless the job is a
    ``claim_verify`` job with a stored result. ``format=json`` returns
    ``{provenance, full_text_coverage, counts, records}``; ``format=csv`` returns one row
    per claim as a file attachment.
    """
    job = await task_service.get_job(db, task_id, user.id)
    if job.job_type != JobType.claim_verify or not job.result:
        raise HTTPException(status_code=404, detail="No claim record for this task")
    if fmt == "csv":
        return Response(
            content=claim_record_csv(job.result),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="claim-record-{task_id}.csv"',
            },
        )
    return build_claim_record(job.result)

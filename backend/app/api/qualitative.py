"""API routes for qualitative coding analysis."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory as _session_factory
from app.dependencies import get_current_user, get_db
from app.models.analysis_job import JobType
from app.models.dataset import Codebook, CoderType, CodingSession, CodingStatus
from app.models.user import User
from app.schemas.qualitative import CodebookSchema, CodingSessionUpdate
from app.schemas.task import TaskCreateResponse
from app.services import dataset as dataset_service
from app.services import qualitative as qualitative_service
from app.services import task as task_service

router = APIRouter(tags=["qualitative"])


@router.post(
    "/datasets/{dataset_id}/analyze/qualitative",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def trigger_qualitative_analysis(
    dataset_id: UUID,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Trigger a qualitative coding job for a dataset."""
    try:
        dataset = await dataset_service.get_dataset(db, dataset_id, user.id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=dataset.project_id,
        job_type=JobType.qualitative,
        dataset_id=dataset_id,
        conflict_detail="Qualitative analysis already in progress",
    )

    job = await task_service.create_job(
        db, dataset.project_id, user.id, JobType.qualitative, dataset_id=dataset_id
    )

    background_tasks.add_task(
        qualitative_service.run_qualitative_coding,
        project_id=dataset.project_id,
        dataset_id=dataset_id,
        job_id=job.id,
        session_factory=_session_factory,
    )

    return TaskCreateResponse(task_id=job.id)


@router.get("/datasets/{dataset_id}/analysis/qualitative")
async def get_qualitative_analysis(
    dataset_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the latest completed qualitative analysis result for a dataset."""
    try:
        await dataset_service.get_dataset(db, dataset_id, user.id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    result = await qualitative_service.get_latest_coding_result(db, dataset_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="No qualitative analysis found"
        )
    return result


@router.put("/datasets/{dataset_id}/codebook")
async def update_codebook(
    dataset_id: UUID,
    codebook: CodebookSchema,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update or create the codebook for a dataset."""
    try:
        dataset = await dataset_service.get_dataset(db, dataset_id, user.id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check if codebook already exists
    existing_result = await db.execute(
        select(Codebook)
        .where(Codebook.dataset_id == dataset_id)
        .order_by(Codebook.created_at.desc())
        .limit(1)
    )
    existing_codebook = existing_result.scalar_one_or_none()

    if existing_codebook:
        existing_codebook.codes = [c.model_dump() for c in codebook.codes]
        existing_codebook.themes = [t.model_dump() for t in codebook.themes]
    else:
        new_codebook = Codebook(
            project_id=dataset.project_id,
            user_id=user.id,
            dataset_id=dataset_id,
            codes=[c.model_dump() for c in codebook.codes],
            themes=[t.model_dump() for t in codebook.themes],
        )
        db.add(new_codebook)

    await db.flush()
    return {"status": "ok"}


@router.put("/datasets/{dataset_id}/coding-session")
async def update_coding_session(
    dataset_id: UUID,
    update: CodingSessionUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update or create a human coding session for a dataset."""
    try:
        dataset = await dataset_service.get_dataset(db, dataset_id, user.id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Find the latest codebook
    codebook_result = await db.execute(
        select(Codebook)
        .where(Codebook.dataset_id == dataset_id)
        .order_by(Codebook.created_at.desc())
        .limit(1)
    )
    codebook = codebook_result.scalar_one_or_none()
    if not codebook:
        raise HTTPException(
            status_code=400,
            detail="No codebook found. Run qualitative analysis first.",
        )

    # Find or create human coding session
    session_result = await db.execute(
        select(CodingSession)
        .where(
            and_(
                CodingSession.codebook_id == codebook.id,
                CodingSession.coder_type == CoderType.human,
            )
        )
        .order_by(CodingSession.created_at.desc())
        .limit(1)
    )
    session = session_result.scalar_one_or_none()

    segments_data = [s.model_dump() for s in update.coded_segments]

    if session:
        session.coded_segments = segments_data
        session.status = CodingStatus.completed
        session.progress = 1.0
    else:
        session = CodingSession(
            user_id=user.id,
            codebook_id=codebook.id,
            coder_type=CoderType.human,
            coded_segments=segments_data,
            progress=1.0,
            status=CodingStatus.completed,
        )
        db.add(session)

    await db.flush()
    return {"status": "ok"}


@router.post(
    "/datasets/{dataset_id}/inter-coder-reliability",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def trigger_inter_coder_reliability(
    dataset_id: UUID,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Trigger inter-coder reliability computation."""
    try:
        dataset = await dataset_service.get_dataset(db, dataset_id, user.id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=dataset.project_id,
        job_type=JobType.inter_coder,
        dataset_id=dataset_id,
        conflict_detail="Inter-coder reliability analysis already in progress",
    )

    job = await task_service.create_job(
        db, dataset.project_id, user.id, JobType.inter_coder, dataset_id=dataset_id
    )

    background_tasks.add_task(
        qualitative_service.run_inter_coder_reliability,
        dataset_id=dataset_id,
        job_id=job.id,
        session_factory=_session_factory,
    )

    return TaskCreateResponse(task_id=job.id)


@router.get("/datasets/{dataset_id}/inter-coder-reliability")
async def get_inter_coder_reliability(
    dataset_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the latest completed inter-coder reliability result."""
    try:
        await dataset_service.get_dataset(db, dataset_id, user.id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    result = await qualitative_service.get_latest_inter_coder_result(db, dataset_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="No inter-coder reliability result found"
        )
    return result

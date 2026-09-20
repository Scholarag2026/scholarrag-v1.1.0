"""API routes for study quality scoring and gap analysis."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import JobType
from app.models.paper_analysis import PaperAnalysisRecord
from app.models.user import User
from app.schemas.analysis import (
    PaperAnalysisListResponse,
    PaperAnalysisResponse,
    QualityScoringRequest,
)
from app.schemas.task import TaskCreateResponse
from app.services import analysis as analysis_service
from app.services import project as project_service
from app.services import task as task_service

router = APIRouter()
_session_factory = async_session_factory


@router.post(
    "/projects/{project_id}/analyze-quality",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def analyze_quality(
    project_id: UUID,
    req: QualityScoringRequest = QualityScoringRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start quality scoring for all papers in the project."""
    # Ownership check
    await project_service.get_project(db, project_id, user.id)

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.quality_scoring,
        conflict_detail="Quality scoring already in progress",
    )

    job = await task_service.create_job(db, project_id, user.id, JobType.quality_scoring)

    background_tasks.add_task(
        analysis_service.run_quality_scoring,
        project_id=project_id,
        job_id=job.id,
        session_factory=_session_factory,
        force=req.force,
    )

    return TaskCreateResponse(task_id=job.id)


@router.post(
    "/projects/{project_id}/analyze-gaps",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def analyze_gaps(
    project_id: UUID,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start gap analysis for the project."""
    # Ownership check
    await project_service.get_project(db, project_id, user.id)

    # Check minimum analyzed papers
    count_result = await db.execute(
        select(func.count()).select_from(PaperAnalysisRecord).where(
            PaperAnalysisRecord.project_id == project_id
        )
    )
    count = count_result.scalar()

    if count < settings.gap_analysis_min_papers:
        raise HTTPException(
            status_code=400,
            detail=(
                f"At least {settings.gap_analysis_min_papers} analyzed papers "
                f"required for gap analysis, found {count}"
            ),
        )

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.gap_analysis,
        conflict_detail="Gap analysis already in progress",
    )

    job = await task_service.create_job(db, project_id, user.id, JobType.gap_analysis)

    background_tasks.add_task(
        analysis_service.run_gap_analysis,
        project_id=project_id,
        job_id=job.id,
        session_factory=_session_factory,
    )

    return TaskCreateResponse(task_id=job.id)


@router.get(
    "/projects/{project_id}/paper-analyses",
    response_model=PaperAnalysisListResponse,
)
async def list_paper_analyses(
    project_id: UUID,
    page: int = Query(1, ge=1),
    limit: int = Query(200, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all paper analyses for a project."""
    await project_service.get_project(db, project_id, user.id)

    records, total = await analysis_service.list_paper_analyses(
        db, project_id, page=page, limit=limit
    )

    analyses = [
        PaperAnalysisResponse(
            id=r.id,
            project_id=r.project_id,
            paper_id=r.paper_id,
            quality_score=r.quality_score,
            relevance_score=r.relevance_score,
            key_findings=r.key_findings,
            methodology=r.methodology,
            methodology_rigor=r.methodology_rigor,
            limitations=r.limitations,
            theories_used=r.theories_used,
            sample_info=r.sample_info,
            paper_title=r.paper.title if r.paper else "Unknown",
            paper_year=r.paper.year if r.paper else None,
            paper_doi=r.paper.doi if r.paper else None,
        )
        for r in records
    ]

    return PaperAnalysisListResponse(
        analyses=analyses, total=total, page=page, limit=limit
    )


@router.get("/projects/{project_id}/gap-report")
async def get_gap_report(
    project_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the latest gap analysis report for a project.

    Returns 200 with a ``null`` body when no report has been generated yet
    ("not generated" is not "not found"). A missing or
    unauthorised project still 404s, raised by ``get_project`` above.
    """
    await project_service.get_project(db, project_id, user.id)

    return await analysis_service.get_latest_gap_report(db, project_id)

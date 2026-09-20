"""API routes for citation graph visualization."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.models.user import User
from app.schemas.graph import ExpandNodeRequest, GraphData, GraphExpansion
from app.schemas.task import TaskCreateResponse
from app.services import graph as graph_service
from app.services import project as project_service
from app.services import task as task_service

logger = logging.getLogger(__name__)

router = APIRouter()
_session_factory = async_session_factory


def _is_stale(job: AnalysisJob, cutoff: datetime) -> bool:
    """True when a pending/running job has not reported progress since *cutoff*.

    AnalysisJob.updated_at is bumped by every per-paper progress write, so it doubles as
    the build heartbeat (decision D12d) — no schema change needed.
    """
    updated_at = job.updated_at
    if updated_at is None:
        return True
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return updated_at < cutoff


@router.post(
    "/projects/{project_id}/build-graph",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def build_graph(
    project_id: UUID,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start building citation graph for the project."""
    await project_service.get_project(db, project_id, user.id)

    # Concurrency guard. A build whose heartbeat stopped more than
    # settings.graph_stale_job_minutes ago is abandoned, not "in progress" — otherwise a
    # worker that died mid-build makes the project permanently un-rebuildable
    # (issue GRAPH-STALE-JOB-BLOCKS-REBUILD, decision D12c).
    cutoff = datetime.now(timezone.utc) - timedelta(
        minutes=settings.graph_stale_job_minutes
    )
    existing = await db.execute(
        select(AnalysisJob).where(
            and_(
                AnalysisJob.project_id == project_id,
                AnalysisJob.job_type == JobType.graph_building,
                AnalysisJob.status.in_(
                    [JobStatus.pending, JobStatus.running]
                ),
            )
        )
    )
    open_jobs = list(existing.scalars().all())
    if any(not _is_stale(job, cutoff) for job in open_jobs):
        raise HTTPException(
            status_code=409,
            detail="Graph building already in progress",
        )
    for job in open_jobs:
        logger.warning(
            "graph.stale_job_reaped job_id=%s project_id=%s", job.id, project_id
        )
        job.status = JobStatus.failed
        job.error = (
            "Abandoned: no progress for more than "
            f"{settings.graph_stale_job_minutes} minutes"
        )
    if open_jobs:
        await db.commit()

    job = await task_service.create_job(
        db, project_id, user.id, JobType.graph_building
    )

    background_tasks.add_task(
        graph_service.build_citation_graph,
        project_id=project_id,
        job_id=job.id,
        session_factory=_session_factory,
    )

    return TaskCreateResponse(task_id=job.id)


@router.get(
    "/projects/{project_id}/citation-graph",
    response_model=GraphData,
)
async def get_citation_graph(
    project_id: UUID,
    focus_id: UUID | None = Query(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get citation graph data for visualization."""
    await project_service.get_project(db, project_id, user.id)
    return await graph_service.get_graph_data(db, project_id, focus_id=focus_id)


@router.post(
    "/projects/{project_id}/citation-graph/expand",
    response_model=GraphExpansion,
)
async def expand_node(
    project_id: UUID,
    req: ExpandNodeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Expand a node to discover its references and citations."""
    await project_service.get_project(db, project_id, user.id)
    try:
        return await asyncio.wait_for(
            graph_service.expand_node(
                paper_id=req.paper_id,
                project_id=project_id,
                session_factory=_session_factory,
            ),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, detail="Node expansion timed out"
        )
    except httpx.HTTPError as exc:
        logger.warning(
            "graph.expand_upstream_failed project_id=%s paper_id=%s error=%s",
            project_id, req.paper_id, exc,
        )
        raise HTTPException(
            status_code=502,
            detail="Citation source unavailable — please try again",
        )

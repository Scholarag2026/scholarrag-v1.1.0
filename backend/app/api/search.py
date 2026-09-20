from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import JobStatus, JobType
from app.models.user import User
from app.schemas.search import SearchRequest
from app.schemas.task import TaskCreateResponse
from app.services import project as project_service
from app.services import task as task_service
from app.services.search import SearchService
from app.services.wos_import import batch_enrich_wos

router = APIRouter(tags=["search"])

# Module-level session factory — overridden in tests to use test DB
_session_factory = async_session_factory


async def _run_search_background(
    job_id: UUID,
    query: str,
    year_from: int | None,
    year_to: int | None,
    min_citations: int | None,
) -> None:
    """Background task that runs the actual search.

    Sessions are short-lived and scoped to writes only. A pooled connection is never held
    across the network wait (DB-SESSION-AND-POOL-PRESSURE) — the engine's 20+10 pool is
    shared with every inbound request, so holding one for the length of a search starves
    unrelated traffic.
    """
    try:
        async with _session_factory() as db:
            await task_service.update_job_status(
                db, job_id,
                status=JobStatus.running,
                progress=0.1,
                progress_message="Searching academic databases...",
            )

        svc = SearchService()
        papers, sources = await svc.search(
            query,
            year_from=year_from,
            year_to=year_to,
            min_citations=min_citations,
        )

        # Enrich with WoS collection data
        async with _session_factory() as db:
            await batch_enrich_wos(db, papers)

        async with _session_factory() as db:
            await task_service.update_job_status(
                db, job_id,
                status=JobStatus.completed,
                progress=1.0,
                progress_message=f"Found {len(papers)} papers",
                result={
                    "papers": [p.model_dump(mode="json") for p in papers],
                    "total": len(papers),
                    "sources": sources,
                },
            )
    except Exception as e:
        async with _session_factory() as err_db:
            await task_service.update_job_status(
                err_db, job_id,
                status=JobStatus.failed,
                error=str(e),
            )


@router.post("/projects/{project_id}/search", response_model=TaskCreateResponse)
async def search_papers(
    project_id: UUID,
    req: SearchRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Verify project ownership
    await project_service.get_project(db, project_id, user.id)

    # Create job record
    job = await task_service.create_job(db, project_id, user.id, JobType.search)

    # Kick off background search
    background_tasks.add_task(
        _run_search_background,
        job.id, req.query, req.year_from, req.year_to, req.min_citations,
    )

    return TaskCreateResponse(task_id=job.id)

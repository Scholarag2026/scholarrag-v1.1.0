import asyncio
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.clients.crossref import CrossRefClient
from app.config import settings
from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import JobStatus, JobType
from app.models.paper import Paper as PaperModel
from app.models.project_paper import ProjectPaper
from app.models.user import User
from app.schemas.task import TaskCreateResponse
from app.schemas.verification import VerifyReferencesRequest
from app.services import project as project_service
from app.services import task as task_service
from app.services.verification import verify_paper
from app.services.wos_import import is_wos_indexed_bulk

router = APIRouter(tags=["verification"])

# Module-level session factory — overridden in tests
_session_factory = async_session_factory


async def _run_verification_background(
    job_id: UUID,
    project_id: UUID,
    paper_ids: list[UUID],
    session_factory,
) -> None:
    """Background task that verifies references."""
    try:
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id,
                status=JobStatus.running,
                progress=0.1,
                progress_message="Loading papers...",
            )

        # Load papers
        async with session_factory() as db:
            query = (
                select(ProjectPaper)
                .options(joinedload(ProjectPaper.paper))
                .where(ProjectPaper.project_id == project_id)
            )
            if paper_ids:
                query = query.where(ProjectPaper.paper_id.in_(paper_ids))
            result = await db.execute(query)
            project_papers = list(result.scalars().unique().all())

        if not project_papers:
            async with session_factory() as db:
                await task_service.update_job_status(
                    db, job_id,
                    status=JobStatus.completed,
                    progress=1.0,
                    progress_message="No papers to verify",
                    result={"total": 0, "passed": 0, "failed": 0, "warnings": 0, "results": []},
                )
            return

        # ONE WoS query for the whole batch instead of one per paper (issue #36).
        async with session_factory() as db:
            wos_lookup = await is_wos_indexed_bulk(
                db, [pp.paper.journal_issn for pp in project_papers]
            )

        crossref = CrossRefClient()
        semaphore = asyncio.Semaphore(max(1, settings.verification_concurrency))
        progress_lock = asyncio.Lock()
        state = {"completed": 0, "cancelled": False}
        total_papers = len(project_papers)

        async def _record_progress() -> None:
            async with progress_lock:
                state["completed"] += 1
                done = state["completed"]
                every = max(1, settings.job_progress_update_every)
                if done % every and done != total_papers:
                    return
                async with session_factory() as db:
                    if await task_service.should_abort(db, job_id):
                        state["cancelled"] = True
                        return
                    await task_service.update_job_status(
                        db, job_id,
                        status=JobStatus.running,
                        progress=0.1 + (0.8 * done / total_papers),
                        progress_message=f"Verified {done}/{total_papers} references...",
                    )

        async def _verify_one(pp):
            paper = pp.paper
            wos_status = None  # None = no ISSN to check (preserves "skipped" semantics)
            wos_collection = None
            wos_categories = None
            if paper.journal_issn:
                wos_status, wos_collection, wos_categories = wos_lookup.get(
                    paper.journal_issn.strip().upper(), (False, None, None)
                )

            async with semaphore:
                if state["cancelled"]:
                    return None
                vr = await verify_paper(
                    paper_id=paper.id,
                    doi=paper.doi,
                    title=paper.title,
                    year=paper.year,
                    citation_count=paper.citation_count,
                    crossref_client=crossref,
                    is_wos_indexed=wos_status,
                    wos_collection=wos_collection,
                )
            await _record_progress()
            needs_write = wos_status is not None and paper.is_wos_indexed is None
            return vr.model_dump(mode="json"), (
                (paper.id, wos_status, wos_collection, wos_categories) if needs_write else None
            )

        gathered = await asyncio.gather(*(_verify_one(pp) for pp in project_papers))

        if state["cancelled"]:
            async with session_factory() as db:
                await task_service.update_job_status(
                    db, job_id,
                    status=JobStatus.cancelled,
                    progress=0.0,
                    progress_message="Reference verification cancelled.",
                )
            return

        results = [item[0] for item in gathered if item is not None]
        pending_writes = [item[1] for item in gathered if item is not None and item[1]]

        # One session for all WoS write-backs instead of one per paper.
        if pending_writes:
            async with session_factory() as db:
                for paper_id, wos_status, wos_collection, wos_categories in pending_writes:
                    await db.execute(
                        update(PaperModel)
                        .where(PaperModel.id == paper_id)
                        .values(
                            is_wos_indexed=wos_status,
                            wos_collection=wos_collection,
                            wos_categories=wos_categories,
                        )
                    )
                await db.commit()

        passed = sum(1 for r in results if r["overall_status"] == "pass")
        failed = sum(1 for r in results if r["overall_status"] == "fail")
        warnings = sum(1 for r in results if r["overall_status"] == "warning")

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id,
                status=JobStatus.completed,
                progress=1.0,
                progress_message=(
                    f"Verified {len(results)} papers: "
                    f"{passed} passed, {failed} failed, "
                    f"{warnings} warnings"
                ),
                result={
                    "total": len(results),
                    "passed": passed,
                    "failed": failed,
                    "warnings": warnings,
                    "results": results,
                },
            )

    except Exception as e:
        async with session_factory() as err_db:
            await task_service.update_job_status(
                err_db, job_id,
                status=JobStatus.failed,
                error=str(e),
            )


@router.post("/projects/{project_id}/verify-references", response_model=TaskCreateResponse)
async def verify_references(
    project_id: UUID,
    req: VerifyReferencesRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)
    job = await task_service.create_job(db, project_id, user.id, JobType.qa)

    background_tasks.add_task(
        _run_verification_background,
        job_id=job.id,
        project_id=project_id,
        paper_ids=req.paper_ids,
        session_factory=_session_factory,
    )

    return TaskCreateResponse(task_id=job.id)

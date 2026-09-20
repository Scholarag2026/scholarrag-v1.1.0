"""Analysis service — orchestrates quality scoring and gap analysis agents."""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.analysis_agent import (
    AnalysisDependencies,
    format_papers_for_prompt,
    get_analysis_agent,
)
from app.agents.gap_agent import format_analyses_for_gap_prompt, get_gap_agent
from app.config import settings
from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.models.paper import Paper
from app.models.paper_analysis import PaperAnalysisRecord
from app.models.project import Project
from app.models.project_paper import ProjectPaper
from app.schemas.analysis import PaperAnalysis
from app.services import task as task_service

logger = logging.getLogger(__name__)


def _split_into_batches(items: list, batch_size: int) -> list[list]:
    """Split a list into batches of given size."""
    if not items:
        return []
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _validate_paper_ids(
    analyses: list[PaperAnalysis], valid_ids: set[str]
) -> list[PaperAnalysis]:
    """Filter out analyses with paper_ids not in the input set."""
    valid = []
    for a in analyses:
        if a.paper_id in valid_ids:
            valid.append(a)
        else:
            logger.warning("Agent returned unknown paper_id=%s, discarding", a.paper_id)
    return valid


async def run_quality_scoring(
    project_id: UUID,
    job_id: UUID,
    session_factory,
    force: bool = False,
) -> None:
    """Background task: analyze all papers in batches using the Quality Scoring Agent."""
    try:
        async with session_factory() as db:
            # Load project for context
            project = await db.get(Project, project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            # Load all project papers with metadata
            result = await db.execute(
                select(ProjectPaper)
                .where(ProjectPaper.project_id == project_id)
                .options(selectinload(ProjectPaper.paper))
            )
            project_papers = result.scalars().all()

            if not project_papers:
                async with session_factory() as db2:
                    await task_service.update_job_status(
                        db2, job_id, JobStatus.completed,
                        progress=1.0, progress_message="No papers to analyze",
                        result={"analyzed": 0, "skipped": 0},
                    )
                return

            # Get already-analyzed paper IDs (skip unless force)
            existing_ids: set[UUID] = set()
            if not force:
                existing_result = await db.execute(
                    select(PaperAnalysisRecord.paper_id).where(
                        PaperAnalysisRecord.project_id == project_id
                    )
                )
                existing_ids = {row[0] for row in existing_result.all()}

        # Prepare paper dicts for the agent
        papers_to_analyze = []
        for pp in project_papers:
            if pp.paper_id in existing_ids:
                continue
            p = pp.paper
            papers_to_analyze.append({
                "id": str(p.id),
                "title": p.title,
                "authors": p.authors or [],
                "year": p.year,
                "journal_name": p.journal_name,
                "is_wos_indexed": p.is_wos_indexed,
                "citation_count": p.citation_count,
                "abstract": p.abstract,
            })

        if not papers_to_analyze:
            async with session_factory() as db:
                await task_service.update_job_status(
                    db, job_id, JobStatus.completed,
                    progress=1.0, progress_message="All papers already analyzed",
                    result={"analyzed": 0, "skipped": len(existing_ids)},
                )
            return

        # Update status to running
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.0,
                progress_message=f"Analyzing {len(papers_to_analyze)} papers...",
            )

        # Process in batches
        batches = _split_into_batches(papers_to_analyze, settings.analysis_batch_size)
        total_papers = len(papers_to_analyze)
        analyzed = 0
        failed_batches = 0

        deps = AnalysisDependencies(
            project_id=str(project_id),
            project_description=project.description if project else None,
            target_journal=project.target_journal if project else None,
            citation_style=project.citation_style.value if project else "APA",
        )

        semaphore = asyncio.Semaphore(max(1, settings.analysis_concurrency))
        progress_lock = asyncio.Lock()
        state = {"completed": 0, "cancelled": False}

        async def _score_batch(batch_idx: int, batch: list[dict]):
            """Run one batch through the agent. Returns (batch_idx, analyses) or None."""
            async with semaphore:
                if state["cancelled"]:
                    return None
                try:
                    prompt = format_papers_for_prompt(batch)
                    batch_ids = {p["id"] for p in batch}
                    result = await get_analysis_agent().run(prompt, deps=deps)
                    return batch_idx, _validate_paper_ids(result.output.analyses, batch_ids)
                except Exception:
                    logger.exception("Batch %d failed for project %s", batch_idx, project_id)
                    return None
                finally:
                    async with progress_lock:
                        state["completed"] += 1
                        done = state["completed"]
                        every = max(1, settings.job_progress_update_every)
                        if done % every == 0 or done == len(batches):
                            async with session_factory() as db:
                                if await task_service.should_abort(db, job_id):
                                    state["cancelled"] = True
                                else:
                                    await task_service.update_job_status(
                                        db, job_id, JobStatus.running,
                                        progress=min(done / len(batches), 1.0),
                                        progress_message=(
                                            f"Scored {done}/{len(batches)} batches..."
                                        ),
                                    )

        batch_results = await asyncio.gather(
            *(_score_batch(i, batch) for i, batch in enumerate(batches))
        )

        if state["cancelled"]:
            async with session_factory() as db:
                await task_service.update_job_status(
                    db, job_id, JobStatus.cancelled,
                    progress=0.0,
                    progress_message="Quality scoring cancelled.",
                )
            return

        # Persist sequentially — one session, deterministic order, no write contention.
        for item in batch_results:
            if item is None:
                failed_batches += 1
                continue
            _batch_idx, valid_analyses = item
            async with session_factory() as db:
                for pa in valid_analyses:
                    existing = await db.execute(
                        select(PaperAnalysisRecord).where(
                            and_(
                                PaperAnalysisRecord.project_id == project_id,
                                PaperAnalysisRecord.paper_id == UUID(pa.paper_id),
                            )
                        )
                    )
                    existing_record = existing.scalar_one_or_none()
                    if existing_record:
                        await db.delete(existing_record)
                        await db.flush()

                    record = PaperAnalysisRecord(
                        project_id=project_id,
                        paper_id=UUID(pa.paper_id),
                        relevance_score=pa.relevance_score,
                        quality_score=pa.quality_score,
                        key_findings=pa.key_findings,
                        methodology=pa.methodology,
                        methodology_rigor=pa.methodology_rigor,
                        limitations=pa.limitations,
                        theories_used=pa.theories_used,
                        sample_info=pa.sample_info,
                    )
                    db.add(record)

                    paper = await db.get(Paper, UUID(pa.paper_id))
                    if paper:
                        paper.quality_score = pa.quality_score

                await db.commit()
            analyzed += len(valid_analyses)

        # Complete
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.completed,
                progress=1.0,
                progress_message=f"Completed: {analyzed} analyzed, {failed_batches} batches failed",
                result={
                    "analyzed": analyzed,
                    "total": total_papers,
                    "failed_batches": failed_batches,
                },
            )

    except Exception as e:
        logger.exception("Quality scoring failed for project %s", project_id)
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.failed,
                error=str(e),
            )


async def run_gap_analysis(
    project_id: UUID,
    job_id: UUID,
    session_factory,
) -> None:
    """Background task: generate gap report from paper analyses."""
    try:
        async with session_factory() as db:
            # Load project
            project = await db.get(Project, project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            # Load all paper analyses with paper metadata
            result = await db.execute(
                select(PaperAnalysisRecord)
                .where(PaperAnalysisRecord.project_id == project_id)
                .options(selectinload(PaperAnalysisRecord.paper))
            )
            records = result.scalars().all()

        if len(records) < settings.gap_analysis_min_papers:
            async with session_factory() as db:
                await task_service.update_job_status(
                    db, job_id, JobStatus.failed,
                    error=(
                        f"Need at least {settings.gap_analysis_min_papers} "
                        f"analyzed papers, found {len(records)}"
                    ),
                )
            return

        # Cap at max papers (sorted by quality_score desc)
        records_sorted = sorted(records, key=lambda r: r.quality_score, reverse=True)
        if len(records_sorted) > settings.gap_analysis_max_papers:
            logger.warning(
                "Project %s has %d analyses, capping at %d",
                project_id, len(records_sorted), settings.gap_analysis_max_papers,
            )
            records_sorted = records_sorted[: settings.gap_analysis_max_papers]

        # Update to running
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.1,
                progress_message=f"Analyzing gaps across {len(records_sorted)} papers...",
            )

        # Format analyses for prompt
        analyses_data = []
        for r in records_sorted:
            analyses_data.append({
                "paper_title": r.paper.title if r.paper else "Unknown",
                "paper_year": r.paper.year if r.paper else None,
                "quality_score": r.quality_score,
                "key_findings": r.key_findings,
                "methodology": r.methodology,
                "methodology_rigor": r.methodology_rigor,
                "limitations": r.limitations,
                "theories_used": r.theories_used,
            })

        prompt = format_analyses_for_gap_prompt(
            analyses_data,
            project_description=project.description,
            target_journal=project.target_journal,
        )

        deps = AnalysisDependencies(
            project_id=str(project_id),
            project_description=project.description,
            target_journal=project.target_journal,
            citation_style=project.citation_style.value if project else "APA",
        )

        result = await get_gap_agent().run(prompt, deps=deps)
        gap_report = result.output

        # Store result
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.completed,
                progress=1.0,
                progress_message="Gap analysis complete",
                result=gap_report.model_dump(),
            )

    except Exception as e:
        logger.exception("Gap analysis failed for project %s", project_id)
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.failed,
                error=str(e),
            )


async def list_paper_analyses(
    db: AsyncSession, project_id: UUID, page: int = 1, limit: int = 50
) -> tuple[list[PaperAnalysisRecord], int]:
    """List paper analyses for a project with paper metadata, paginated."""
    count_result = await db.execute(
        select(func.count())
        .select_from(PaperAnalysisRecord)
        .where(PaperAnalysisRecord.project_id == project_id)
    )
    total = count_result.scalar() or 0

    result = await db.execute(
        select(PaperAnalysisRecord)
        .where(PaperAnalysisRecord.project_id == project_id)
        .options(selectinload(PaperAnalysisRecord.paper))
        .order_by(PaperAnalysisRecord.quality_score.desc(), PaperAnalysisRecord.id)
        .offset((page - 1) * limit)
        .limit(limit)
    )
    return list(result.scalars().all()), total


async def get_latest_gap_report(db: AsyncSession, project_id: UUID) -> dict | None:
    """Get the most recent completed gap analysis result for a project."""
    result = await db.execute(
        select(AnalysisJob)
        .where(
            and_(
                AnalysisJob.project_id == project_id,
                AnalysisJob.job_type == JobType.gap_analysis,
                AnalysisJob.status == JobStatus.completed,
            )
        )
        .order_by(AnalysisJob.created_at.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    return job.result if job else None

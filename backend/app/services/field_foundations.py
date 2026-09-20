"""Field Foundations service — foundational works identification and verification."""

from __future__ import annotations

import asyncio
import logging
import re
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.field_foundations_agent import (
    format_foundations_prompt,
    get_field_foundations_agent,
)
from app.clients.openalex import OpenAlexClient
from app.config import settings
from app.models.analysis_job import JobStatus
from app.models.paper import Paper
from app.models.project_paper import ProjectPaper
from app.schemas.field_foundations import FoundationalWork
from app.services import task as task_service

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Normalize a string for fuzzy title matching: lowercase, strip punctuation."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _titles_match(a: str, b: str) -> bool:
    """Check whether two titles are similar enough to consider a match."""
    na = _normalize(a)
    nb = _normalize(b)
    if not na or not nb:
        return False
    # Exact normalized match
    if na == nb:
        return True
    # One contains the other (handles subtitle differences)
    if na in nb or nb in na:
        return len(min(na, nb, key=len)) / len(max(na, nb, key=len)) > 0.7
    return False


async def _get_existing_titles(db: AsyncSession, project_id: UUID) -> list[str]:
    """Get titles of papers already in the project's library."""
    result = await db.execute(
        select(Paper.title)
        .join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
        .where(ProjectPaper.project_id == project_id)
    )
    return [row[0] for row in result.all()]


async def _verify_work(
    work: FoundationalWork,
) -> FoundationalWork:
    """Try to verify a suggested foundational work via OpenAlex.

    Searches by title, then retries once qualified with the first author name (which is
    what the removed Semantic Scholar leg added for precision). If a match is found, the
    work is marked verified with the matched PaperData attached.
    """
    oa = OpenAlexClient()
    first_author = work.suggested_authors[0] if work.suggested_authors else ""

    attempts = [work.suggested_title]
    if first_author:
        attempts.append(f"{work.suggested_title} {first_author}")

    for query in attempts:
        try:
            results = await oa.search(query, per_page=5)
        except Exception:
            logger.warning("OpenAlex search failed for: %s", query)
            continue
        for paper in results:
            if _titles_match(paper.title, work.suggested_title):
                return work.model_copy(
                    update={"verified": True, "matched_paper": paper}
                )

    # Could not verify
    return work


async def run_field_foundations(
    project_id: UUID,
    job_id: UUID,
    session_factory,
    topic: str,
    research_questions: list[str],
) -> None:
    """Background task: identify and verify foundational works for a field.

    Steps:
    1. Gather existing paper titles from library for context
    2. Run field foundations agent to get 20-30 suggested works
    3. For each suggested work, search OpenAlex + S2 by title + first author
    4. Mark as verified/unverified based on match
    5. Store result in job
    """
    async with session_factory() as session:
        await task_service.update_job_status(
            session, job_id, JobStatus.running,
            progress=0.0,
            progress_message="Starting field foundations analysis...",
        )

    try:
        # 1. Gather existing titles
        async with session_factory() as session:
            existing_titles = await _get_existing_titles(session, project_id)

        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, JobStatus.running,
                progress=0.1,
                progress_message="Identifying foundational works...",
            )

        # 2. Run agent
        agent = get_field_foundations_agent()
        deps = AnalysisDependencies(
            project_id=str(project_id),
            project_description="",
            target_journal="",
            citation_style="",
        )
        prompt = format_foundations_prompt(topic, research_questions, existing_titles)
        result = await agent.run(prompt, deps=deps)
        suggested_works = result.output.works

        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, JobStatus.running,
                progress=0.4,
                progress_message=f"Verifying {len(suggested_works)} suggested works...",
            )

        # 3 & 4. Verify the suggested works with bounded concurrency (issue #29).
        total = len(suggested_works)
        semaphore = asyncio.Semaphore(max(1, settings.analysis_concurrency))
        progress_lock = asyncio.Lock()
        state = {"completed": 0, "cancelled": False}

        async def _record_progress() -> None:
            async with progress_lock:
                state["completed"] += 1
                done = state["completed"]
                every = max(1, settings.job_progress_update_every)
                if done % every and done != total:
                    return
                async with session_factory() as session:
                    if await task_service.should_abort(session, job_id):
                        state["cancelled"] = True
                        return
                    await task_service.update_job_status(
                        session, job_id, JobStatus.running,
                        progress=0.4 + 0.55 * (done / total if total else 1.0),
                        progress_message=f"Verified {done}/{total} works",
                    )

        async def _verify_one(work: FoundationalWork) -> FoundationalWork:
            async with semaphore:
                if state["cancelled"]:
                    return work
                try:
                    verified = await _verify_work(work)
                except Exception:
                    logger.warning("Verification failed for: %s", work.suggested_title)
                    verified = work
            await _record_progress()
            return verified

        verified_works: list[FoundationalWork] = list(
            await asyncio.gather(*(_verify_one(w) for w in suggested_works))
        )

        if state["cancelled"]:
            async with session_factory() as session:
                await task_service.update_job_status(
                    session, job_id, JobStatus.cancelled,
                    progress=0.0,
                    progress_message="Field foundations cancelled.",
                )
            return

        # 5. Store result
        verified_count = sum(1 for w in verified_works if w.verified)

        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, JobStatus.completed,
                progress=1.0,
                progress_message=(
                    f"Field foundations complete: {verified_count}/{total} verified"
                ),
                result={
                    "field": topic,
                    "works": [w.model_dump() for w in verified_works],
                    "verified_count": verified_count,
                    "total_count": total,
                },
            )

    except Exception as e:
        logger.exception(
            "Field foundations failed for project %s", project_id
        )
        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, JobStatus.failed, error=str(e),
            )

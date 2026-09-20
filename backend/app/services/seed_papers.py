"""Seed paper resolution and citation/reference expansion service (OpenAlex only)."""

from __future__ import annotations

import asyncio
import logging
import re
from uuid import UUID

from app.clients.openalex import OpenAlexClient, to_paper_data
from app.config import settings
from app.models.analysis_job import JobStatus
from app.schemas.paper import PaperData
from app.services import task as task_service
from app.services.openalex_relations import fetch_relations, resolve_work_id
from app.services.wos_import import batch_enrich_wos

logger = logging.getLogger(__name__)


def normalize_title(title: str) -> str:
    """Normalize title for dedup: lowercase, strip punctuation."""
    return re.sub(r"[^\w\s]", "", title.lower()).strip()


async def resolve_seed(doi_or_title: str, *, is_doi: bool) -> PaperData | None:
    """Resolve a single seed to a PaperData object via OpenAlex.

    Strategy:
    - DOI seeds are an exact record lookup (``/works/doi:10.x/y``); if OpenAlex has no
      record (404 — happens for very old CNKI DOIs) we fall back to a keyword search.
    - Title seeds go straight to keyword search.

    Returns the first matching PaperData or None. Never raises.
    """
    oa = OpenAlexClient()
    query = doi_or_title.strip()

    if is_doi:
        try:
            work = await oa.get_work(query)
        except Exception:
            logger.warning("OpenAlex get_work failed for seed: %s", query)
            work = None
        if work:
            return to_paper_data(work)

    try:
        results = await oa.search(query, per_page=3)
    except Exception:
        logger.warning("OpenAlex search failed for seed: %s", query)
        return None

    if not results:
        return None
    if is_doi:
        for paper in results:
            if paper.doi and paper.doi.lower() == query.lower():
                return paper
    return results[0]


def _dedup_papers(
    papers: list[PaperData],
    seen_dois: set[str],
    seen_titles: set[str],
) -> list[PaperData]:
    """Deduplicate papers against already-seen DOIs and normalized titles.

    Returns only new papers and updates seen_dois / seen_titles in-place.
    """
    unique: list[PaperData] = []
    for p in papers:
        doi_lower = p.doi.lower() if p.doi else None
        norm_title = normalize_title(p.title)

        if doi_lower and doi_lower in seen_dois:
            continue
        if norm_title in seen_titles:
            continue

        if doi_lower:
            seen_dois.add(doi_lower)
        seen_titles.add(norm_title)
        unique.append(p)

    return unique


async def run_seed_expansion(
    project_id: UUID,
    job_id: UUID,
    session_factory,
    dois: list[str],
    titles: list[str],
) -> None:
    """Background task: resolve seed papers, then expand via references + citations.

    Parameters
    ----------
    project_id : UUID
        The project this expansion belongs to.
    job_id : UUID
        The analysis job tracking progress.
    session_factory : callable
        Async context manager that yields DB sessions.
    dois : list[str]
        DOIs to resolve as seeds.
    titles : list[str]
        Titles to resolve as seeds.
    """
    async with session_factory() as session:
        await task_service.update_job_status(
            session, job_id, JobStatus.running,
            progress=0.0,
            progress_message="Resolving seeds...",
        )

    try:
        # 1. Resolve each DOI/title to a paper
        seeds_to_resolve: list[tuple[str, bool]] = []
        for doi in dois:
            seeds_to_resolve.append((doi, True))
        for title in titles:
            seeds_to_resolve.append((title, False))

        resolved_papers: list[PaperData] = []
        seeds_failed: list[str] = []
        total_seeds = len(seeds_to_resolve)

        for i, (query, is_doi) in enumerate(seeds_to_resolve):
            try:
                paper = await resolve_seed(query, is_doi=is_doi)
                if paper:
                    resolved_papers.append(paper)
                else:
                    seeds_failed.append(query)
            except Exception:
                logger.exception("Failed to resolve seed: %s", query)
                seeds_failed.append(query)

            progress = 0.3 * ((i + 1) / total_seeds)
            async with session_factory() as session:
                await task_service.update_job_status(
                    session, job_id, JobStatus.running,
                    progress=progress,
                    progress_message=f"Resolved {i + 1}/{total_seeds} seeds",
                )

        if not resolved_papers:
            async with session_factory() as session:
                await task_service.update_job_status(
                    session, job_id, JobStatus.completed,
                    progress=1.0,
                    progress_message="No seeds could be resolved",
                    result={
                        "seeds_resolved": 0,
                        "seeds_failed": seeds_failed,
                        "expanded_papers": [],
                        "total_expanded": 0,
                    },
                )
            return

        # 2. Expand every resolved seed through OpenAlex, bounded and concurrent.
        oa = OpenAlexClient()
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()

        # Add resolved seeds themselves to the seen sets to avoid returning them
        for p in resolved_papers:
            if p.doi:
                seen_dois.add(p.doi.lower())
            seen_titles.add(normalize_title(p.title))

        all_expanded: list[PaperData] = []
        total_resolved = len(resolved_papers)
        semaphore = asyncio.Semaphore(settings.graph_build_concurrency)
        counter_lock = asyncio.Lock()
        completed = 0

        async def _expand(seed: PaperData) -> list[PaperData]:
            nonlocal completed
            # external_id is an OpenAlex W-id; sending it to any other provider is the
            # bug this rewrite removes (issue SEED-OPENALEX-ID-TO-S2).
            work_id = resolve_work_id(seed.external_id, seed.doi)
            found: list[PaperData] = []
            if work_id is None:
                logger.warning("Seed has no OpenAlex identifier: %s", seed.title)
            else:
                async with semaphore:
                    try:
                        fetched = await fetch_relations(
                            oa,
                            work_id,
                            max_refs=settings.graph_max_refs_per_paper,
                            max_cites=settings.graph_max_cites_per_paper,
                        )
                    except Exception:
                        logger.warning("OpenAlex expansion failed for: %s", work_id)
                        fetched = None
                if fetched is not None:
                    # fetched[1] is None when the cites: query is rate-limited (refs
                    # are still trustworthy) — expand from whatever we have.
                    found = [to_paper_data(w) for w in fetched[0] + (fetched[1] or [])]

            async with counter_lock:
                completed += 1
                done = completed

            async with session_factory() as session:
                await task_service.update_job_status(
                    session, job_id, JobStatus.running,
                    progress=0.3 + 0.7 * (done / total_resolved),
                    progress_message=f"Expanded {done}/{total_resolved} seeds",
                )
            return found

        per_seed = await asyncio.gather(*(_expand(s) for s in resolved_papers))
        for candidates in per_seed:
            all_expanded.extend(_dedup_papers(candidates, seen_dois, seen_titles))

        # Enrich with WoS collection data
        async with session_factory() as wos_db:
            await batch_enrich_wos(wos_db, all_expanded)

        # 4. Store result
        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, JobStatus.completed,
                progress=1.0,
                progress_message=(
                    f"Seed expansion complete: {len(all_expanded)} papers "
                    f"from {total_resolved} seeds"
                ),
                result={
                    "seeds_resolved": total_resolved,
                    "seeds_failed": seeds_failed,
                    "expanded_papers": [p.model_dump() for p in all_expanded],
                    "total_expanded": len(all_expanded),
                },
            )

    except Exception as e:
        logger.exception("Seed expansion failed for project %s", project_id)
        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, JobStatus.failed, error=str(e),
            )

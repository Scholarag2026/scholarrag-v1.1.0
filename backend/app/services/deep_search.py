"""Multi-round iterative deepening search pipeline."""

from __future__ import annotations

import logging
import re
import time
from uuid import UUID

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.query_expansion_agent import format_expansion_prompt, get_query_expansion_agent
from app.config import settings
from app.models.analysis_job import JobStatus
from app.schemas.deep_search import CoverageMetrics, RoundMetrics
from app.schemas.paper import PaperData
from app.services import task as task_service
from app.services.search import SearchService, gather_search_results
from app.services.wos_import import batch_enrich_wos

logger = logging.getLogger(__name__)


def normalize_title(title: str) -> str:
    """Normalize title for dedup: lowercase, strip punctuation."""
    return re.sub(r"[^\w\s]", "", title.lower()).strip()


class DeepSearchPipeline:
    """Orchestrates multi-round iterative deepening search with deduplication."""

    def __init__(self) -> None:
        self.seen_dois: set[str] = set()
        self.seen_titles: set[str] = set()
        self.all_papers: list[PaperData] = []
        self.round_metrics: list[RoundMetrics] = []
        self.total_scanned: int = 0
        self.sources: dict[str, int] = {"openalex": 0}

    def _is_new_paper(self, paper: PaperData) -> bool:
        """Check if a paper is new (not seen before) via DOI or normalized title."""
        if paper.doi and paper.doi.lower() in self.seen_dois:
            return False
        norm_title = normalize_title(paper.title)
        if norm_title in self.seen_titles:
            return False
        if paper.doi:
            self.seen_dois.add(paper.doi.lower())
        self.seen_titles.add(norm_title)
        return True

    def _add_papers(self, papers: list[PaperData]) -> int:
        """Add papers to the pipeline, deduplicating. Returns count of new papers added."""
        new_count = 0
        for p in papers:
            self.total_scanned += 1
            if self._is_new_paper(p):
                self.all_papers.append(p)
                source = p.source_api or "unknown"
                self.sources[source] = self.sources.get(source, 0) + 1
                new_count += 1
        return new_count

    def build_coverage_metrics(self) -> CoverageMetrics:
        """Build final coverage metrics from accumulated pipeline state."""
        total_unique = len(self.all_papers)
        yield_curve = [rm.new_papers for rm in self.round_metrics]

        # Determine confidence based on total unique papers found
        if total_unique >= 30:
            confidence = "high"
        elif total_unique >= 10:
            confidence = "medium"
        else:
            confidence = "low"

        return CoverageMetrics(
            total_scanned=self.total_scanned,
            total_unique=total_unique,
            rounds=self.round_metrics,
            sources=self.sources,
            yield_curve=yield_curve,
            confidence=confidence,
        )


async def run_deep_search(
    project_id: UUID,
    job_id: UUID,
    query: str,
    filters: dict,
    session_factory,
) -> None:
    """Background task: run multi-round iterative deep search.

    Parameters
    ----------
    project_id : UUID
        The project this search belongs to.
    job_id : UUID
        The analysis job tracking progress.
    query : str
        The user's original search query.
    filters : dict
        Search filters (year_from, year_to, min_citations, max_rounds).
    session_factory : callable
        Async context manager that yields DB sessions.
    """
    year_from = filters.get("year_from")
    year_to = filters.get("year_to")
    min_citations = filters.get("min_citations")
    max_rounds = min(filters.get("max_rounds", settings.deep_search_max_rounds),
                     settings.deep_search_max_rounds)

    start_time = time.monotonic()
    time_budget_seconds = settings.deep_search_max_time_minutes * 60.0
    stop_reason: str | None = None
    cancelled = False

    def _budget_exhausted() -> bool:
        return (time.monotonic() - start_time) >= time_budget_seconds

    async def _cancelled() -> bool:
        async with session_factory() as abort_session:
            return await task_service.should_abort(abort_session, job_id)

    async with session_factory() as session:
        await task_service.update_job_status(
            session, job_id, JobStatus.running,
            progress=0.0,
            progress_message="Starting deep search...",
        )

    try:
        pipeline = DeepSearchPipeline()
        search_svc = SearchService()

        async def _stop_expanded() -> bool:
            """Consulted before every individual expanded query."""
            nonlocal stop_reason, cancelled
            if _budget_exhausted():
                stop_reason = "time_limit"
                return True
            if pipeline.total_scanned >= settings.deep_search_max_scanned:
                stop_reason = "max_scanned"
                return True
            if await _cancelled():
                stop_reason = "cancelled"
                cancelled = True
                return True
            return False

        # ---- Round 1: Initial search ----
        papers, _sources = await search_svc.search(
            query,
            year_from=year_from,
            year_to=year_to,
            min_citations=min_citations,
        )
        new_count = pipeline._add_papers(papers)

        pipeline.round_metrics.append(RoundMetrics(
            round=1,
            strategy="initial_search",
            new_papers=new_count,
            queries=[query],
        ))

        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, JobStatus.running,
                progress=1.0 / max_rounds,
                progress_message=(
                    f"Round 1 complete: {new_count} papers found"
                ),
            )

        # ---- Rounds 2-N: Query expansion ----
        consecutive_low_yield = 0

        for round_num in range(2, max_rounds + 1):
            # Check stopping criteria: cancellation
            if await _cancelled():
                cancelled = True
                stop_reason = "cancelled"
                logger.info("Deep search %s cancelled before round %d", job_id, round_num)
                pipeline.round_metrics.append(RoundMetrics(
                    round=round_num,
                    strategy="query_expansion",
                    new_papers=0,
                    stopped=True,
                ))
                break

            # Check stopping criteria: wall-clock budget
            if _budget_exhausted():
                stop_reason = "time_limit"
                logger.info(
                    "Deep search stopped: time budget (%.1f min) exhausted",
                    settings.deep_search_max_time_minutes,
                )
                pipeline.round_metrics.append(RoundMetrics(
                    round=round_num,
                    strategy="query_expansion",
                    new_papers=0,
                    stopped=True,
                ))
                break

            # Check stopping criteria: max scanned
            if pipeline.total_scanned >= settings.deep_search_max_scanned:
                stop_reason = "max_scanned"
                logger.info(
                    "Deep search stopped: max scanned (%d) reached",
                    settings.deep_search_max_scanned,
                )
                pipeline.round_metrics.append(RoundMetrics(
                    round=round_num,
                    strategy="query_expansion",
                    new_papers=0,
                    stopped=True,
                ))
                break

            # Run query expansion agent
            try:
                agent = get_query_expansion_agent()
                deps = AnalysisDependencies(
                    project_id=project_id,
                    project_description="",
                    target_journal="",
                    citation_style="",
                )
                paper_titles = [p.title for p in pipeline.all_papers[:10]]
                paper_abstracts = [p.abstract for p in pipeline.all_papers[:10]]
                prompt = format_expansion_prompt(query, paper_titles, paper_abstracts)
                expansion_result = await agent.run(prompt, deps=deps)
                expanded_queries = expansion_result.output.queries
            except Exception:
                logger.exception("Query expansion failed at round %d", round_num)
                expanded_queries = []

            if not expanded_queries:
                stop_reason = stop_reason or "no_expanded_queries"
                pipeline.round_metrics.append(RoundMetrics(
                    round=round_num,
                    strategy="query_expansion",
                    new_papers=0,
                    queries=[],
                    stopped=True,
                ))
                break

            # Search with each expanded query, bounded-concurrently
            query_results, _stopped_early, _failures = await gather_search_results(
                search_svc,
                expanded_queries,
                year_from=year_from,
                year_to=year_to,
                min_citations=min_citations,
                stop_check=_stop_expanded,
            )
            round_new = 0
            for _eq, eq_papers in query_results:
                round_new += pipeline._add_papers(eq_papers)

            pipeline.round_metrics.append(RoundMetrics(
                round=round_num,
                strategy="query_expansion",
                new_papers=round_new,
                queries=expanded_queries,
                stopped=cancelled or stop_reason == "time_limit",
            ))

            if cancelled or stop_reason == "time_limit":
                break

            # Check yield-based stopping
            if round_new < settings.deep_search_yield_threshold:
                consecutive_low_yield += 1
            else:
                consecutive_low_yield = 0

            if consecutive_low_yield >= 2:
                stop_reason = "low_yield"
                logger.info(
                    "Deep search stopped: low yield for %d consecutive rounds",
                    consecutive_low_yield,
                )
                # Mark the last round as stopped
                pipeline.round_metrics[-1].stopped = True
                break

            # Update progress
            progress = round_num / max_rounds
            async with session_factory() as session:
                await task_service.update_job_status(
                    session, job_id, JobStatus.running,
                    progress=progress,
                    progress_message=(
                        f"Round {round_num} complete: {round_new} new papers"
                        f" ({len(pipeline.all_papers)} total unique)"
                    ),
                )

        # ---- Build final metrics and complete ----
        coverage = pipeline.build_coverage_metrics()

        # Enrich with WoS collection data
        async with session_factory() as wos_db:
            await batch_enrich_wos(wos_db, pipeline.all_papers)

        final_status = JobStatus.cancelled if cancelled else JobStatus.completed
        headline = "Deep search cancelled" if cancelled else "Deep search complete"
        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, final_status,
                progress=1.0,
                progress_message=(
                    f"{headline}: {coverage.total_unique} unique papers"
                    f" from {coverage.total_scanned} scanned"
                ),
                result={
                    "papers": [p.model_dump() for p in pipeline.all_papers],
                    "coverage": coverage.model_dump(),
                    "stop_reason": stop_reason,
                    "elapsed_minutes": round((time.monotonic() - start_time) / 60.0, 2),
                },
            )

        logger.info(
            "Deep search %s %s: %d unique / %d scanned (stop_reason=%s)",
            job_id, final_status.value, coverage.total_unique,
            coverage.total_scanned, stop_reason,
        )

    except Exception as e:
        logger.exception("Deep search failed for project %s", project_id)
        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, JobStatus.failed, error=str(e),
            )

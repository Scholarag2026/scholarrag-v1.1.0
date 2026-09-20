"""Graph service — builds and serves citation graph data (OpenAlex only)."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from uuid import UUID

import networkx as nx
from sqlalchemy import and_, delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.clients.circuit_breaker import CircuitBreaker
from app.clients.openalex import OpenAlexClient, filter_queries_blocked, to_paper_data
from app.config import settings
from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.models.citation_edge import CitationEdge
from app.models.paper import Paper
from app.models.project_paper import ProjectPaper
from app.schemas.graph import (
    GraphBuildSummary,
    GraphData,
    GraphEdge,
    GraphExpansion,
    GraphNode,
)
from app.services import paper as paper_service
from app.services import task as task_service
from app.services.openalex_relations import cap_works, fetch_relations, resolve_work_id

logger = logging.getLogger(__name__)


def _paper_to_graph_node(paper, in_library: bool, is_seed: bool = False) -> GraphNode:
    """Convert a Paper ORM object to a GraphNode schema."""
    authors = []
    for a in (paper.authors or []):
        if isinstance(a, dict) and a.get("name"):
            authors.append(a["name"])
        elif isinstance(a, str):
            authors.append(a)

    return GraphNode(
        id=str(paper.id),
        title=paper.title,
        authors=authors,
        year=paper.year,
        citation_count=paper.citation_count,
        quality_score=paper.quality_score,
        in_library=in_library,
        doi=paper.doi,
        abstract=paper.abstract,
        is_seed=is_seed,
    )


@dataclass
class _RelationFetch:
    """Outcome of fetching one library paper's references and citations."""

    paper_id: UUID
    refs: list[dict] = field(default_factory=list)
    cites: list[dict] = field(default_factory=list)
    status: str = "ok"  # ok | failed | not_found | skipped
    cites_unavailable: bool = False  # cites: query rate-limited; refs still trusted


async def _job_is_cancelled(session_factory, job_id: UUID) -> bool:
    """Read the cancellation flag on a short-lived session."""
    async with session_factory() as db:
        return await task_service.should_abort(db, job_id)


async def build_citation_graph(
    project_id: UUID, job_id: UUID, session_factory
) -> None:
    """Background task: rebuild the project's citation graph from OpenAlex.

    Staged rebuild (issue GRAPH-DESTRUCTIVE-WIPE): every reference and citation is
    fetched first; only then, and only if the fetch was healthy, are the old edges
    deleted and the new ones inserted — inside ONE transaction. Health is judged on the
    whole build, not on whether `works` came back non-empty: if the OpenAlex circuit
    breaker tripped, the wall-clock budget was exceeded, or more than half of the
    attempted papers failed, the swap is skipped and the job is reported as `failed`
    (issue GRAPH-FAILURE-SWALLOWED) even when a handful of papers did succeed — one
    healthy paper must never be enough to overwrite a working graph with a partial one.
    A failed or empty run therefore leaves a previously working graph untouched.

    Papers are processed under a semaphore (issue GRAPH-SEQUENTIAL-NO-CONCURRENCY); the
    process-wide 8 rps OpenAlex limiter does the throttling, so this function never sleeps
    to pace requests. Per-paper HTTP failures are counted and reported instead of
    swallowed.
    """
    client = OpenAlexClient()
    breaker = CircuitBreaker("openalex")
    deadline = time.monotonic() + settings.graph_build_max_minutes * 60

    try:
        async with session_factory() as db:
            result = await db.execute(
                select(ProjectPaper)
                .where(ProjectPaper.project_id == project_id)
                .options(selectinload(ProjectPaper.paper))
            )
            project_papers = list(result.scalars().all())

        total = len(project_papers)
        logger.info(
            "graph.build_started project_id=%s job_id=%s papers=%s",
            project_id, job_id, total,
        )

        if not project_papers:
            async with session_factory() as db:
                await task_service.update_job_status(
                    db, job_id, JobStatus.completed,
                    progress=1.0,
                    progress_message="No papers to build graph from",
                    result={
                        "papers_processed": 0,
                        "papers_failed": 0,
                        "papers_skipped": 0,
                        "papers_not_found": 0,
                        "citations_unavailable": 0,
                        "total_papers": 0,
                        "edges_created": 0,
                        "nodes": 0,
                        "degradation": breaker.status(),
                    },
                )
            return

        if filter_queries_blocked("cites") or filter_queries_blocked("batch"):
            # Known quota block: a rebuild could only produce a degraded result —
            # refs-only under a cites: block, or a slow ~100-singles-per-paper
            # hydration under a batch block. If a graph already exists, bail BEFORE
            # burning the fetch budget; the outcome would be "kept existing edges"
            # anyway.
            async with session_factory() as db:
                existing = await db.execute(
                    select(CitationEdge.id)
                    .where(CitationEdge.project_id == project_id)
                    .limit(1)
                )
                if existing.first() is not None:
                    logger.warning(
                        "graph.build_skipped_quota_block job_id=%s papers=%s",
                        job_id, total,
                    )
                    await task_service.update_job_status(
                        db, job_id, JobStatus.failed,
                        progress=1.0,
                        error=(
                            "OpenAlex citation data is rate-limited right now — "
                            "your existing graph was kept untouched. Rebuild after "
                            "the quota resets for a full refresh."
                        ),
                        result={
                            "papers_processed": 0,
                            "papers_failed": 0,
                            "papers_skipped": total,
                            "papers_not_found": 0,
                            "citations_unavailable": 0,
                            "total_papers": total,
                            "edges_created": 0,
                            "nodes": 0,
                            "kept_existing_edges": True,
                            "degradation": breaker.status(),
                        },
                    )
                    return

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.0,
                progress_message=f"Building citation graph for {total} papers...",
            )

        semaphore = asyncio.Semaphore(settings.graph_build_concurrency)
        counter_lock = asyncio.Lock()
        completed = 0

        async def _process(project_paper) -> _RelationFetch:
            nonlocal completed
            paper = project_paper.paper
            outcome = _RelationFetch(paper_id=paper.id)
            work_id = resolve_work_id(paper.external_id, paper.doi)

            async with semaphore:
                if work_id is None:
                    outcome.status = "skipped"
                elif breaker.is_open or time.monotonic() > deadline:
                    outcome.status = "skipped"
                elif await _job_is_cancelled(session_factory, job_id):
                    outcome.status = "skipped"
                else:
                    try:
                        fetched = await fetch_relations(
                            client,
                            work_id,
                            max_refs=settings.graph_max_refs_per_paper,
                            max_cites=settings.graph_max_cites_per_paper,
                        )
                    except Exception:
                        logger.warning(
                            "graph.fetch_failed job_id=%s paper_id=%s work_id=%s",
                            job_id, paper.id, work_id, exc_info=True,
                        )
                        breaker.record_failure()
                        outcome.status = "failed"
                    else:
                        breaker.record_success()
                        if fetched is None:
                            outcome.status = "not_found"
                        else:
                            refs, cites = fetched
                            outcome.refs = refs
                            if cites is None:
                                outcome.cites_unavailable = True
                            else:
                                outcome.cites = cites

            async with counter_lock:
                completed += 1
                done = completed

            async with session_factory() as db:
                await task_service.update_job_status(
                    db, job_id, JobStatus.running,
                    progress=done / total,
                    progress_message=f"Fetched {done}/{total} papers",
                )
            return outcome

        fetches = await asyncio.gather(*(_process(pp) for pp in project_papers))

        papers_processed = sum(1 for f in fetches if f.status == "ok")
        papers_failed = sum(1 for f in fetches if f.status == "failed")
        papers_skipped = sum(1 for f in fetches if f.status == "skipped")
        papers_not_found = sum(1 for f in fetches if f.status == "not_found")
        citations_unavailable = sum(
            1 for f in fetches if f.status == "ok" and f.cites_unavailable
        )
        attempted = papers_processed + papers_failed + papers_not_found

        if await _job_is_cancelled(session_factory, job_id):
            logger.info("graph.build_aborted job_id=%s reason=cancelled", job_id)
            return

        # Health check (issues GRAPH-FAILURE-SWALLOWED / GRAPH-DESTRUCTIVE-WIPE): a build
        # that tripped the circuit breaker, ran past its wall-clock budget, or failed more
        # than half of its attempted fetches is not trustworthy enough to report as a plain
        # success or to overwrite the existing graph — even if a few papers did come back
        # with data. The swap below is gated on this, not on `works` being non-empty.
        deadline_exceeded = time.monotonic() > deadline
        failed_majority = attempted > 0 and papers_failed * 2 > attempted
        degraded = breaker.is_open or deadline_exceeded or failed_majority

        # Flatten every fetched work, remembering which slice belongs to which paper.
        works: list[dict] = []
        ref_spans: list[tuple[UUID, int, int]] = []
        cite_spans: list[tuple[UUID, int, int]] = []
        for fetch in fetches:
            start = len(works)
            works.extend(fetch.refs)
            ref_spans.append((fetch.paper_id, start, len(works)))
            start = len(works)
            works.extend(fetch.cites)
            cite_spans.append((fetch.paper_id, start, len(works)))

        stats = {
            "papers_processed": papers_processed,
            "papers_failed": papers_failed,
            "papers_skipped": papers_skipped,
            "papers_not_found": papers_not_found,
            "citations_unavailable": citations_unavailable,
            "total_papers": total,
            "edges_created": 0,
            "nodes": 0,
            "degradation": breaker.status(),
        }

        if degraded:
            # Never wipe on an unhealthy fetch, whether or not some work came back.
            reasons = []
            if failed_majority:
                reasons.append(
                    f"OpenAlex returned errors for {papers_failed} of "
                    f"{attempted} papers"
                )
            if breaker.is_open and not failed_majority:
                reasons.append(
                    "the OpenAlex circuit breaker opened after repeated failures"
                )
            if deadline_exceeded:
                reasons.append("the build exceeded its time budget")
            logger.warning(
                "graph.build_degraded job_id=%s failed=%s attempted=%s "
                "breaker_open=%s deadline_exceeded=%s",
                job_id, papers_failed, attempted, breaker.is_open, deadline_exceeded,
            )
            async with session_factory() as db:
                await task_service.update_job_status(
                    db, job_id, JobStatus.failed,
                    progress=1.0,
                    progress_message="Citation source unavailable",
                    result=stats,
                    error=(
                        "; ".join(reasons)
                        + "; no citation links could be trusted. Existing edges "
                        "were kept."
                    ),
                )
            return

        if not works:
            logger.info(
                "graph.build_completed job_id=%s edges=0 processed=%s "
                "citations_unavailable=%s",
                job_id, papers_processed, citations_unavailable,
            )
            empty_message = (
                "No links found — citing-paper data was rate-limited; "
                "rebuild after the quota resets"
                if citations_unavailable
                else "Done: no citation links found"
            )
            async with session_factory() as db:
                await task_service.update_job_status(
                    db, job_id, JobStatus.completed,
                    progress=1.0,
                    progress_message=empty_message,
                    result=stats,
                )
            return

        if citations_unavailable:
            # A refs-only build is strictly poorer than a refs+cites graph. Never
            # overwrite an existing (presumably fuller) edge set with it; only a
            # first build may proceed, because refs-only beats nothing.
            async with session_factory() as db:
                existing = await db.execute(
                    select(CitationEdge.id)
                    .where(CitationEdge.project_id == project_id)
                    .limit(1)
                )
                has_existing_edges = existing.first() is not None
            if has_existing_edges:
                logger.warning(
                    "graph.build_kept_existing job_id=%s citations_unavailable=%s",
                    job_id, citations_unavailable,
                )
                stats["kept_existing_edges"] = True
                async with session_factory() as db:
                    await task_service.update_job_status(
                        db, job_id, JobStatus.failed,
                        progress=1.0,
                        error=(
                            "OpenAlex citing-paper data is rate-limited right now — "
                            "your existing graph was kept untouched. Rebuild after "
                            "the quota resets for a full refresh."
                        ),
                        result=stats,
                    )
                return

        # Single transaction: resolve papers, delete the old edges, insert the new set.
        async with session_factory() as db:
            rows = await paper_service.get_or_create_papers_bulk(
                db, [to_paper_data(work) for work in works]
            )
            row_ids = [row.id for row in rows]

            pairs: set[tuple[UUID, UUID]] = set()
            for paper_id, start, end in ref_spans:
                for other_id in row_ids[start:end]:
                    if other_id != paper_id:
                        pairs.add((paper_id, other_id))
            for paper_id, start, end in cite_spans:
                for other_id in row_ids[start:end]:
                    if other_id != paper_id:
                        pairs.add((other_id, paper_id))

            await db.execute(
                delete(CitationEdge).where(CitationEdge.project_id == project_id)
            )
            for citing_id, cited_id in pairs:
                db.add(
                    CitationEdge(
                        project_id=project_id,
                        citing_paper_id=citing_id,
                        cited_paper_id=cited_id,
                    )
                )
            await db.commit()

        stats["edges_created"] = len(pairs)
        stats["nodes"] = len({node for pair in pairs for node in pair})

        logger.info(
            "graph.build_completed job_id=%s edges=%s processed=%s failed=%s",
            job_id, stats["edges_created"], papers_processed, papers_failed,
        )
        done_message = (
            f"Done: {stats['edges_created']} citation links from "
            f"{papers_processed}/{total} papers"
        )
        if citations_unavailable:
            done_message += (
                f" (citing-paper data was rate-limited for {citations_unavailable} "
                f"papers — built from reference links; rebuild later for citing links)"
            )
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.completed,
                progress=1.0,
                progress_message=done_message,
                result=stats,
            )

    except Exception as e:
        logger.exception("Graph building failed for project %s", project_id)
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.failed, error=str(e)
            )


def _compute_clusters(
    edges: list[tuple[str, str]],
) -> dict[str, int]:
    """Run Louvain community detection on the citation edge list.

    Returns a dict mapping node_id -> cluster_id (0-indexed).
    Returns empty dict if no edges.
    """
    if not edges:
        return {}
    G = nx.Graph()
    G.add_edges_from(edges)
    communities = nx.community.louvain_communities(G, seed=42)
    cluster_map: dict[str, int] = {}
    for cluster_id, community in enumerate(communities):
        for node_id in community:
            cluster_map[node_id] = cluster_id
    return cluster_map


def _compute_doi_scores(
    node_ids: list[str],
    adjacency: dict[str, list[str]],
    library_ids: set[str],
    citation_counts: dict[str, int | None],
    focus_id: str | None = None,
) -> dict[str, float]:
    """Degree-of-Interest scoring (Furnas 1986, adapted)."""
    scores: dict[str, float] = {}
    lib_dist = _bfs_multi(node_ids, adjacency, library_ids)
    focus_dist = _bfs_multi(node_ids, adjacency, {focus_id}) if focus_id else {}

    for nid in node_ids:
        intrinsic = 0.0
        if nid in library_ids:
            intrinsic += 3.0
        cc = citation_counts.get(nid) or 0
        intrinsic += math.log2(cc + 1) * 0.3

        penalty = 0.0
        d_lib = lib_dist.get(nid, 5)
        penalty += d_lib * 0.5

        bonus = 0.0
        if focus_id and focus_id != nid:
            d_focus = focus_dist.get(nid, 5)
            bonus += max(0, 3.0 - d_focus * 0.5)

        scores[nid] = intrinsic - penalty + bonus
    return scores


def _bfs_multi(
    node_ids: list[str],
    adjacency: dict[str, list[str]],
    sources: set[str],
) -> dict[str, int]:
    """BFS from multiple source nodes, return min distance to each node."""
    dist: dict[str, int] = {}
    queue: deque[str] = deque()
    for s in sources:
        if s in adjacency or s in set(node_ids):
            dist[s] = 0
            queue.append(s)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor not in dist:
                dist[neighbor] = dist[current] + 1
                queue.append(neighbor)
    return dist


async def _load_last_build(
    db: AsyncSession, project_id: UUID
) -> GraphBuildSummary | None:
    """Summarise the most recent graph_building job, or None if never built."""
    result = await db.execute(
        select(AnalysisJob)
        .where(
            and_(
                AnalysisJob.project_id == project_id,
                AnalysisJob.job_type == JobType.graph_building,
            )
        )
        .order_by(desc(AnalysisJob.created_at))
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None

    stats = job.result or {}
    status = job.status.value if hasattr(job.status, "value") else str(job.status)
    return GraphBuildSummary(
        status=status,
        edges_created=int(stats.get("edges_created") or 0),
        papers_processed=int(stats.get("papers_processed") or 0),
        papers_failed=int(stats.get("papers_failed") or 0),
        citations_unavailable=int(stats.get("citations_unavailable") or 0),
        error=job.error,
        completed_at=job.updated_at,
    )


async def get_graph_data(
    db: AsyncSession, project_id: UUID, focus_id: UUID | None = None
) -> GraphData:
    """Return graph nodes and edges for visualization."""
    # Load all edges
    edge_result = await db.execute(
        select(CitationEdge).where(
            CitationEdge.project_id == project_id
        )
    )
    edges = edge_result.scalars().all()

    last_build = await _load_last_build(db, project_id)

    if not edges:
        return GraphData(nodes=[], edges=[], last_build=last_build)

    # Collect all paper IDs
    paper_ids: set[UUID] = set()
    for e in edges:
        paper_ids.add(e.citing_paper_id)
        paper_ids.add(e.cited_paper_id)

    # Load papers
    paper_result = await db.execute(
        select(Paper).where(Paper.id.in_(paper_ids))
    )
    papers = {p.id: p for p in paper_result.scalars().all()}

    # Check which are in library
    lib_result = await db.execute(
        select(ProjectPaper.paper_id).where(
            and_(
                ProjectPaper.project_id == project_id,
                ProjectPaper.paper_id.in_(paper_ids),
            )
        )
    )
    library_ids = {row[0] for row in lib_result.all()}

    # Build adjacency for DOI scoring
    adjacency: dict[str, list[str]] = {}
    for e in edges:
        src = str(e.citing_paper_id)
        tgt = str(e.cited_paper_id)
        adjacency.setdefault(src, []).append(tgt)
        adjacency.setdefault(tgt, []).append(src)

    node_ids_list = [str(pid) for pid in papers]
    citation_counts_map = {str(pid): p.citation_count for pid, p in papers.items()}
    lib_id_strs = {str(lid) for lid in library_ids}

    doi_scores = _compute_doi_scores(
        node_ids=node_ids_list,
        adjacency=adjacency,
        library_ids=lib_id_strs,
        citation_counts=citation_counts_map,
        focus_id=str(focus_id) if focus_id else None,
    )

    # Compute Louvain clusters
    edge_tuples = [
        (str(e.citing_paper_id), str(e.cited_paper_id)) for e in edges
    ]
    cluster_map = _compute_clusters(edge_tuples)

    # Build nodes
    nodes = []
    for pid, paper in papers.items():
        node = _paper_to_graph_node(paper, in_library=pid in library_ids, is_seed=pid in library_ids)
        node.doi_score = doi_scores.get(str(pid))
        node.cluster_id = cluster_map.get(str(pid))
        nodes.append(node)

    # Sort by DOI score desc and cap, always keeping library papers
    nodes.sort(key=lambda n: n.doi_score or 0, reverse=True)
    library_nodes = [n for n in nodes if n.is_seed]
    non_library_nodes = [n for n in nodes if not n.is_seed]
    max_external = settings.graph_max_nodes - len(library_nodes)
    nodes = library_nodes + non_library_nodes[:max(0, max_external)]
    visible_ids = {n.id for n in nodes}

    # Build edges (only between visible nodes)
    graph_edges = []
    for e in edges:
        src = str(e.citing_paper_id)
        tgt = str(e.cited_paper_id)
        if src in visible_ids and tgt in visible_ids:
            graph_edges.append(GraphEdge(source=src, target=tgt))

    return GraphData(nodes=nodes, edges=graph_edges, last_build=last_build)


async def expand_node(
    paper_id: UUID, project_id: UUID, session_factory
) -> GraphExpansion:
    """Fetch refs/citations for one paper from OpenAlex, return new nodes+edges.

    References and citations are fetched concurrently and the whole write happens in one
    transaction, so the call is two OpenAlex round trips deep instead of two sequential
    retry chains that could never fit the API's 15s ceiling (issue EXPAND-NODE-504).

    Raises:
        ``httpx.HTTPError`` when OpenAlex fails; ``app/api/graph.py`` turns that into a
        502 so the UI can show a real message instead of a silent spinner.
    """
    client = OpenAlexClient()

    async with session_factory() as db:
        paper = await db.get(Paper, paper_id)
        if not paper:
            return GraphExpansion(new_nodes=[], new_edges=[])
        work_id = resolve_work_id(paper.external_id, paper.doi)

    if not work_id:
        logger.info("graph.expand_skipped paper_id=%s reason=no_openalex_id", paper_id)
        return GraphExpansion(new_nodes=[], new_edges=[])

    fetched = await fetch_relations(
        client,
        work_id,
        # Healthy path: hydrate the full referenced_works list (2 batch requests) so
        # cap_works below surfaces the MOST-CITED 15, not the first-listed 15. Under
        # the singles fallback that same list would cost ~100 sequential requests and
        # blow the route's 15s budget, so degrade to the display cap instead.
        max_refs=(
            settings.graph_expand_max_refs
            if filter_queries_blocked("batch")
            else settings.graph_max_refs_per_paper
        ),
        # get_citations returns cited_by_count-sorted results, so per_page at the
        # display cap already yields the top-N — no quality loss from the small ask.
        max_cites=settings.graph_expand_max_cites,
    )
    if fetched is None:
        logger.info("graph.expand_not_found paper_id=%s work_id=%s", paper_id, work_id)
        return GraphExpansion(new_nodes=[], new_edges=[])

    refs = cap_works(fetched[0], settings.graph_expand_max_refs)
    cites = cap_works(fetched[1] or [], settings.graph_expand_max_cites)

    new_nodes: list[GraphNode] = []
    new_edges: list[GraphEdge] = []

    async with session_factory() as db:
        rows = await paper_service.get_or_create_papers_bulk(
            db, [to_paper_data(work) for work in refs + cites]
        )
        ref_rows = rows[: len(refs)]
        cite_rows = rows[len(refs):]

        lib_result = await db.execute(
            select(ProjectPaper.paper_id).where(
                ProjectPaper.project_id == project_id
            )
        )
        library_ids = {row[0] for row in lib_result.all()}

        existing_result = await db.execute(
            select(CitationEdge.citing_paper_id, CitationEdge.cited_paper_id)
            .where(CitationEdge.project_id == project_id)
        )
        existing_pairs = {(row[0], row[1]) for row in existing_result.all()}
        seen_node_ids: set[str] = set()

        def _add_node(row) -> None:
            node = _paper_to_graph_node(
                row,
                in_library=row.id in library_ids,
                is_seed=row.id in library_ids,
            )
            if node.id not in seen_node_ids:
                seen_node_ids.add(node.id)
                new_nodes.append(node)

        for ref_row in ref_rows:
            pair = (paper_id, ref_row.id)
            if pair[0] != pair[1] and pair not in existing_pairs:
                db.add(CitationEdge(
                    project_id=project_id,
                    citing_paper_id=paper_id,
                    cited_paper_id=ref_row.id,
                ))
                new_edges.append(GraphEdge(
                    source=str(paper_id), target=str(ref_row.id)
                ))
                existing_pairs.add(pair)
            _add_node(ref_row)

        for cite_row in cite_rows:
            pair = (cite_row.id, paper_id)
            if pair[0] != pair[1] and pair not in existing_pairs:
                db.add(CitationEdge(
                    project_id=project_id,
                    citing_paper_id=cite_row.id,
                    cited_paper_id=paper_id,
                ))
                new_edges.append(GraphEdge(
                    source=str(cite_row.id), target=str(paper_id)
                ))
                existing_pairs.add(pair)
            _add_node(cite_row)

        await db.commit()

    logger.info(
        "graph.expand_completed paper_id=%s nodes=%s edges=%s",
        paper_id, len(new_nodes), len(new_edges),
    )
    return GraphExpansion(new_nodes=new_nodes, new_edges=new_edges)

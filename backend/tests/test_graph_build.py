"""build_citation_graph: staged rebuild, concurrency, failure accounting, cancellation.

Every OpenAlex call is mocked at the client boundary — no network access. The service
opens its own sessions through session_factory, so these tests build a second engine
against the same test database (the same pattern as tests/test_seed_papers_api.py).
"""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.models.citation_edge import CitationEdge
from app.models.paper import Paper, SourceApi
from app.models.project import Project
from app.models.project_paper import ProjectPaper
from app.models.user import User
from app.services.graph import build_citation_graph
from tests.conftest import TEST_DATABASE_URL


@pytest.fixture
async def session_factory():
    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _work(w_id: str, title: str, *, refs: list[str] | None = None, citations: int = 0) -> dict:
    """A parse_work_dict-shaped payload (PaperData keys + openalex_id + referenced_works)."""
    return {
        "doi": None,
        "title": title,
        "authors": [{"name": "Author One"}],
        "year": 2020,
        "journal_name": "Journal",
        "journal_issn": None,
        "citation_count": citations,
        "abstract": None,
        "source_api": "openalex",
        "external_id": w_id,
        "full_text_url": None,
        "wos_collection": None,
        "wos_categories": None,
        "openalex_id": w_id,
        "referenced_works": list(refs or []),
    }


def _fake_client(*, work=None, batch=None, cites=None, error=None) -> MagicMock:
    client = MagicMock()
    if error is not None:
        client.get_work = AsyncMock(side_effect=error)
    else:
        client.get_work = AsyncMock(return_value=work)
    client.get_works_batch = AsyncMock(return_value=list(batch or []))
    client.get_citations = AsyncMock(return_value=list(cites or []))
    return client


async def _seed(db, titles_and_ids: list[tuple[str, str]]):
    """Create a user, a project, its library papers and a pending graph_building job."""
    user = User(
        email=f"graph-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="not-a-real-hash",
        name="Graph Tester",
    )
    db.add(user)
    await db.flush()

    project = Project(user_id=user.id, title="Graph Project")
    db.add(project)
    await db.flush()

    papers = []
    for title, external_id in titles_and_ids:
        paper = Paper(
            title=title,
            authors=[],
            external_id=external_id,
            source_api=SourceApi.openalex,
        )
        db.add(paper)
        await db.flush()
        db.add(ProjectPaper(project_id=project.id, paper_id=paper.id))
        papers.append(paper)

    job = AnalysisJob(
        project_id=project.id,
        user_id=user.id,
        job_type=JobType.graph_building,
        status=JobStatus.pending,
        progress=0.0,
    )
    db.add(job)
    await db.commit()
    return project.id, job.id, papers


async def _edges(session_factory, project_id):
    async with session_factory() as db:
        result = await db.execute(
            select(CitationEdge).where(CitationEdge.project_id == project_id)
        )
        return list(result.scalars().all())


async def _job(session_factory, job_id):
    async with session_factory() as db:
        return await db.get(AnalysisJob, job_id)


async def test_build_creates_edges_and_reports_stats(db_session, session_factory):
    project_id, job_id, _ = await _seed(db_session, [("Seed", "W1")])
    fake = _fake_client(
        work=_work("W1", "Seed", refs=["W2"]),
        batch=[_work("W2", "Reference", citations=5)],
        cites=[_work("W3", "Citing work", citations=2)],
    )

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        await build_citation_graph(
            project_id=project_id, job_id=job_id, session_factory=session_factory
        )

    edges = await _edges(session_factory, project_id)
    job = await _job(session_factory, job_id)

    assert len(edges) == 2
    assert job.status is JobStatus.completed
    assert job.result["edges_created"] == 2
    assert job.result["papers_processed"] == 1
    assert job.result["papers_failed"] == 0
    assert job.result["total_papers"] == 1
    assert job.result["degradation"]["source"] == "openalex"
    assert fake.get_work.await_count == 1


async def test_build_replaces_old_edges_in_one_transaction(db_session, session_factory):
    project_id, job_id, papers = await _seed(db_session, [("A", "W1"), ("B", "W2")])
    async with session_factory() as db:
        db.add(
            CitationEdge(
                project_id=project_id,
                citing_paper_id=papers[0].id,
                cited_paper_id=papers[1].id,
            )
        )
        await db.commit()

    fake = _fake_client(
        work=_work("W1", "A", refs=["W9"]), batch=[_work("W9", "Fresh reference")], cites=[]
    )

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        await build_citation_graph(
            project_id=project_id, job_id=job_id, session_factory=session_factory
        )

    edges = await _edges(session_factory, project_id)
    job = await _job(session_factory, job_id)

    # The stale A -> B edge is gone; only the freshly fetched edges remain.
    assert (papers[0].id, papers[1].id) not in {
        (e.citing_paper_id, e.cited_paper_id) for e in edges
    }
    assert len(edges) == job.result["edges_created"] > 0
    assert job.status is JobStatus.completed


async def test_build_keeps_existing_edges_and_fails_when_every_paper_errors(
    db_session, session_factory
):
    project_id, job_id, papers = await _seed(db_session, [("A", "W1"), ("B", "W2")])
    async with session_factory() as db:
        db.add(
            CitationEdge(
                project_id=project_id,
                citing_paper_id=papers[0].id,
                cited_paper_id=papers[1].id,
            )
        )
        await db.commit()

    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("500", request=MagicMock(), response=response)
    fake = _fake_client(error=error)

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        await build_citation_graph(
            project_id=project_id, job_id=job_id, session_factory=session_factory
        )

    edges = await _edges(session_factory, project_id)
    job = await _job(session_factory, job_id)

    assert len(edges) == 1, "a failed rebuild must never destroy a working graph"
    assert job.status is JobStatus.failed
    assert "OpenAlex" in (job.error or "")
    assert job.result["papers_failed"] == 2
    assert job.result["edges_created"] == 0


async def test_build_completes_when_openalex_has_no_record(db_session, session_factory):
    """A 404 is honest emptiness, not an error — the job completes with 0 edges."""
    project_id, job_id, _ = await _seed(db_session, [("Obscure CNKI paper", "W1")])
    fake = _fake_client(work=None)

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        await build_citation_graph(
            project_id=project_id, job_id=job_id, session_factory=session_factory
        )

    job = await _job(session_factory, job_id)
    assert job.status is JobStatus.completed
    assert job.result["edges_created"] == 0
    assert job.result["papers_not_found"] == 1
    assert job.result["papers_failed"] == 0


async def test_build_skips_papers_without_an_openalex_identifier(db_session, session_factory):
    project_id, job_id, _ = await _seed(db_session, [("Manual entry", None)])
    fake = _fake_client(work=_work("W1", "unused"))

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        await build_citation_graph(
            project_id=project_id, job_id=job_id, session_factory=session_factory
        )

    job = await _job(session_factory, job_id)
    assert fake.get_work.await_count == 0
    assert job.status is JobStatus.completed
    assert job.result["papers_skipped"] == 1


async def test_build_with_no_library_papers_completes_immediately(db_session, session_factory):
    project_id, job_id, _ = await _seed(db_session, [])
    fake = _fake_client(work=_work("W1", "unused"))

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        await build_citation_graph(
            project_id=project_id, job_id=job_id, session_factory=session_factory
        )

    job = await _job(session_factory, job_id)
    assert job.status is JobStatus.completed
    assert job.result["total_papers"] == 0
    assert job.result["edges_created"] == 0
    assert fake.get_work.await_count == 0


async def test_build_aborts_without_touching_edges_when_cancelled(db_session, session_factory):
    """A cancelled job exits cleanly and must not run the destructive swap."""
    project_id, job_id, papers = await _seed(db_session, [("A", "W1"), ("B", "W2")])
    async with session_factory() as db:
        db.add(
            CitationEdge(
                project_id=project_id,
                citing_paper_id=papers[0].id,
                cited_paper_id=papers[1].id,
            )
        )
        await db.commit()

    fake = _fake_client(work=_work("W1", "A", refs=["W9"]), batch=[_work("W9", "Ref")])

    with patch("app.services.graph.OpenAlexClient", return_value=fake), patch(
        "app.services.task.should_abort", new_callable=AsyncMock, return_value=True
    ):
        await build_citation_graph(
            project_id=project_id, job_id=job_id, session_factory=session_factory
        )

    edges = await _edges(session_factory, project_id)
    assert fake.get_work.await_count == 0
    assert len(edges) == 1


async def test_build_skips_swap_when_majority_fails_despite_one_healthy_paper(
    db_session, session_factory
):
    """GRAPH-DESTRUCTIVE-WIPE(#16): the swap must be gated on build health, not on
    whether any work at all came back. One successful paper must not be enough to
    overwrite a working graph when most of the fetch failed.
    """
    titles = [("Healthy", "W1")] + [(f"Broken {i}", f"W{i + 2}") for i in range(4)]
    project_id, job_id, papers = await _seed(db_session, titles)
    async with session_factory() as db:
        db.add(
            CitationEdge(
                project_id=project_id,
                citing_paper_id=papers[0].id,
                cited_paper_id=papers[1].id,
            )
        )
        await db.commit()

    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("500", request=MagicMock(), response=response)

    async def _get_work(work_id):
        if work_id == "W1":
            return _work("W1", "Healthy", refs=["W99"])
        raise error

    fake = MagicMock()
    fake.get_work = AsyncMock(side_effect=_get_work)
    fake.get_works_batch = AsyncMock(return_value=[_work("W99", "Fresh reference")])
    fake.get_citations = AsyncMock(return_value=[])

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        await build_citation_graph(
            project_id=project_id, job_id=job_id, session_factory=session_factory
        )

    edges = await _edges(session_factory, project_id)
    job = await _job(session_factory, job_id)

    assert (papers[0].id, papers[1].id) in {
        (e.citing_paper_id, e.cited_paper_id) for e in edges
    }, "the pre-existing edge must survive a majority-failed build"
    assert len(edges) == 1
    assert job.status is JobStatus.failed
    assert job.result["papers_processed"] == 1
    assert job.result["papers_failed"] == 4
    assert job.result["edges_created"] == 0


async def test_build_completes_refs_only_when_cites_query_is_rate_limited(
    db_session, session_factory
):
    """A quota-length 429 on cites: degrades to a refs-only build, not a failure."""
    project_id, job_id, _ = await _seed(db_session, [("Paper A", "W100")])

    request = httpx.Request("GET", "https://api.openalex.org/works")
    response = httpx.Response(429, request=request, headers={"Retry-After": "33926"})
    rate_limited = httpx.HTTPStatusError("HTTP 429", request=request, response=response)

    fake = _fake_client(
        work=_work("W100", "Paper A", refs=["W200"]),
        batch=[_work("W200", "Reference")],
    )
    fake.get_citations = AsyncMock(side_effect=rate_limited)

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        await build_citation_graph(
            project_id=project_id, job_id=job_id, session_factory=session_factory
        )

    job = await _job(session_factory, job_id)
    assert job.status is JobStatus.completed, job.error
    assert job.result["citations_unavailable"] == 1
    assert job.result["papers_failed"] == 0
    assert job.result["edges_created"] == 1  # the refs-only edge still lands
    assert "rate-limited" in (job.progress_message or "")
    edges = await _edges(session_factory, project_id)
    assert len(edges) == 1


async def test_build_keeps_existing_edges_when_cites_throttled(db_session, session_factory):
    """A refs-only rebuild must never overwrite a fuller existing graph (review #2)."""
    project_id, job_id, papers = await _seed(db_session, [("A", "W1"), ("B", "W2")])
    async with session_factory() as db:
        db.add(
            CitationEdge(
                project_id=project_id,
                citing_paper_id=papers[0].id,
                cited_paper_id=papers[1].id,
            )
        )
        await db.commit()

    request = httpx.Request("GET", "https://api.openalex.org/works")
    response = httpx.Response(429, request=request, headers={"Retry-After": "33926"})
    rate_limited = httpx.HTTPStatusError("HTTP 429", request=request, response=response)

    fake = _fake_client(
        work=_work("W1", "A", refs=["W3"]),
        batch=[_work("W3", "Reference")],
    )
    fake.get_citations = AsyncMock(side_effect=rate_limited)

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        await build_citation_graph(
            project_id=project_id, job_id=job_id, session_factory=session_factory
        )

    edges = await _edges(session_factory, project_id)
    job = await _job(session_factory, job_id)

    assert len(edges) == 1, "refs-only rebuild must not replace the existing graph"
    assert (edges[0].citing_paper_id, edges[0].cited_paper_id) == (
        papers[0].id,
        papers[1].id,
    ), "the surviving edge must be the ORIGINAL one, not a refs-only replacement"
    assert job.status is JobStatus.failed
    assert "rate-limited" in (job.error or "")
    assert job.result["kept_existing_edges"] is True


async def test_build_bails_early_under_known_quota_block(db_session, session_factory):
    """With a memoized quota block and existing edges, no fetch budget is burned."""
    from app.clients.openalex import note_filter_block

    project_id, job_id, papers = await _seed(db_session, [("A", "W1"), ("B", "W2")])
    async with session_factory() as db:
        db.add(
            CitationEdge(
                project_id=project_id,
                citing_paper_id=papers[0].id,
                cited_paper_id=papers[1].id,
            )
        )
        await db.commit()

    note_filter_block("cites", 99999.0)  # quota-length block (> cap)
    fake = _fake_client(work=_work("W1", "A", refs=["W3"]))

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        await build_citation_graph(
            project_id=project_id, job_id=job_id, session_factory=session_factory
        )

    edges = await _edges(session_factory, project_id)
    job = await _job(session_factory, job_id)

    assert fake.get_work.await_count == 0, "must not spend upstream requests"
    assert len(edges) == 1
    assert job.status is JobStatus.failed
    assert job.result["kept_existing_edges"] is True
    assert "rate-limited" in (job.error or "")

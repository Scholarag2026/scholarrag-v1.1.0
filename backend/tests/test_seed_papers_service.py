"""Tests for seed papers service -- resolve_seed, _dedup_papers, normalize_title."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.paper import PaperData
from tests.conftest import TEST_DATABASE_URL


def _make_paper(
    doi: str | None,
    title: str,
    source: str = "openalex",
    external_id: str | None = None,
) -> PaperData:
    return PaperData(
        doi=doi,
        title=title,
        authors=[{"name": "Test Author"}],
        year=2023,
        citation_count=10,
        source_api=source,
        external_id=external_id,
    )


# ---------- normalize_title tests ----------


def test_normalize_title_lowercases():
    from app.services.seed_papers import normalize_title

    assert normalize_title("Machine Learning In Education") == "machine learning in education"


def test_normalize_title_strips_punctuation():
    from app.services.seed_papers import normalize_title

    assert normalize_title("A Study: Results & Analysis!") == "a study results  analysis"


# ---------- _dedup_papers tests ----------


def test_dedup_papers_removes_doi_duplicates():
    from app.services.seed_papers import _dedup_papers

    papers = [
        _make_paper("10.1/abc", "Paper A"),
        _make_paper("10.1/ABC", "Paper A Different Title"),  # same DOI, different case
        _make_paper("10.1/def", "Paper B"),
    ]
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()

    unique = _dedup_papers(papers, seen_dois, seen_titles)
    assert len(unique) == 2
    assert "10.1/abc" in seen_dois
    assert "10.1/def" in seen_dois


def test_dedup_papers_removes_title_duplicates():
    from app.services.seed_papers import _dedup_papers

    papers = [
        _make_paper(None, "Machine Learning in Education"),
        _make_paper(None, "machine learning in education"),  # same normalized title
        _make_paper(None, "Deep Learning Methods"),
    ]
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()

    unique = _dedup_papers(papers, seen_dois, seen_titles)
    assert len(unique) == 2


def test_dedup_papers_respects_pre_existing_seen():
    from app.services.seed_papers import _dedup_papers

    papers = [
        _make_paper("10.1/abc", "Paper A"),
        _make_paper("10.1/def", "Paper B"),
    ]
    seen_dois: set[str] = {"10.1/abc"}  # already seen
    seen_titles: set[str] = set()

    unique = _dedup_papers(papers, seen_dois, seen_titles)
    assert len(unique) == 1
    assert unique[0].title == "Paper B"


# ---------- resolve_seed tests ----------


def _work(w_id: str, title: str, *, doi: str | None = None) -> dict:
    return {
        "doi": doi,
        "title": title,
        "authors": [{"name": "Author One"}],
        "year": 2022,
        "journal_name": None,
        "journal_issn": None,
        "citation_count": 7,
        "abstract": None,
        "source_api": "openalex",
        "external_id": w_id,
        "full_text_url": None,
        "wos_collection": None,
        "wos_categories": None,
        "openalex_id": w_id,
        "referenced_works": [],
    }


@pytest.mark.asyncio
async def test_resolve_seed_doi_uses_openalex_get_work():
    """A DOI seed is an exact record lookup, not a keyword search."""
    from app.services.seed_papers import resolve_seed

    with patch(
        "app.services.seed_papers.OpenAlexClient.get_work",
        new_callable=AsyncMock,
        return_value=_work("W123", "Test Paper", doi="10.1000/test"),
    ) as get_work, patch(
        "app.services.seed_papers.OpenAlexClient.search",
        new_callable=AsyncMock,
        return_value=[],
    ) as search:
        result = await resolve_seed("10.1000/test", is_doi=True)

    assert result is not None
    assert result.doi == "10.1000/test"
    assert result.external_id == "W123"
    get_work.assert_awaited_once_with("10.1000/test")
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_seed_doi_falls_back_to_search_on_404():
    from app.services.seed_papers import resolve_seed

    fallback = _make_paper("10.1000/oa", "OpenAlex Paper", source="openalex")

    with patch(
        "app.services.seed_papers.OpenAlexClient.get_work",
        new_callable=AsyncMock,
        return_value=None,
    ), patch(
        "app.services.seed_papers.OpenAlexClient.search",
        new_callable=AsyncMock,
        return_value=[fallback],
    ):
        result = await resolve_seed("10.1000/oa", is_doi=True)

    assert result is not None
    assert result.doi == "10.1000/oa"


@pytest.mark.asyncio
async def test_resolve_seed_title_uses_search():
    from app.services.seed_papers import resolve_seed

    match = _make_paper(None, "Deep Learning", source="openalex", external_id="W456")

    with patch(
        "app.services.seed_papers.OpenAlexClient.get_work",
        new_callable=AsyncMock,
    ) as get_work, patch(
        "app.services.seed_papers.OpenAlexClient.search",
        new_callable=AsyncMock,
        return_value=[match],
    ):
        result = await resolve_seed("Deep Learning", is_doi=False)

    assert result is not None
    assert result.title == "Deep Learning"
    get_work.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_seed_returns_none_when_openalex_fails():
    from app.services.seed_papers import resolve_seed

    with patch(
        "app.services.seed_papers.OpenAlexClient.get_work",
        new_callable=AsyncMock,
        side_effect=Exception("OA down"),
    ), patch(
        "app.services.seed_papers.OpenAlexClient.search",
        new_callable=AsyncMock,
        side_effect=Exception("OA down"),
    ):
        result = await resolve_seed("nonexistent-doi", is_doi=True)

    assert result is None


def test_seed_service_has_no_semantic_scholar_references():
    """User decision #1: OpenAlex is the only scholarly data source."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "app" / "services" / "seed_papers.py"
    ).read_text(encoding="utf-8")

    assert "SemanticScholarClient" not in source
    assert "get_paper_references" not in source


@pytest.mark.asyncio
async def test_seed_expansion_survives_throttled_cites(db_session):
    """fetched[1] is None under a cites: rate limit — the JOB must complete from refs.

    Regression guard for the crash ``fetched[0] + fetched[1]`` (TypeError on None),
    which failed the entire seed-expansion job. Drives run_seed_expansion end to end.
    """
    import uuid as _uuid

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.models.analysis_job import AnalysisJob, JobStatus, JobType
    from app.models.project import Project
    from app.models.user import User
    from app.services.seed_papers import run_seed_expansion

    user = User(
        email=f"seedthrottle-{_uuid.uuid4().hex[:8]}@example.com",
        password_hash="not-a-real-hash",
        name="Seed Throttle",
    )
    db_session.add(user)
    await db_session.flush()
    project = Project(user_id=user.id, title="Seed Throttle Project")
    db_session.add(project)
    await db_session.flush()
    job = AnalysisJob(
        project_id=project.id,
        user_id=user.id,
        job_type=JobType.seed_expand,
        status=JobStatus.pending,
        progress=0.0,
    )
    db_session.add(job)
    await db_session.commit()

    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        seed = _make_paper("10.1000/seed", "Seed Paper", external_id="W1")
        ref = _work("W77", "Reference Work", doi="10.1000/ref")
        with patch(
            "app.services.seed_papers.resolve_seed",
            new=AsyncMock(return_value=seed),
        ), patch(
            "app.services.seed_papers.fetch_relations",
            new=AsyncMock(return_value=([ref], None)),
        ):
            await run_seed_expansion(
                project_id=project.id,
                job_id=job.id,
                session_factory=factory,
                dois=["10.1000/seed"],
                titles=[],
            )

        async with factory() as check:
            fresh = await check.get(AnalysisJob, job.id)
            assert fresh.status is JobStatus.completed, fresh.error
            assert fresh.result["total_expanded"] >= 1
    finally:
        await engine.dispose()

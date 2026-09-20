"""Deep analysis must analyse full-text papers concurrently, bounded and cancellable."""

import asyncio
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from app.models.analysis_job import JobStatus, JobType  # noqa: E402
from tests.t5_fixtures import (  # noqa: E402
    make_session_factory,
    seed_job,
    seed_paper,
    seed_project,
    seed_user,
)

CHUNKS = [{"section": "results", "text": "Findings are strong."}]


class _DeepResult:
    def model_dump(self):
        return {"key_findings": "x", "themes": ["t"]}


@pytest.mark.asyncio
async def test_full_text_papers_are_analysed_concurrently(db_session):
    from app.services.deep_analysis import analyze_all_papers

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    for n in range(10):
        await seed_paper(
            db_session,
            project,
            title=f"FT Paper {n}",
            metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
        )
    job = await seed_job(db_session, project, JobType.deep_analysis)

    state = {"in_flight": 0, "max_in_flight": 0}

    async def _fake_analyze(*_args, **_kwargs):
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        await asyncio.sleep(0.05)
        state["in_flight"] -= 1
        return _DeepResult()

    factory, engine = make_session_factory()
    with patch("app.services.deep_analysis.analyze_paper_text", new=_fake_analyze):
        await analyze_all_papers(
            project_id=project.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    assert state["max_in_flight"] > 1, "deep analysis is still strictly sequential"
    assert state["max_in_flight"] <= 4, "must stay bounded by analysis_concurrency"

    await db_session.refresh(job)
    assert job.status == JobStatus.completed, job.error
    assert job.result == {"analyzed": 10, "total": 10}


@pytest.mark.asyncio
async def test_already_analysed_papers_are_skipped(db_session):
    from app.services.deep_analysis import analyze_all_papers

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(
        db_session,
        project,
        title="Done",
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": CHUNKS,
            "deep_analysis": {"key_findings": "already"},
        },
    )
    job = await seed_job(db_session, project, JobType.deep_analysis)

    factory, engine = make_session_factory()
    with patch(
        "app.services.deep_analysis.analyze_paper_text", new_callable=AsyncMock
    ) as mock_analyze:
        await analyze_all_papers(
            project_id=project.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    mock_analyze.assert_not_awaited()
    await db_session.refresh(job)
    assert job.result == {"analyzed": 1, "total": 1}


@pytest.mark.asyncio
async def test_deep_analysis_stops_when_cancelled(db_session):
    from app.services.deep_analysis import analyze_all_papers

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    for n in range(20):
        await seed_paper(
            db_session,
            project,
            title=f"C{n}",
            metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
        )
    job = await seed_job(db_session, project, JobType.deep_analysis)

    async def _fake_analyze(*_args, **_kwargs):
        await asyncio.sleep(0.01)
        return _DeepResult()

    factory, engine = make_session_factory()
    with patch(
        "app.services.deep_analysis.analyze_paper_text", new=_fake_analyze
    ), patch(
        "app.services.task.should_abort", new_callable=AsyncMock, return_value=True
    ):
        await analyze_all_papers(
            project_id=project.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status == JobStatus.cancelled


@pytest.mark.asyncio
async def test_many_already_analysed_papers_do_not_deadlock_the_pool(db_session):
    """Regression for the skip-branch deadlock: `_record_progress()` must never be
    awaited while a DB session is still checked out, or a library larger than the
    pool (size 5 + overflow 10 by default) hangs forever on a QueuePool timeout."""
    from app.services.deep_analysis import analyze_all_papers

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    for n in range(40):
        await seed_paper(
            db_session,
            project,
            title=f"Done {n}",
            metadata_={
                "fulltext_status": "acquired",
                "fulltext_chunks": CHUNKS,
                "deep_analysis": {"key_findings": "already"},
            },
        )
    job = await seed_job(db_session, project, JobType.deep_analysis)

    factory, engine = make_session_factory()
    with patch(
        "app.services.deep_analysis.analyze_paper_text", new_callable=AsyncMock
    ) as mock_analyze:
        await analyze_all_papers(
            project_id=project.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    mock_analyze.assert_not_awaited()
    await db_session.refresh(job)
    assert job.status == JobStatus.completed, job.error
    assert job.result == {"analyzed": 40, "total": 40}


@pytest.mark.asyncio
async def test_one_paper_commit_failure_does_not_sink_the_job(db_session):
    """A single paper whose commit raises (e.g. non-serialisable analysis payload)
    must be skipped, not abort the whole job or leave it stuck in `running`."""
    from app.services.deep_analysis import analyze_all_papers

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    for n in range(5):
        await seed_paper(
            db_session,
            project,
            title=f"Good {n}",
            metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
        )
    await seed_paper(
        db_session,
        project,
        title="Bad",
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
    )
    job = await seed_job(db_session, project, JobType.deep_analysis)

    class _BadResult:
        def model_dump(self):
            return {"key_findings": {1, 2, 3}}  # a set is not JSON-serialisable

    async def _fake_analyze(title, *_args, **_kwargs):
        if title == "Bad":
            return _BadResult()
        return _DeepResult()

    factory, engine = make_session_factory()
    with patch("app.services.deep_analysis.analyze_paper_text", new=_fake_analyze):
        await analyze_all_papers(
            project_id=project.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status == JobStatus.completed, job.error
    assert job.result == {"analyzed": 5, "total": 6}

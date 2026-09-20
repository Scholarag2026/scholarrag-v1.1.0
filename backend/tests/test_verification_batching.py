"""Reference verification: one WoS query for the batch, bounded CrossRef fan-out."""

import asyncio
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from app.models.analysis_job import JobStatus, JobType  # noqa: E402
from app.models.wos_journal import WosJournal  # noqa: E402
from tests.t5_fixtures import (  # noqa: E402
    make_session_factory,
    seed_job,
    seed_paper,
    seed_project,
    seed_user,
)


@pytest.mark.asyncio
async def test_wos_is_looked_up_once_for_the_whole_batch(db_session):
    from app.api.verification import _run_verification_background

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    db_session.add(
        WosJournal(
            journal_title="J",
            issn="1234-5678",
            collection="SSCI",
            categories="Education",
        )
    )
    await db_session.commit()
    for n in range(8):
        await seed_paper(
            db_session,
            project,
            title=f"Paper {n}",
            doi=f"10.1/p{n}",
            journal_issn="1234-5678",
        )
    job = await seed_job(db_session, project, JobType.qa)

    calls = {"bulk": 0, "single": 0}

    real_bulk = None
    from app.services import wos_import as wos_module

    real_bulk = wos_module.is_wos_indexed_bulk

    async def _counting_bulk(db, issns):
        calls["bulk"] += 1
        return await real_bulk(db, issns)

    async def _counting_single(*_args, **_kwargs):
        calls["single"] += 1
        return False, None, None

    factory, engine = make_session_factory()
    with patch(
        "app.api.verification.is_wos_indexed_bulk", new=_counting_bulk
    ), patch(
        "app.api.verification.CrossRefClient.verify_doi",
        new_callable=AsyncMock,
        return_value=None,
    ), patch(
        "app.services.wos_import.is_wos_indexed", new=_counting_single
    ):
        await _run_verification_background(
            job_id=job.id,
            project_id=project.id,
            paper_ids=[],
            session_factory=factory,
        )
    await engine.dispose()

    assert calls["bulk"] == 1
    assert calls["single"] == 0

    await db_session.refresh(job)
    assert job.status == JobStatus.completed, job.error
    assert job.result["total"] == 8
    # The WoS collection reaches the structured indicator details (frontend contract).
    wos = [i for i in job.result["results"][0]["indicators"] if i["check_type"] == "wos_indexed"]
    assert wos[0]["details"] == {"indexed": True, "collection": "SSCI"}


@pytest.mark.asyncio
async def test_crossref_verification_runs_concurrently(db_session):
    from app.api.verification import _run_verification_background

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    for n in range(10):
        await seed_paper(db_session, project, title=f"Paper {n}", doi=f"10.2/p{n}")
    job = await seed_job(db_session, project, JobType.qa)

    state = {"in_flight": 0, "max_in_flight": 0}

    async def _slow_verify(_self, _doi):
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        await asyncio.sleep(0.05)
        state["in_flight"] -= 1
        return None

    factory, engine = make_session_factory()
    with patch("app.api.verification.CrossRefClient.verify_doi", new=_slow_verify):
        await _run_verification_background(
            job_id=job.id,
            project_id=project.id,
            paper_ids=[],
            session_factory=factory,
        )
    await engine.dispose()

    assert state["max_in_flight"] > 1, "CrossRef verification is still serial"
    assert state["max_in_flight"] <= 8, "must stay bounded by verification_concurrency"

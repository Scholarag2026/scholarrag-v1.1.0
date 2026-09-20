"""Foundational-work verification must fan out, bounded, instead of running serially."""

import asyncio
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from app.models.analysis_job import JobStatus, JobType  # noqa: E402
from app.schemas.field_foundations import FoundationalWork  # noqa: E402
from tests.t5_fixtures import (  # noqa: E402
    FakeAgent,
    make_session_factory,
    seed_job,
    seed_project,
    seed_user,
)


class _Works:
    def __init__(self, works):
        self.works = works


def _work(n: int) -> FoundationalWork:
    return FoundationalWork(
        suggested_title=f"Foundational Work {n}",
        suggested_authors=["Author A"],
        suggested_year=1990 + n,
        why_essential="Seminal.",
    )


@pytest.mark.asyncio
async def test_works_are_verified_concurrently(db_session):
    from app.services.field_foundations import run_field_foundations

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    job = await seed_job(db_session, project, JobType.field_foundations)

    state = {"in_flight": 0, "max_in_flight": 0}

    async def _slow_verify(work):
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        await asyncio.sleep(0.05)
        state["in_flight"] -= 1
        return work

    agent = FakeAgent(output=_Works([_work(n) for n in range(12)]))
    factory, engine = make_session_factory()
    with patch(
        "app.services.field_foundations.get_field_foundations_agent", return_value=agent
    ), patch("app.services.field_foundations._verify_work", new=_slow_verify):
        await run_field_foundations(
            project_id=project.id,
            job_id=job.id,
            session_factory=factory,
            topic="AI tutoring",
            research_questions=["Does it work?"],
        )
    await engine.dispose()

    assert state["max_in_flight"] > 1, "field foundations verification is still serial"
    assert state["max_in_flight"] <= 4, "must stay bounded by analysis_concurrency"

    await db_session.refresh(job)
    assert job.status == JobStatus.completed, job.error
    assert job.result["total_count"] == 12
    assert len(job.result["works"]) == 12


@pytest.mark.asyncio
async def test_field_foundations_stops_when_cancelled(db_session):
    from app.services.field_foundations import run_field_foundations

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    job = await seed_job(db_session, project, JobType.field_foundations)

    async def _verify(work):
        await asyncio.sleep(0.01)
        return work

    agent = FakeAgent(output=_Works([_work(n) for n in range(12)]))
    factory, engine = make_session_factory()
    with patch(
        "app.services.field_foundations.get_field_foundations_agent", return_value=agent
    ), patch("app.services.field_foundations._verify_work", new=_verify), patch(
        "app.services.task.should_abort", new_callable=AsyncMock, return_value=True
    ):
        await run_field_foundations(
            project_id=project.id,
            job_id=job.id,
            session_factory=factory,
            topic="AI tutoring",
            research_questions=["Does it work?"],
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status == JobStatus.cancelled

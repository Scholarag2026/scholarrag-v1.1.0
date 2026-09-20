"""Quality-scoring batches must run concurrently and stop when the job is cancelled."""

import asyncio
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from app.models.analysis_job import JobStatus, JobType  # noqa: E402
from app.schemas.analysis import BatchAnalysisResult, PaperAnalysis  # noqa: E402
from tests.t5_fixtures import (  # noqa: E402
    make_session_factory,
    seed_job,
    seed_paper,
    seed_project,
    seed_user,
)


def _analysis_for(paper_id: str) -> PaperAnalysis:
    return PaperAnalysis(
        paper_id=paper_id,
        relevance_score=0.5,
        quality_score=0.5,
        key_findings=["F"],
        methodology="M",
        methodology_rigor="high",
        limitations=[],
        theories_used=[],
    )


class _ProbeAgent:
    """Records maximum in-flight calls and echoes back one analysis per input paper."""

    def __init__(self, state):
        self.state = state

    async def run(self, prompt, deps=None):
        self.state["in_flight"] += 1
        self.state["max_in_flight"] = max(
            self.state["max_in_flight"], self.state["in_flight"]
        )
        await asyncio.sleep(0.05)
        self.state["in_flight"] -= 1
        ids = [
            line.split("- ID: ", 1)[1].strip()
            for line in prompt.splitlines()
            if line.startswith("- ID: ")
        ]
        return type(
            "R",
            (),
            {"output": BatchAnalysisResult(analyses=[_analysis_for(i) for i in ids])},
        )()


@pytest.mark.asyncio
async def test_quality_scoring_batches_run_concurrently(db_session):
    from app.services.analysis import run_quality_scoring

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    for n in range(20):  # analysis_batch_size=5 -> 4 batches
        await seed_paper(db_session, project, title=f"Paper {n}", abstract="Abstract.")
    job = await seed_job(db_session, project, JobType.quality_scoring)

    state = {"in_flight": 0, "max_in_flight": 0}
    factory, engine = make_session_factory()
    with patch(
        "app.services.analysis.get_analysis_agent", return_value=_ProbeAgent(state)
    ):
        await run_quality_scoring(
            project_id=project.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    assert state["max_in_flight"] > 1, "batches are still processed strictly serially"
    assert state["max_in_flight"] <= 4, "must stay bounded by analysis_concurrency"

    await db_session.refresh(job)
    assert job.status == JobStatus.completed, job.error
    assert job.result["analyzed"] == 20


@pytest.mark.asyncio
async def test_quality_scoring_stops_when_cancelled(db_session):
    from app.services.analysis import run_quality_scoring

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    for n in range(20):
        await seed_paper(db_session, project, title=f"Paper {n}", abstract="Abstract.")
    job = await seed_job(db_session, project, JobType.quality_scoring)

    state = {"in_flight": 0, "max_in_flight": 0}
    factory, engine = make_session_factory()
    with patch(
        "app.services.analysis.get_analysis_agent", return_value=_ProbeAgent(state)
    ), patch(
        "app.services.task.should_abort", new_callable=AsyncMock, return_value=True
    ):
        await run_quality_scoring(
            project_id=project.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status == JobStatus.cancelled

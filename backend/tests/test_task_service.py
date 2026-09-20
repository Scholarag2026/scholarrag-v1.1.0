"""task_service cancellation primitives."""

import logging
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.services import task as task_service


async def _seed(client, db_session, status: JobStatus = JobStatus.running) -> tuple[UUID, UUID]:
    """Register a user + project and insert one job. Returns (job_id, user_id)."""
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"svc-{uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Svc User",
            "expertise_level": "researcher",
        },
    )
    token = res.json()["access_token"]
    me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    user_id = UUID(me.json()["id"])
    proj = await client.post(
        "/api/v1/projects",
        json={"title": "Svc Project"},
        headers={"Authorization": f"Bearer {token}"},
    )
    project_id = UUID(proj.json()["id"])
    job_id = uuid4()
    db_session.add(
        AnalysisJob(
            id=job_id,
            project_id=project_id,
            user_id=user_id,
            job_type=JobType.smart_search,
            status=status,
            progress=0.5,
        )
    )
    await db_session.commit()
    return job_id, user_id


def test_terminal_statuses_contains_the_three_end_states():
    assert task_service.TERMINAL_STATUSES == frozenset(
        {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}
    )


@pytest.mark.asyncio
async def test_cancel_job_marks_a_running_job_cancelled(client, db_session):
    job_id, user_id = await _seed(client, db_session, JobStatus.running)

    job = await task_service.cancel_job(db_session, job_id, user_id)

    assert job.status is JobStatus.cancelled
    assert job.error == "Cancelled by user"


@pytest.mark.asyncio
async def test_cancel_job_marks_a_pending_job_cancelled(client, db_session):
    job_id, user_id = await _seed(client, db_session, JobStatus.pending)

    job = await task_service.cancel_job(db_session, job_id, user_id)

    assert job.status is JobStatus.cancelled


@pytest.mark.asyncio
async def test_cancel_job_409s_on_a_completed_job(client, db_session):
    job_id, user_id = await _seed(client, db_session, JobStatus.completed)

    with pytest.raises(HTTPException) as exc:
        await task_service.cancel_job(db_session, job_id, user_id)

    assert exc.value.status_code == 409
    assert "completed" in exc.value.detail


@pytest.mark.asyncio
async def test_cancel_job_409s_on_an_already_cancelled_job(client, db_session):
    job_id, user_id = await _seed(client, db_session, JobStatus.cancelled)

    with pytest.raises(HTTPException) as exc:
        await task_service.cancel_job(db_session, job_id, user_id)

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_cancel_job_404s_for_another_users_job(client, db_session):
    job_id, _user_id = await _seed(client, db_session, JobStatus.running)

    with pytest.raises(HTTPException) as exc:
        await task_service.cancel_job(db_session, job_id, uuid4())

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_should_abort_is_false_for_a_running_job(client, db_session):
    job_id, _user_id = await _seed(client, db_session, JobStatus.running)

    assert await task_service.should_abort(db_session, job_id) is False


@pytest.mark.asyncio
async def test_should_abort_is_true_after_cancellation(client, db_session):
    job_id, user_id = await _seed(client, db_session, JobStatus.running)
    await task_service.cancel_job(db_session, job_id, user_id)

    assert await task_service.should_abort(db_session, job_id) is True


@pytest.mark.asyncio
async def test_should_abort_is_true_for_a_missing_job(db_session):
    assert await task_service.should_abort(db_session, uuid4()) is True


@pytest.mark.asyncio
async def test_should_abort_sees_a_cancel_committed_by_another_session(client, db_session):
    """Wave-2 loops call should_abort with their own session; it must not read stale state."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from tests.conftest import TEST_DATABASE_URL

    job_id, user_id = await _seed(client, db_session, JobStatus.running)

    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with factory() as worker_session:
            # The "background worker" has already read the job once.
            assert await task_service.should_abort(worker_session, job_id) is False
            # The request handler cancels it on a different session and commits.
            await task_service.cancel_job(db_session, job_id, user_id)
            # The worker's next check must observe the cancellation.
            assert await task_service.should_abort(worker_session, job_id) is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_update_job_status_cannot_resurrect_a_cancelled_job(client, db_session):
    job_id, user_id = await _seed(client, db_session, JobStatus.running)
    await task_service.cancel_job(db_session, job_id, user_id)

    await task_service.update_job_status(
        db_session, job_id, JobStatus.completed, progress=1.0, result={"papers": 3}
    )

    row = await db_session.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = row.scalar_one()
    assert job.status is JobStatus.cancelled
    assert job.result is None


@pytest.mark.asyncio
async def test_update_job_status_still_updates_a_running_job(client, db_session):
    job_id, _user_id = await _seed(client, db_session, JobStatus.running)

    await task_service.update_job_status(
        db_session, job_id, JobStatus.completed, progress=1.0, result={"papers": 3}
    )

    row = await db_session.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = row.scalar_one()
    assert job.status is JobStatus.completed
    assert job.result == {"papers": 3}


@pytest.mark.asyncio
async def test_update_job_status_truncates_an_overlong_progress_message(client, db_session):
    """AnalysisJob.progress_message is String(500); a longer value must be truncated here
    (not left to raise StringDataRightTruncationError at commit, which would abort the
    whole status update — including a completed job's ``result`` — regressed in practice by
    a Smart Search completion message that embedded a verbatim OpenAlex rate-limit body)."""
    job_id, _user_id = await _seed(client, db_session, JobStatus.running)
    overlong = "Completed: 290 papers after 18 round(s) (retrieval failed: " + ("x" * 600)

    await task_service.update_job_status(
        db_session, job_id, JobStatus.completed, progress=1.0,
        progress_message=overlong, result={"papers": 290},
    )

    row = await db_session.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = row.scalar_one()
    assert job.status is JobStatus.completed
    assert job.result == {"papers": 290}
    assert len(job.progress_message) == task_service.PROGRESS_MESSAGE_MAX_LEN
    assert job.progress_message.endswith("…")
    assert job.progress_message.startswith("Completed: 290 papers after 18 round(s)")


@pytest.mark.asyncio
async def test_status_transitions_are_logged_at_info(client, db_session, caplog):
    job_id, _user_id = await _seed(client, db_session, JobStatus.pending)

    with caplog.at_level(logging.INFO, logger="app.services.task"):
        await task_service.update_job_status(db_session, job_id, JobStatus.running, progress=0.1)

    messages = [r.getMessage() for r in caplog.records if r.name == "app.services.task"]
    assert any("job.status" in m and str(job_id) in m for m in messages), messages

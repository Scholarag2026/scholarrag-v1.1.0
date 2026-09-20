"""Tests for job durability: staleness predicate, boot sweep, reaper, shutdown interrupt."""

import asyncio
import time
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.models.project import Project
from app.models.user import User
from tests.conftest import TEST_DATABASE_URL


@pytest.fixture
async def session_factory():
    """A session factory on its own engine — mimics the app's async_session_factory."""
    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def _make_project(db) -> tuple[uuid.UUID, uuid.UUID]:
    user = User(
        email=f"durability-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="x",
        name="Durability User",
    )
    db.add(user)
    await db.flush()
    project = Project(user_id=user.id, title="Durability Project")
    db.add(project)
    await db.flush()
    return project.id, user.id


async def _make_job(
    db,
    project_id: uuid.UUID,
    user_id: uuid.UUID,
    status: JobStatus,
    age_minutes: float,
    job_type: JobType = JobType.graph_building,
) -> uuid.UUID:
    """Insert a job and back-date its updated_at by *age_minutes*."""
    job = AnalysisJob(
        project_id=project_id,
        user_id=user_id,
        job_type=job_type,
        status=status,
        progress=0.0,
    )
    db.add(job)
    await db.flush()
    stamp = datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
    await db.execute(
        update(AnalysisJob).where(AnalysisJob.id == job.id).values(updated_at=stamp)
    )
    await db.commit()
    return job.id


async def _reload(db, job_id: uuid.UUID) -> AnalysisJob:
    job = await db.get(AnalysisJob, job_id)
    await db.refresh(job)
    return job


async def test_fail_stale_jobs_ignores_a_fresh_running_job(db_session):
    """A sibling worker's live job must survive; fail_stale_jobs must not touch it."""
    from app.services.task import fail_stale_jobs

    project_id, user_id = await _make_project(db_session)
    fresh = await _make_job(db_session, project_id, user_id, JobStatus.running, 1)

    failed = await fail_stale_jobs(db_session, older_than_minutes=10)

    assert failed == []
    assert (await _reload(db_session, fresh)).status == JobStatus.running


async def test_fail_stale_jobs_fails_an_abandoned_running_job(db_session):
    from app.services.task import STALE_JOB_ERROR, fail_stale_jobs

    project_id, user_id = await _make_project(db_session)
    stale = await _make_job(db_session, project_id, user_id, JobStatus.running, 30)

    failed = await fail_stale_jobs(db_session, older_than_minutes=10)

    assert failed == [stale]
    job = await _reload(db_session, stale)
    assert job.status == JobStatus.failed
    assert job.error == STALE_JOB_ERROR


async def test_fail_stale_jobs_fails_an_abandoned_pending_job(db_session):
    from app.services.task import fail_stale_jobs

    project_id, user_id = await _make_project(db_session)
    stale = await _make_job(db_session, project_id, user_id, JobStatus.pending, 30)

    assert await fail_stale_jobs(db_session, older_than_minutes=10) == [stale]


async def test_fail_stale_jobs_treats_cancelled_as_terminal(db_session):
    """A cancelled job stays cancelled, it is never rewritten to failed."""
    from app.services.task import fail_stale_jobs

    project_id, user_id = await _make_project(db_session)
    cancelled = await _make_job(db_session, project_id, user_id, JobStatus.cancelled, 120)

    assert await fail_stale_jobs(db_session, older_than_minutes=10) == []
    assert (await _reload(db_session, cancelled)).status == JobStatus.cancelled


async def test_fail_stale_jobs_scopes_to_project_and_type(db_session):
    """T4's 409 guard needs to reap only the project/job_type it is about to start."""
    from app.services.task import fail_stale_jobs

    project_a, user_id = await _make_project(db_session)
    project_b, _ = await _make_project(db_session)
    target = await _make_job(db_session, project_a, user_id, JobStatus.running, 30)
    other_project = await _make_job(db_session, project_b, user_id, JobStatus.running, 30)
    other_type = await _make_job(
        db_session, project_a, user_id, JobStatus.running, 30, JobType.smart_search
    )

    failed = await fail_stale_jobs(
        db_session,
        older_than_minutes=10,
        project_id=project_a,
        job_type=JobType.graph_building,
    )

    assert failed == [target]
    assert (await _reload(db_session, other_project)).status == JobStatus.running
    assert (await _reload(db_session, other_type)).status == JobStatus.running


async def test_fail_stale_jobs_unregisters_reaped_ids(db_session):
    from app.services import job_registry
    from app.services.task import fail_stale_jobs

    project_id, user_id = await _make_project(db_session)
    stale = await _make_job(db_session, project_id, user_id, JobStatus.running, 30)
    job_registry.clear()
    job_registry.mark_in_flight(stale)

    await fail_stale_jobs(db_session, older_than_minutes=10)

    assert job_registry.in_flight_job_ids() == []


def test_stale_job_settings_defaults():
    from app.config import settings

    assert settings.job_stale_after_minutes == 10
    assert settings.job_reaper_interval_seconds == 300


async def test_sweep_stale_jobs_uses_the_boot_error_message(db_session, session_factory):
    from app.services.task import BOOT_SWEEP_ERROR, sweep_stale_jobs

    project_id, user_id = await _make_project(db_session)
    stale = await _make_job(db_session, project_id, user_id, JobStatus.running, 30)
    fresh = await _make_job(db_session, project_id, user_id, JobStatus.running, 1)

    swept = await sweep_stale_jobs(session_factory)

    assert swept == [stale]
    assert (await _reload(db_session, stale)).error == BOOT_SWEEP_ERROR
    assert (await _reload(db_session, fresh)).status == JobStatus.running


async def test_reaper_fails_abandoned_jobs_then_stops_on_cancel(db_session, session_factory):
    from app.services.task import stale_job_reaper

    project_id, user_id = await _make_project(db_session)
    stale = await _make_job(db_session, project_id, user_id, JobStatus.running, 30)

    task = asyncio.create_task(
        stale_job_reaper(session_factory, interval_seconds=0.05, older_than_minutes=10)
    )
    try:
        job = await _reload(db_session, stale)
        for _ in range(100):
            if job.status == JobStatus.failed:
                break
            await asyncio.sleep(0.05)
            job = await _reload(db_session, stale)
        assert job.status == JobStatus.failed
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    assert task.cancelled()


async def test_reaper_survives_a_failing_iteration(session_factory):
    """One bad iteration must not kill the reaper for the life of the process."""
    from app.services import task as task_mod

    calls = {"n": 0}

    async def exploding_fail_stale_jobs(db, **kwargs):
        calls["n"] += 1
        raise RuntimeError("boom")

    original = task_mod.fail_stale_jobs
    task_mod.fail_stale_jobs = exploding_fail_stale_jobs
    try:
        task = asyncio.create_task(
            task_mod.stale_job_reaper(session_factory, interval_seconds=0.02)
        )
        # Wait for the CONDITION, not a fixed interval: each reaper iteration opens
        # a fresh DB session, which can take >100ms when the full suite has the
        # machine and Postgres under load — a fixed 0.2s window flakes there.
        deadline = time.monotonic() + 5.0
        while calls["n"] < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    finally:
        task_mod.fail_stale_jobs = original

    assert calls["n"] >= 2


async def test_interrupt_in_flight_jobs_fails_this_process_jobs(db_session, session_factory):
    from app.services import job_registry
    from app.services.task import SHUTDOWN_ERROR, interrupt_in_flight_jobs

    project_id, user_id = await _make_project(db_session)
    running = await _make_job(db_session, project_id, user_id, JobStatus.running, 0)
    done = await _make_job(db_session, project_id, user_id, JobStatus.completed, 0)
    job_registry.clear()
    job_registry.mark_in_flight(running)
    job_registry.mark_in_flight(done)

    interrupted = await interrupt_in_flight_jobs(session_factory)

    assert interrupted == [running]
    job = await _reload(db_session, running)
    assert job.status == JobStatus.failed
    assert job.error == SHUTDOWN_ERROR
    assert (await _reload(db_session, done)).status == JobStatus.completed
    assert job_registry.in_flight_job_ids() == []


async def test_interrupt_in_flight_jobs_is_a_noop_when_registry_empty(session_factory):
    from app.services import job_registry
    from app.services.task import interrupt_in_flight_jobs

    job_registry.clear()
    assert await interrupt_in_flight_jobs(session_factory) == []

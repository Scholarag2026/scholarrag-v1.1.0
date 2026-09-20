import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.services import job_registry

logger = logging.getLogger("app.services.task")

#: Statuses a job can never leave. Consumed by the cancel endpoint, the stale-job
#: reaper (track T7) and the 409 concurrency guards.
TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}
)

ACTIVE_STATUSES: tuple[JobStatus, ...] = (JobStatus.pending, JobStatus.running)

#: Must match ``AnalysisJob.progress_message`` (``String(500)``). A message longer than
#: this would raise ``StringDataRightTruncationError`` at commit and abort the whole
#: status update — including a completed job's ``result`` — so it is truncated here
#: instead of at the column. Seen in practice when a Smart Search completion message
#: embeds a verbatim provider error (e.g. an OpenAlex rate-limit body) after a run that
#: had already found hundreds of papers over many rounds.
PROGRESS_MESSAGE_MAX_LEN = 500

STALE_JOB_ERROR = (
    "Job was abandoned — no progress was recorded within the staleness window, "
    "so it was marked failed by the stale-job reaper."
)
BOOT_SWEEP_ERROR = "Server restarted — job was interrupted"
SHUTDOWN_ERROR = "Server shut down — job was interrupted"


async def create_job(
    db: AsyncSession,
    project_id: UUID,
    user_id: UUID,
    job_type: JobType,
    dataset_id: UUID | None = None,
) -> AnalysisJob:
    job = AnalysisJob(
        project_id=project_id,
        user_id=user_id,
        job_type=job_type,
        status=JobStatus.pending,
        progress=0.0,
        dataset_id=dataset_id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    logger.info(
        "job.created job_id=%s job_type=%s project_id=%s", job.id, job_type.value, project_id
    )
    job_registry.mark_in_flight(job.id)
    return job


async def get_job(db: AsyncSession, job_id: UUID, user_id: UUID) -> AnalysisJob:
    result = await db.execute(
        select(AnalysisJob).where(AnalysisJob.id == job_id, AnalysisJob.user_id == user_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")
    return job


async def cancel_job(db: AsyncSession, job_id: UUID, user_id: UUID) -> AnalysisJob:
    """Mark a pending/running job as cancelled.

    Raises 404 when the job does not exist or belongs to another user, and 409 when
    the job has already reached a terminal status.
    """
    job = await get_job(db, job_id, user_id)
    if job.status in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Task is already {job.status.value} and cannot be cancelled",
        )
    job.status = JobStatus.cancelled
    job.error = "Cancelled by user"
    await db.commit()
    await db.refresh(job)
    logger.info("job.cancelled job_id=%s job_type=%s", job.id, job.job_type.value)
    return job


async def should_abort(db: AsyncSession, job_id: UUID) -> bool:
    """Return True when a long-running loop for ``job_id`` should stop immediately.

    True when the job row has been deleted or its status is ``cancelled``.

    Safe to call with any session, including one opened fresh from
    ``async_session_factory``: it selects only the status column, so it never returns a
    stale ORM identity-map copy, and PostgreSQL READ COMMITTED gives every statement a
    fresh snapshot, so a cancellation committed by the HTTP request handler is visible
    to a background task's long-lived session on its next check.

    Call this between units of work (round, query, paper, analysis item) — never inside
    a tight inner loop, it costs one round trip.
    """
    result = await db.execute(select(AnalysisJob.status).where(AnalysisJob.id == job_id))
    status = result.scalar_one_or_none()
    return status is None or status == JobStatus.cancelled


async def update_job_status(
    db: AsyncSession,
    job_id: UUID,
    status: JobStatus,
    progress: float = 0.0,
    progress_message: str | None = None,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    job_result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        return
    previous = job.status
    if previous == JobStatus.cancelled and status != JobStatus.cancelled:
        # A background task finished after the user cancelled: never resurrect the job.
        logger.info(
            "job.update_ignored job_id=%s reason=cancelled attempted_status=%s",
            job_id,
            status.value,
        )
        return
    job.status = status
    job.progress = progress
    if progress_message is not None:
        if len(progress_message) > PROGRESS_MESSAGE_MAX_LEN:
            progress_message = progress_message[: PROGRESS_MESSAGE_MAX_LEN - 1] + "…"
        job.progress_message = progress_message
    if result is not None:
        job.result = result
    if error is not None:
        job.error = error
    await db.commit()
    if status != previous:
        logger.info(
            "job.status job_id=%s job_type=%s %s->%s progress=%.2f",
            job_id,
            job.job_type.value,
            previous.value,
            status.value,
            progress,
        )
    if status in ACTIVE_STATUSES:
        job_registry.mark_in_flight(job_id)
    else:
        job_registry.mark_done(job_id)


async def fail_stale_jobs(
    db: AsyncSession,
    *,
    older_than_minutes: int | None = None,
    project_id: UUID | None = None,
    job_type: JobType | None = None,
    error: str = STALE_JOB_ERROR,
) -> list[UUID]:
    """Fail pending/running jobs that have shown no progress for the staleness window.

    The ``updated_at`` predicate is the whole point (D12a): ``AnalysisJob.updated_at`` is
    bumped by every ``update_job_status`` progress write, so a job that is still being
    worked on by a sibling uvicorn worker is never touched. Terminal statuses
    (``completed``/``failed``/``cancelled``) are excluded by ``ACTIVE_STATUSES``.

    Optional *project_id* / *job_type* narrow the sweep so a 409 concurrency guard can
    reap only the slot it is about to occupy.

    Commits its own transaction. Returns the ids of the jobs it failed.
    """
    minutes = (
        settings.job_stale_after_minutes if older_than_minutes is None else older_than_minutes
    )
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=minutes)

    conditions = [
        AnalysisJob.status.in_(ACTIVE_STATUSES),
        AnalysisJob.updated_at < cutoff,
    ]
    if project_id is not None:
        conditions.append(AnalysisJob.project_id == project_id)
    if job_type is not None:
        conditions.append(AnalysisJob.job_type == job_type)

    result = await db.execute(
        update(AnalysisJob)
        .where(*conditions)
        .values(status=JobStatus.failed, error=error, updated_at=now)
        .returning(AnalysisJob.id)
        .execution_options(synchronize_session=False)
    )
    job_ids = [row[0] for row in result.all()]
    await db.commit()

    for job_id in job_ids:
        job_registry.mark_done(job_id)
    return job_ids


async def guard_no_active_job(
    db: AsyncSession,
    *,
    project_id: UUID,
    job_type: JobType,
    conflict_detail: str,
    dataset_id: UUID | None = None,
) -> None:
    """Shared 409 concurrency guard (D12c).

    Reaps stale pending/running jobs of *job_type* for *project_id* via
    ``fail_stale_jobs`` first, so a worker that died mid-run never blocks a retry for
    longer than the staleness window, then raises ``HTTPException(409)`` with
    *conflict_detail* if a genuinely active (non-stale) job of that type remains.

    ``fail_stale_jobs`` only takes *project_id* / *job_type*, so the reap always runs
    project-wide for that job type — harmless, since it only touches jobs already past
    the staleness cutoff. Pass *dataset_id* to additionally scope the post-reap
    conflict *check* to a single dataset (e.g. qualitative/quantitative jobs, which are
    per-dataset), so concurrent jobs on other datasets in the same project are left
    alone.

    Every "already in progress" guard across the API should call this instead of
    hand-rolling a ``status.in_(ACTIVE_STATUSES)`` select — the blanket check alone
    cannot distinguish a live job from one whose worker vanished.
    """
    await fail_stale_jobs(db, project_id=project_id, job_type=job_type)

    conditions = [
        AnalysisJob.project_id == project_id,
        AnalysisJob.job_type == job_type,
        AnalysisJob.status.in_(ACTIVE_STATUSES),
    ]
    if dataset_id is not None:
        conditions.append(AnalysisJob.dataset_id == dataset_id)

    existing = await db.execute(select(AnalysisJob.id).where(*conditions))
    if existing.first() is not None:
        raise HTTPException(status_code=409, detail=conflict_detail)


async def sweep_stale_jobs(session_factory) -> list[UUID]:
    """Boot-time sweep (D12a).

    Only jobs with no progress for ``settings.job_stale_after_minutes`` are failed, so a
    booting worker can never kill a job that its sibling worker is actively running.
    """
    async with session_factory() as db:
        return await fail_stale_jobs(db, error=BOOT_SWEEP_ERROR)


async def stale_job_reaper(
    session_factory,
    *,
    interval_seconds: float | None = None,
    older_than_minutes: int | None = None,
) -> None:
    """Periodically fail abandoned jobs (D12b). Runs until cancelled.

    Started as an asyncio task by the FastAPI lifespan. A failing iteration is logged and
    the loop continues; ``asyncio.CancelledError`` propagates so shutdown is immediate.
    """
    interval = (
        settings.job_reaper_interval_seconds if interval_seconds is None else interval_seconds
    )
    while True:
        try:
            await asyncio.sleep(interval)
            async with session_factory() as db:
                reaped = await fail_stale_jobs(db, older_than_minutes=older_than_minutes)
            if reaped:
                logger.warning(
                    "Stale-job reaper failed %d abandoned job(s): %s",
                    len(reaped),
                    [str(job_id) for job_id in reaped],
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Stale-job reaper iteration failed")


async def mark_jobs_interrupted(db: AsyncSession, job_ids: Sequence[UUID]) -> list[UUID]:
    """Fail the given jobs if they are still pending/running. Commits.

    Terminal jobs (including ``cancelled``) are left untouched.
    """
    if not job_ids:
        return []
    now = datetime.now(timezone.utc)
    result = await db.execute(
        update(AnalysisJob)
        .where(
            AnalysisJob.id.in_(list(job_ids)),
            AnalysisJob.status.in_(ACTIVE_STATUSES),
        )
        .values(status=JobStatus.failed, error=SHUTDOWN_ERROR, updated_at=now)
        .returning(AnalysisJob.id)
        .execution_options(synchronize_session=False)
    )
    interrupted = [row[0] for row in result.all()]
    await db.commit()
    for job_id in interrupted:
        job_registry.mark_done(job_id)
    return interrupted


async def interrupt_in_flight_jobs(session_factory) -> list[UUID]:
    """Graceful-shutdown hook: fail the jobs this process still has in flight."""
    job_ids = job_registry.in_flight_job_ids()
    if not job_ids:
        return []
    async with session_factory() as db:
        interrupted = await mark_jobs_interrupted(db, job_ids)
    job_registry.clear()
    return interrupted

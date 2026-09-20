"""In-process registry of jobs this uvicorn worker is currently running.

Jobs are dispatched with Starlette ``BackgroundTasks`` and therefore live in the same
process as the request handler (an out-of-process worker is a future migration).
This registry lets the lifespan shutdown hook mark *this* process's in-flight jobs as
interrupted immediately, instead of leaving them stuck at ``running`` until the
stale-job reaper notices ten minutes later.

Deliberately tiny and lock-free: every mutation happens on the single asyncio event
loop of one worker process.
"""

from __future__ import annotations

from uuid import UUID

_IN_FLIGHT: set[UUID] = set()


def mark_in_flight(job_id: UUID) -> None:
    """Record *job_id* as pending/running in this process."""
    _IN_FLIGHT.add(job_id)


def mark_done(job_id: UUID) -> None:
    """Forget *job_id* — it reached a terminal status or was reaped."""
    _IN_FLIGHT.discard(job_id)


def in_flight_job_ids() -> list[UUID]:
    """Return a snapshot of job ids still in flight in this process."""
    return list(_IN_FLIGHT)


def clear() -> None:
    """Drop all tracked ids (used after a shutdown sweep and by tests)."""
    _IN_FLIGHT.clear()

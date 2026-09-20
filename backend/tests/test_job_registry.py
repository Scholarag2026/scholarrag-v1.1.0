"""Tests for the in-process in-flight job registry (graceful shutdown support)."""

import uuid

from app.models.analysis_job import JobStatus, JobType
from app.models.project import Project
from app.models.user import User


async def _make_project(db) -> tuple[uuid.UUID, uuid.UUID]:
    user = User(
        email=f"registry-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="x",
        name="Registry User",
    )
    db.add(user)
    await db.flush()
    project = Project(user_id=user.id, title="Registry Project")
    db.add(project)
    await db.flush()
    return project.id, user.id


async def test_create_job_marks_job_in_flight(db_session):
    from app.services import job_registry
    from app.services.task import create_job

    job_registry.clear()
    project_id, user_id = await _make_project(db_session)
    job = await create_job(db_session, project_id, user_id, JobType.graph_building)

    assert job.id in job_registry.in_flight_job_ids()


async def test_running_status_keeps_job_registered(db_session):
    from app.services import job_registry
    from app.services.task import create_job, update_job_status

    job_registry.clear()
    project_id, user_id = await _make_project(db_session)
    job = await create_job(db_session, project_id, user_id, JobType.graph_building)
    await update_job_status(db_session, job.id, JobStatus.running, progress=0.2)

    assert job.id in job_registry.in_flight_job_ids()


async def test_completed_status_unregisters_job(db_session):
    from app.services import job_registry
    from app.services.task import create_job, update_job_status

    job_registry.clear()
    project_id, user_id = await _make_project(db_session)
    job = await create_job(db_session, project_id, user_id, JobType.graph_building)
    await update_job_status(db_session, job.id, JobStatus.completed, progress=1.0)

    assert job.id not in job_registry.in_flight_job_ids()


async def test_cancelled_status_unregisters_job(db_session):
    from app.services import job_registry
    from app.services.task import create_job, update_job_status

    job_registry.clear()
    project_id, user_id = await _make_project(db_session)
    job = await create_job(db_session, project_id, user_id, JobType.graph_building)
    await update_job_status(db_session, job.id, JobStatus.cancelled)

    assert job.id not in job_registry.in_flight_job_ids()


def test_registry_clear_empties_snapshot():
    from app.services import job_registry

    job_registry.clear()
    job_id = uuid.uuid4()
    job_registry.mark_in_flight(job_id)
    assert job_registry.in_flight_job_ids() == [job_id]
    job_registry.clear()
    assert job_registry.in_flight_job_ids() == []

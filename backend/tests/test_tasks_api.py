"""Tests for the tasks API router, including cancellation."""

from uuid import UUID, uuid4

import pytest

from app.models.analysis_job import AnalysisJob, JobStatus, JobType


def test_tasks_route_exists():
    from app.api.tasks import router

    routes = [r.path for r in router.routes]
    assert any("/tasks/" in r for r in routes)


async def _register(client, email: str) -> str:
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Task User",
            "expertise_level": "researcher",
        },
    )
    assert res.status_code == 201, res.text
    return res.json()["access_token"]


async def _seed_job(client, db_session, token: str, status: JobStatus) -> UUID:
    me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    user_id = UUID(me.json()["id"])
    proj = await client.post(
        "/api/v1/projects",
        json={"title": "Task Project"},
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
    return job_id


@pytest.mark.asyncio
async def test_delete_task_cancels_a_running_job(client, db_session):
    token = await _register(client, "cancel-run@example.com")
    job_id = await _seed_job(client, db_session, token, JobStatus.running)

    res = await client.delete(
        f"/api/v1/tasks/{job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["id"] == str(job_id)
    assert body["status"] == "cancelled"
    assert body["error"] == "Cancelled by user"


@pytest.mark.asyncio
async def test_get_task_reports_cancelled_after_delete(client, db_session):
    token = await _register(client, "cancel-get@example.com")
    job_id = await _seed_job(client, db_session, token, JobStatus.running)

    await client.delete(f"/api/v1/tasks/{job_id}", headers={"Authorization": f"Bearer {token}"})
    res = await client.get(f"/api/v1/tasks/{job_id}", headers={"Authorization": f"Bearer {token}"})

    assert res.status_code == 200
    assert res.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_delete_task_409s_when_already_completed(client, db_session):
    token = await _register(client, "cancel-done@example.com")
    job_id = await _seed_job(client, db_session, token, JobStatus.completed)

    res = await client.delete(
        f"/api/v1/tasks/{job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 409
    assert "completed" in res.json()["detail"]


@pytest.mark.asyncio
async def test_delete_task_404s_for_an_unknown_id(client):
    token = await _register(client, "cancel-404@example.com")

    res = await client.delete(
        f"/api/v1/tasks/{uuid4()}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 404


@pytest.mark.asyncio
async def test_delete_task_404s_for_another_users_job(client, db_session):
    owner_token = await _register(client, "cancel-owner@example.com")
    intruder_token = await _register(client, "cancel-intruder@example.com")
    job_id = await _seed_job(client, db_session, owner_token, JobStatus.running)

    res = await client.delete(
        f"/api/v1/tasks/{job_id}",
        headers={"Authorization": f"Bearer {intruder_token}"},
    )

    assert res.status_code == 404


@pytest.mark.asyncio
async def test_delete_task_requires_authentication(client):
    res = await client.delete(f"/api/v1/tasks/{uuid4()}")

    assert res.status_code in (401, 403)

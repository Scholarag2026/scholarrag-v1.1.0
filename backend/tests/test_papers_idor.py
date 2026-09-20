"""Cross-tenant access tests for the paper upload-confirm endpoint."""

import uuid

import pytest

from app.models.analysis_job import AnalysisJob, JobStatus, JobType


async def register(client, email):
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "IDOR Test User",
            "expertise_level": "researcher",
        },
    )
    data = res.json()
    return {"Authorization": f"Bearer {data['access_token']}"}, uuid.UUID(data["user"]["id"])


async def create_project(client, headers, title="IDOR Project"):
    res = await client.post(
        "/api/v1/projects", json={"title": title}, headers=headers
    )
    return uuid.UUID(res.json()["id"])


@pytest.mark.asyncio
async def test_confirm_upload_rejects_other_users_job(client, db_session):
    victim_headers, victim_id = await register(client, f"victim-{uuid.uuid4().hex[:8]}@e.com")
    victim_project = await create_project(client, victim_headers, "Victim Project")

    attacker_headers, _ = await register(client, f"attacker-{uuid.uuid4().hex[:8]}@e.com")
    attacker_project = await create_project(client, attacker_headers, "Attacker Project")

    job = AnalysisJob(
        project_id=victim_project,
        user_id=victim_id,
        job_type=JobType.paper_upload,
        status=JobStatus.completed,
        result={
            "_fulltext_data": {
                "0": {"chunks": ["secret full text"], "char_count": 16},
            }
        },
    )
    db_session.add(job)
    await db_session.flush()

    res = await client.post(
        f"/api/v1/projects/{attacker_project}/papers/upload-confirm",
        json={
            "task_id": str(job.id),
            "papers": [
                {
                    "index": 0,
                    "title": "Stolen Paper",
                    "authors": [{"name": "Someone"}],
                    "year": 2020,
                }
            ],
        },
        headers=attacker_headers,
    )
    assert res.status_code == 404
    assert res.json()["detail"] == "Upload job not found"


@pytest.mark.asyncio
async def test_confirm_upload_accepts_own_job(client, db_session):
    headers, user_id = await register(client, f"owner-{uuid.uuid4().hex[:8]}@e.com")
    project_id = await create_project(client, headers, "Owner Project")

    job = AnalysisJob(
        project_id=project_id,
        user_id=user_id,
        job_type=JobType.paper_upload,
        status=JobStatus.completed,
        result={"_fulltext_data": {"0": {"chunks": ["my text"], "char_count": 7}}},
    )
    db_session.add(job)
    await db_session.flush()

    res = await client.post(
        f"/api/v1/projects/{project_id}/papers/upload-confirm",
        json={
            "task_id": str(job.id),
            "papers": [
                {
                    "index": 0,
                    "title": "My Paper",
                    "authors": [{"name": "Me"}],
                    "year": 2021,
                }
            ],
        },
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json()["imported_count"] == 1

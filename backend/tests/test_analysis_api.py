"""HTTP round-trip tests for the analysis API."""

from uuid import UUID

import pytest

from app.models.analysis_job import AnalysisJob, JobStatus, JobType


async def _register(client, email: str) -> str:
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Gap User",
            "expertise_level": "researcher",
        },
    )
    assert res.status_code == 201, res.text
    return res.json()["access_token"]


async def _me_id(client, token: str) -> UUID:
    res = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200, res.text
    return UUID(res.json()["id"])


async def _create_project(client, token: str, title: str = "Gap Project") -> UUID:
    res = await client.post(
        "/api/v1/projects",
        json={"title": title},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 201, res.text
    return UUID(res.json()["id"])


@pytest.mark.asyncio
async def test_gap_report_returns_200_null_when_not_generated(client):
    token = await _register(client, "gap-none@example.com")
    project_id = await _create_project(client, token)

    res = await client.get(
        f"/api/v1/projects/{project_id}/gap-report",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 200
    assert res.json() is None


@pytest.mark.asyncio
async def test_gap_report_returns_the_report_when_generated(client, db_session):
    token = await _register(client, "gap-some@example.com")
    user_id = await _me_id(client, token)
    project_id = await _create_project(client, token)

    payload = {
        "summary": "Two gaps found",
        "gaps": [{"title": "Gap A", "description": "d", "evidence": []}],
        "controversies": [],
        "suggested_questions": [],
        "theoretical_landscape": [],
    }
    db_session.add(
        AnalysisJob(
            project_id=project_id,
            user_id=user_id,
            job_type=JobType.gap_analysis,
            status=JobStatus.completed,
            progress=1.0,
            result=payload,
        )
    )
    await db_session.commit()

    res = await client.get(
        f"/api/v1/projects/{project_id}/gap-report",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 200
    assert res.json() == payload


@pytest.mark.asyncio
async def test_gap_report_still_404s_for_a_project_the_user_does_not_own(client):
    token = await _register(client, "gap-owner@example.com")
    other_token = await _register(client, "gap-intruder@example.com")
    project_id = await _create_project(client, token)

    res = await client.get(
        f"/api/v1/projects/{project_id}/gap-report",
        headers={"Authorization": f"Bearer {other_token}"},
    )

    assert res.status_code == 404

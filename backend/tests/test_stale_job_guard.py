"""Stale-aware 409 concurrency guard (D12c) — verified across representative endpoints.

`task_service.guard_no_active_job` reaps stale pending/running jobs of a given
project/job_type via `fail_stale_jobs` before deciding whether to 409. Every
"already in progress" guard in the API should call it instead of a blanket
`status.in_(ACTIVE_STATUSES)` check, otherwise a worker that died mid-run blocks a
retry for up to `job_stale_after_minutes + job_reaper_interval_seconds` (~15 min).

These tests exercise two representative HTTP endpoints backed by two different
router modules to prove the fix isn't a one-off.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import update

from app.models.analysis_job import AnalysisJob, JobStatus


async def _register(client, email: str) -> str:
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Stale Guard Tester",
            "expertise_level": "researcher",
        },
    )
    assert res.status_code == 201, res.text
    return res.json()["access_token"]


async def _create_project(client, token: str, title: str) -> str:
    res = await client.post(
        "/api/v1/projects",
        json={"title": title},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 201, res.text
    return res.json()["id"]


async def _backdate(db_session, job_id: str, minutes: float) -> None:
    await db_session.execute(
        update(AnalysisJob)
        .where(AnalysisJob.id == uuid.UUID(job_id))
        .values(
            status=JobStatus.running,
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=minutes),
        )
    )
    await db_session.commit()


@pytest.mark.asyncio
async def test_analyze_quality_reaps_a_stale_running_job(client, db_session):
    """A quality-scoring job with no heartbeat for >10 min must not block a retry."""
    token = await _register(client, f"quality-stale-{uuid.uuid4().hex[:8]}@example.com")
    project_id = await _create_project(client, token, "Quality Stale Project")

    with patch(
        "app.api.analysis.analysis_service.run_quality_scoring", new_callable=AsyncMock
    ):
        first = await client.post(
            f"/api/v1/projects/{project_id}/analyze-quality",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert first.status_code == 202, first.text
        stale_id = first.json()["task_id"]

        await _backdate(db_session, stale_id, minutes=45)

        second = await client.post(
            f"/api/v1/projects/{project_id}/analyze-quality",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert second.status_code == 202, second.text
    assert second.json()["task_id"] != stale_id

    reaped = await db_session.get(AnalysisJob, uuid.UUID(stale_id))
    await db_session.refresh(reaped)
    assert reaped.status is JobStatus.failed
    assert "abandoned" in (reaped.error or "").lower()


@pytest.mark.asyncio
async def test_acquire_full_texts_reaps_a_stale_running_job(client, db_session):
    """A fulltext_acquire job with no heartbeat for >10 min must not block a retry."""
    token = await _register(client, f"fulltext-stale-{uuid.uuid4().hex[:8]}@example.com")
    project_id = await _create_project(client, token, "Fulltext Stale Project")

    with patch("app.api.fulltext.acquire_full_texts", new_callable=AsyncMock):
        first = await client.post(
            f"/api/v1/projects/{project_id}/acquire-full-texts",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert first.status_code == 202, first.text
        stale_id = first.json()["task_id"]

        await _backdate(db_session, stale_id, minutes=45)

        second = await client.post(
            f"/api/v1/projects/{project_id}/acquire-full-texts",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert second.status_code == 202, second.text
    assert second.json()["task_id"] != stale_id

    reaped = await db_session.get(AnalysisJob, uuid.UUID(stale_id))
    await db_session.refresh(reaped)
    assert reaped.status is JobStatus.failed
    assert "abandoned" in (reaped.error or "").lower()

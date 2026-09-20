"""Tests for the graph API router."""

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

import httpx
from sqlalchemy import update

from app.models.analysis_job import AnalysisJob, JobStatus


def test_graph_routes_exist():
    from app.api.graph import router

    routes = [r.path for r in router.routes]
    assert any("build-graph" in r for r in routes)
    assert any("citation-graph" in r for r in routes)


async def _register(client) -> str:
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"graphapi-{uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Graph API Tester",
            "expertise_level": "researcher",
        },
    )
    return res.json()["access_token"]


async def _create_project(client, token: str) -> str:
    res = await client.post(
        "/api/v1/projects",
        json={"title": "Graph API Project"},
        headers={"Authorization": f"Bearer {token}"},
    )
    return res.json()["id"]


async def test_build_graph_returns_202(client):
    token = await _register(client)
    project_id = await _create_project(client, token)

    with patch(
        "app.api.graph.graph_service.build_citation_graph", new_callable=AsyncMock
    ):
        res = await client.post(
            f"/api/v1/projects/{project_id}/build-graph",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert res.status_code == 202
    uuid.UUID(res.json()["task_id"])


async def test_build_graph_409_while_a_live_build_is_running(client):
    token = await _register(client)
    project_id = await _create_project(client, token)

    with patch(
        "app.api.graph.graph_service.build_citation_graph", new_callable=AsyncMock
    ):
        first = await client.post(
            f"/api/v1/projects/{project_id}/build-graph",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert first.status_code == 202
        second = await client.post(
            f"/api/v1/projects/{project_id}/build-graph",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert second.status_code == 409
    assert "already in progress" in second.json()["detail"].lower()


async def test_build_graph_reaps_a_stale_running_job(client, db_session):
    """D12c: a 'running' row with no heartbeat for >10 min must not block rebuilds."""
    token = await _register(client)
    project_id = await _create_project(client, token)

    with patch(
        "app.api.graph.graph_service.build_citation_graph", new_callable=AsyncMock
    ):
        first = await client.post(
            f"/api/v1/projects/{project_id}/build-graph",
            headers={"Authorization": f"Bearer {token}"},
        )
        stale_id = uuid.UUID(first.json()["task_id"])

        await db_session.execute(
            update(AnalysisJob)
            .where(AnalysisJob.id == stale_id)
            .values(
                status=JobStatus.running,
                updated_at=datetime.now(timezone.utc) - timedelta(minutes=45),
            )
        )
        await db_session.commit()

        second = await client.post(
            f"/api/v1/projects/{project_id}/build-graph",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert second.status_code == 202
    assert uuid.UUID(second.json()["task_id"]) != stale_id

    reaped = await db_session.get(AnalysisJob, stale_id)
    await db_session.refresh(reaped)
    assert reaped.status is JobStatus.failed
    assert "no progress" in (reaped.error or "").lower()


async def test_expand_node_maps_upstream_failure_to_502(client):
    token = await _register(client)
    project_id = await _create_project(client, token)

    error = httpx.HTTPStatusError("503", request=httpx.Request("GET", "http://x"), response=None)
    with patch(
        "app.api.graph.graph_service.expand_node",
        new_callable=AsyncMock,
        side_effect=error,
    ):
        res = await client.post(
            f"/api/v1/projects/{project_id}/citation-graph/expand",
            json={"paper_id": str(uuid.uuid4())},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert res.status_code == 502
    assert "citation source" in res.json()["detail"].lower()

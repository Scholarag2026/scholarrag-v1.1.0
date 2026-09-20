"""Tests for deep search API endpoint."""

import os
import uuid

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch

import pytest


def test_deep_search_route_exists():
    """Verify the POST deep-search endpoint is registered."""
    from app.api.deep_search import router

    routes = [r.path for r in router.routes]
    assert "/projects/{project_id}/deep-search" in routes


async def get_auth_token(client):
    email = f"deepsearch-{uuid.uuid4().hex[:8]}@example.com"
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Deep Search User",
            "expertise_level": "researcher",
        },
    )
    return res.json()["access_token"]


async def create_project(client, token):
    res = await client.post(
        "/api/v1/projects",
        json={"title": "Deep Search Project"},
        headers={"Authorization": f"Bearer {token}"},
    )
    return res.json()["id"]


@pytest.mark.asyncio
async def test_deep_search_returns_202(client):
    """Happy path: POST deep-search returns 202 with a task_id."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    with patch(
        "app.api.deep_search.deep_search_service.run_deep_search",
        new_callable=AsyncMock,
    ):
        response = await client.post(
            f"/api/v1/projects/{project_id}/deep-search",
            json={"query": "machine learning meta-analysis"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    # task_id should be a valid UUID
    uuid.UUID(data["task_id"])


@pytest.mark.asyncio
async def test_deep_search_concurrency_guard_returns_409(client):
    """Second deep search while one is pending should return 409."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    with patch(
        "app.api.deep_search.deep_search_service.run_deep_search",
        new_callable=AsyncMock,
    ):
        # First request should succeed
        resp1 = await client.post(
            f"/api/v1/projects/{project_id}/deep-search",
            json={"query": "systematic review methods"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp1.status_code == 202

        # Second request should be rejected
        resp2 = await client.post(
            f"/api/v1/projects/{project_id}/deep-search",
            json={"query": "another search query"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp2.status_code == 409
        assert "already in progress" in resp2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_deep_search_empty_query_returns_422(client):
    """An empty query should fail validation with 422."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    response = await client.post(
        f"/api/v1/projects/{project_id}/deep-search",
        json={"query": ""},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422

"""Tests for seed papers API endpoint."""

import os
import uuid

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch

import pytest


def test_seed_expand_route_exists():
    """Verify the POST seed-expand endpoint is registered."""
    from app.api.seed_papers import router

    routes = [r.path for r in router.routes]
    assert "/projects/{project_id}/seed-expand" in routes


async def get_auth_token(client):
    email = f"seedtest-{uuid.uuid4().hex[:8]}@example.com"
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Seed Test User",
            "expertise_level": "researcher",
        },
    )
    return res.json()["access_token"]


async def create_project(client, token):
    res = await client.post(
        "/api/v1/projects",
        json={"title": "Seed Expand Project"},
        headers={"Authorization": f"Bearer {token}"},
    )
    return res.json()["id"]


@pytest.mark.asyncio
async def test_seed_expand_returns_202(client):
    """Happy path: POST seed-expand returns 202 with a task_id."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    with patch(
        "app.api.seed_papers.seed_papers_service.run_seed_expansion",
        new_callable=AsyncMock,
    ):
        response = await client.post(
            f"/api/v1/projects/{project_id}/seed-expand",
            json={"dois": ["10.1000/test-doi"]},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    uuid.UUID(data["task_id"])


@pytest.mark.asyncio
async def test_seed_expand_concurrency_guard_returns_409(client):
    """Second seed expand while one is pending should return 409."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    with patch(
        "app.api.seed_papers.seed_papers_service.run_seed_expansion",
        new_callable=AsyncMock,
    ):
        # First request should succeed
        resp1 = await client.post(
            f"/api/v1/projects/{project_id}/seed-expand",
            json={"dois": ["10.1000/first"]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp1.status_code == 202

        # Second request should be rejected
        resp2 = await client.post(
            f"/api/v1/projects/{project_id}/seed-expand",
            json={"dois": ["10.1000/second"]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp2.status_code == 409
        assert "already in progress" in resp2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_seed_expand_empty_request_returns_422(client):
    """An empty request (no dois, no titles) should fail validation with 422."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    response = await client.post(
        f"/api/v1/projects/{project_id}/seed-expand",
        json={"dois": [], "titles": []},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422

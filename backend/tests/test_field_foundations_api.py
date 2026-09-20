"""Tests for field foundations API endpoint."""

import os
import uuid

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch

import pytest


def test_field_foundations_route_exists():
    """Verify the POST field-foundations endpoint is registered."""
    from app.api.field_foundations import router

    routes = [r.path for r in router.routes]
    assert "/projects/{project_id}/field-foundations" in routes


async def get_auth_token(client):
    email = f"fftest-{uuid.uuid4().hex[:8]}@example.com"
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "FF Test User",
            "expertise_level": "researcher",
        },
    )
    return res.json()["access_token"]


async def create_project(client, token):
    res = await client.post(
        "/api/v1/projects",
        json={"title": "Field Foundations Test Project"},
        headers={"Authorization": f"Bearer {token}"},
    )
    return res.json()["id"]


@pytest.mark.asyncio
async def test_field_foundations_returns_202(client):
    """Happy path: POST field-foundations returns 202 with a task_id."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    with patch(
        "app.api.field_foundations.ff_service.run_field_foundations",
        new_callable=AsyncMock,
    ):
        response = await client.post(
            f"/api/v1/projects/{project_id}/field-foundations",
            json={"topic": "Educational Technology"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    uuid.UUID(data["task_id"])


@pytest.mark.asyncio
async def test_field_foundations_concurrency_guard_returns_409(client):
    """Second field foundations request while one is pending should return 409."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    with patch(
        "app.api.field_foundations.ff_service.run_field_foundations",
        new_callable=AsyncMock,
    ):
        # First request should succeed
        resp1 = await client.post(
            f"/api/v1/projects/{project_id}/field-foundations",
            json={"topic": "EdTech"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp1.status_code == 202

        # Second request should be rejected
        resp2 = await client.post(
            f"/api/v1/projects/{project_id}/field-foundations",
            json={"topic": "EdTech Again"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp2.status_code == 409
        assert "already in progress" in resp2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_field_foundations_unauthenticated_returns_401(client):
    """Request without auth token should return 401."""
    fake_project_id = str(uuid.uuid4())
    response = await client.post(
        f"/api/v1/projects/{fake_project_id}/field-foundations",
        json={"topic": "EdTech"},
    )
    assert response.status_code == 401

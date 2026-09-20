"""Pagination contract tests for the previously unbounded list endpoints."""

import uuid

import pytest
from sqlalchemy import update

from app.models.chat_message import ChatMessage
from app.models.user import User, UserRole


async def _register(client, prefix="page"):
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"{prefix}-{uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Page User",
            "expertise_level": "researcher",
        },
    )
    data = res.json()
    return {"Authorization": f"Bearer {data['access_token']}"}, data["user"]["id"]


@pytest.mark.asyncio
async def test_list_projects_is_paginated(client):
    headers, _ = await _register(client, "proj")
    for i in range(3):
        await client.post(
            "/api/v1/projects", json={"title": f"P{i}"}, headers=headers
        )

    first = await client.get("/api/v1/projects?page=1&limit=2", headers=headers)
    assert first.status_code == 200
    body = first.json()
    assert len(body["projects"]) == 2
    assert body["total"] == 3
    assert body["page"] == 1
    assert body["limit"] == 2

    second = await client.get("/api/v1/projects?page=2&limit=2", headers=headers)
    assert len(second.json()["projects"]) == 1

    assert (await client.get("/api/v1/projects?limit=1000", headers=headers)).status_code == 422


@pytest.mark.asyncio
async def test_list_drafts_is_paginated(client):
    headers, _ = await _register(client, "draftpage")
    proj = await client.post(
        "/api/v1/projects", json={"title": "Draft Page"}, headers=headers
    )
    project_id = proj.json()["id"]
    for i in range(3):
        await client.post(
            f"/api/v1/projects/{project_id}/drafts",
            json={"title": f"D{i}"},
            headers=headers,
        )

    res = await client.get(
        f"/api/v1/projects/{project_id}/drafts?page=1&limit=2", headers=headers
    )
    assert res.status_code == 200
    body = res.json()
    assert len(body["drafts"]) == 2
    assert body["total"] == 3
    assert body["page"] == 1
    assert body["limit"] == 2

    too_big = await client.get(
        f"/api/v1/projects/{project_id}/drafts?limit=1000", headers=headers
    )
    assert too_big.status_code == 422


@pytest.mark.asyncio
async def test_list_datasets_rejects_absurd_limit(client):
    headers, _ = await _register(client, "dspage")
    proj = await client.post(
        "/api/v1/projects", json={"title": "DS Page"}, headers=headers
    )
    project_id = proj.json()["id"]

    ok = await client.get(
        f"/api/v1/projects/{project_id}/datasets?limit=200", headers=headers
    )
    assert ok.status_code == 200
    assert ok.json() == []

    too_big = await client.get(
        f"/api/v1/projects/{project_id}/datasets?limit=1000", headers=headers
    )
    assert too_big.status_code == 422


@pytest.mark.asyncio
async def test_list_paper_analyses_rejects_absurd_limit(client):
    headers, _ = await _register(client, "pa")
    proj = await client.post(
        "/api/v1/projects", json={"title": "PA Page"}, headers=headers
    )
    project_id = proj.json()["id"]

    ok = await client.get(
        f"/api/v1/projects/{project_id}/paper-analyses?page=1&limit=10", headers=headers
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["analyses"] == []
    assert body["total"] == 0
    assert body["page"] == 1
    assert body["limit"] == 10

    too_big = await client.get(
        f"/api/v1/projects/{project_id}/paper-analyses?limit=1000", headers=headers
    )
    assert too_big.status_code == 422

    no_params = await client.get(
        f"/api/v1/projects/{project_id}/paper-analyses", headers=headers
    )
    assert no_params.status_code == 200
    assert no_params.json()["limit"] == 200


@pytest.mark.asyncio
async def test_admin_list_users_rejects_absurd_limit(client, db_session):
    headers, user_id = await _register(client, "adminpage")
    await db_session.execute(
        update(User).where(User.id == uuid.UUID(user_id)).values(role=UserRole.admin)
    )
    await db_session.flush()

    ok = await client.get("/api/v1/admin/users?page=1&limit=10", headers=headers)
    assert ok.status_code == 200
    assert isinstance(ok.json(), list)

    too_big = await client.get("/api/v1/admin/users?limit=1000", headers=headers)
    assert too_big.status_code == 422


@pytest.mark.asyncio
async def test_chat_history_returns_newest_page_in_ascending_order(client, db_session):
    headers, _ = await _register(client, "chatpage")
    proj = await client.post(
        "/api/v1/projects", json={"title": "Chat Page"}, headers=headers
    )
    project_id = uuid.UUID(proj.json()["id"])

    for i in range(5):
        db_session.add(
            ChatMessage(project_id=project_id, role="user", content=f"m{i}")
        )
        await db_session.flush()

    res = await client.get(
        f"/api/v1/projects/{project_id}/chat/history?limit=2", headers=headers
    )
    assert res.status_code == 200
    contents = [m["content"] for m in res.json()]
    assert contents == ["m3", "m4"]

    too_big = await client.get(
        f"/api/v1/projects/{project_id}/chat/history?limit=1000", headers=headers
    )
    assert too_big.status_code == 422

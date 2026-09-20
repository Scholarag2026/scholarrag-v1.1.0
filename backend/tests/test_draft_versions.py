"""Draft version hot-path and retention tests."""

import uuid

import pytest
from sqlalchemy import event, func, select

from app.models.draft import DraftVersion


async def _setup(client):
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"ver-{uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Version User",
            "expertise_level": "researcher",
        },
    )
    headers = {"Authorization": f"Bearer {res.json()['access_token']}"}
    proj = await client.post(
        "/api/v1/projects", json={"title": "Version Project"}, headers=headers
    )
    project_id = proj.json()["id"]
    draft = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Versioned"},
        headers=headers,
    )
    return headers, draft.json()["id"]


def _doc(text):
    return {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": text}]}
        ],
    }


@pytest.mark.asyncio
async def test_draft_read_does_not_load_version_content(client, db_session):
    headers, draft_id = await _setup(client)
    await client.put(
        f"/api/v1/drafts/{draft_id}", json={"content": _doc("v1")}, headers=headers
    )

    statements: list[str] = []
    bind = db_session.sync_session.get_bind()

    def _capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(bind, "before_cursor_execute", _capture)
    try:
        res = await client.get(f"/api/v1/drafts/{draft_id}", headers=headers)
    finally:
        event.remove(bind, "before_cursor_execute", _capture)

    assert res.status_code == 200
    assert len(res.json()["versions"]) == 1
    offenders = [s for s in statements if "draft_versions.content" in s]
    assert offenders == [], offenders


@pytest.mark.asyncio
async def test_draft_history_is_capped_at_50(client, db_session):
    headers, draft_id = await _setup(client)

    for i in range(55):
        res = await client.put(
            f"/api/v1/drafts/{draft_id}",
            json={"content": _doc(f"revision {i}")},
            headers=headers,
        )
        assert res.status_code == 200

    count = (
        await db_session.execute(
            select(func.count())
            .select_from(DraftVersion)
            .where(DraftVersion.draft_id == uuid.UUID(draft_id))
        )
    ).scalar()
    assert count == 50

    detail = await client.get(f"/api/v1/drafts/{draft_id}", headers=headers)
    versions = detail.json()["versions"]
    assert len(versions) == 50
    assert versions[0]["version"] == 55
    assert versions[-1]["version"] == 6


@pytest.mark.asyncio
async def test_delete_draft_still_cascades_versions(client, db_session):
    headers, draft_id = await _setup(client)
    await client.put(
        f"/api/v1/drafts/{draft_id}", json={"content": _doc("x")}, headers=headers
    )

    res = await client.delete(f"/api/v1/drafts/{draft_id}", headers=headers)
    assert res.status_code == 200

    remaining = (
        await db_session.execute(
            select(func.count())
            .select_from(DraftVersion)
            .where(DraftVersion.draft_id == uuid.UUID(draft_id))
        )
    ).scalar()
    assert remaining == 0

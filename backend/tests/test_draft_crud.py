import uuid

import pytest


async def get_auth_token(client):
    email = f"draft-{uuid.uuid4().hex[:8]}@example.com"
    res = await client.post("/api/v1/auth/register", json={
        "email": email, "password": "StrongPass123!",
        "name": "Draft User", "expertise_level": "researcher",
    })
    return res.json()["access_token"]


async def create_project(client, token):
    res = await client.post("/api/v1/projects", json={"title": "Draft Test Project"},
                            headers={"Authorization": f"Bearer {token}"})
    return res.json()["id"]


@pytest.mark.asyncio
async def test_create_draft(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    response = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "My Literature Review", "paper_type": "literature_review"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "My Literature Review"
    assert data["paper_type"] == "literature_review"
    assert data["status"] == "draft"
    assert data["current_version"] == 1
    assert data["content"] is None


@pytest.mark.asyncio
async def test_list_drafts(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Draft A"},
        headers={"Authorization": f"Bearer {token}"},
    )
    await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Draft B"},
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.get(
        f"/api/v1/projects/{project_id}/drafts",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["drafts"]) == 2


@pytest.mark.asyncio
async def test_get_draft_with_versions(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Versioned Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]

    response = await client.get(
        f"/api/v1/drafts/{draft_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["draft"]["title"] == "Versioned Draft"
    assert isinstance(data["versions"], list)


@pytest.mark.asyncio
async def test_update_draft_creates_version(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Editable Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]

    new_content = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [
                {"type": "text", "text": "Hello world"},
            ]},
        ],
    }
    response = await client.put(
        f"/api/v1/drafts/{draft_id}",
        json={"content": new_content},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == new_content
    assert data["current_version"] == 2

    # Verify version was created
    detail_res = await client.get(
        f"/api/v1/drafts/{draft_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    versions = detail_res.json()["versions"]
    assert len(versions) >= 1


@pytest.mark.asyncio
async def test_delete_draft(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Deletable"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]

    response = await client.delete(
        f"/api/v1/drafts/{draft_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200

    # Verify gone
    get_res = await client.get(
        f"/api/v1/drafts/{draft_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_res.status_code == 404


@pytest.mark.asyncio
async def test_drafts_unauthorized(client):
    response = await client.get("/api/v1/projects/00000000-0000-0000-0000-000000000000/drafts")
    assert response.status_code == 401 or response.status_code == 403

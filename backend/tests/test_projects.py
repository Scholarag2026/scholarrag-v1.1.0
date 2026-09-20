import pytest


async def get_auth_token(client):
    """Helper to register a user and get their auth token."""
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "proj@example.com",
            "password": "StrongPass123!",
            "name": "Proj User",
            "expertise_level": "researcher",
        },
    )
    return res.json()["access_token"]


@pytest.mark.asyncio
async def test_create_project(client):
    token = await get_auth_token(client)
    response = await client.post(
        "/api/v1/projects",
        json={
            "title": "My Research",
            "description": "A test project",
            "citation_style": "APA",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "My Research"
    assert data["citation_style"] == "APA"
    assert data["status"] == "active"
    assert "id" in data


@pytest.mark.asyncio
async def test_list_projects(client):
    token = await get_auth_token(client)
    await client.post(
        "/api/v1/projects",
        json={"title": "P1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    await client.post(
        "/api/v1/projects",
        json={"title": "P2"},
        headers={"Authorization": f"Bearer {token}"},
    )
    response = await client.get(
        "/api/v1/projects",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert len(response.json()["projects"]) == 2


@pytest.mark.asyncio
async def test_get_project(client):
    token = await get_auth_token(client)
    create_res = await client.post(
        "/api/v1/projects",
        json={"title": "Get Me"},
        headers={"Authorization": f"Bearer {token}"},
    )
    pid = create_res.json()["id"]
    response = await client.get(
        f"/api/v1/projects/{pid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Get Me"


@pytest.mark.asyncio
async def test_update_project(client):
    token = await get_auth_token(client)
    create_res = await client.post(
        "/api/v1/projects",
        json={"title": "Old"},
        headers={"Authorization": f"Bearer {token}"},
    )
    pid = create_res.json()["id"]
    response = await client.put(
        f"/api/v1/projects/{pid}",
        json={"title": "New"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "New"


@pytest.mark.asyncio
async def test_delete_project(client):
    token = await get_auth_token(client)
    create_res = await client.post(
        "/api/v1/projects",
        json={"title": "Delete Me"},
        headers={"Authorization": f"Bearer {token}"},
    )
    pid = create_res.json()["id"]
    response = await client.delete(
        f"/api/v1/projects/{pid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


@pytest.mark.asyncio
async def test_project_not_found(client):
    token = await get_auth_token(client)
    response = await client.get(
        "/api/v1/projects/00000000-0000-0000-0000-000000000000",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_projects_unauthorized(client):
    response = await client.get("/api/v1/projects")
    assert response.status_code == 401 or response.status_code == 403

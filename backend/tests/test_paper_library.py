import pytest


async def get_auth_token(client):
    res = await client.post("/api/v1/auth/register", json={
        "email": "paper@example.com", "password": "StrongPass123!",
        "name": "Paper User", "expertise_level": "researcher",
    })
    return res.json()["access_token"]


async def create_project(client, token):
    res = await client.post("/api/v1/projects", json={"title": "Test Project"},
                            headers={"Authorization": f"Bearer {token}"})
    return res.json()["id"]


SAMPLE_PAPER_DATA = {
    "doi": "10.1038/s41586-019-1724-z",
    "title": "Deep learning for computational biology",
    "authors": [{"name": "John Smith"}, {"name": "Jane Doe"}],
    "year": 2019,
    "journal_name": "Nature",
    "citation_count": 150,
    "abstract": "Deep learning transforms biology.",
    "source_api": "openalex",
    "external_id": "W2741809807",
}


@pytest.mark.asyncio
async def test_add_paper_to_project(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    response = await client.post(
        f"/api/v1/projects/{project_id}/papers",
        json={"paper_data": SAMPLE_PAPER_DATA},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["paper"]["title"] == "Deep learning for computational biology"
    assert data["paper"]["doi"] == "10.1038/s41586-019-1724-z"
    assert data["project_id"] == project_id


@pytest.mark.asyncio
async def test_add_duplicate_paper_fails(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    await client.post(
        f"/api/v1/projects/{project_id}/papers",
        json={"paper_data": SAMPLE_PAPER_DATA},
        headers={"Authorization": f"Bearer {token}"},
    )
    response = await client.post(
        f"/api/v1/projects/{project_id}/papers",
        json={"paper_data": SAMPLE_PAPER_DATA},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_list_project_papers(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    await client.post(
        f"/api/v1/projects/{project_id}/papers",
        json={"paper_data": SAMPLE_PAPER_DATA},
        headers={"Authorization": f"Bearer {token}"},
    )

    paper2 = {
        **SAMPLE_PAPER_DATA,
        "doi": "10.1/other",
        "title": "Other Paper",
        "external_id": "W9999999999",
    }
    await client.post(
        f"/api/v1/projects/{project_id}/papers",
        json={"paper_data": paper2},
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.get(
        f"/api/v1/projects/{project_id}/papers",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["papers"]) == 2


@pytest.mark.asyncio
async def test_remove_paper_from_project(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    add_res = await client.post(
        f"/api/v1/projects/{project_id}/papers",
        json={"paper_data": SAMPLE_PAPER_DATA},
        headers={"Authorization": f"Bearer {token}"},
    )
    paper_id = add_res.json()["paper"]["id"]

    response = await client.delete(
        f"/api/v1/projects/{project_id}/papers/{paper_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200

    # Verify it's gone
    list_res = await client.get(
        f"/api/v1/projects/{project_id}/papers",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_res.json()["total"] == 0


@pytest.mark.asyncio
async def test_papers_unauthorized(client):
    response = await client.get("/api/v1/projects/00000000-0000-0000-0000-000000000000/papers")
    assert response.status_code == 401 or response.status_code == 403


@pytest.mark.asyncio
async def test_list_papers_rejects_absurd_limit(client):
    import uuid as _uuid

    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"cap-{_uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Cap User",
            "expertise_level": "researcher",
        },
    )
    headers = {"Authorization": f"Bearer {res.json()['access_token']}"}
    proj = await client.post(
        "/api/v1/projects", json={"title": "Cap Project"}, headers=headers
    )
    project_id = proj.json()["id"]

    too_big = await client.get(
        f"/api/v1/projects/{project_id}/papers?limit=100000", headers=headers
    )
    assert too_big.status_code == 422

    zero_page = await client.get(
        f"/api/v1/projects/{project_id}/papers?page=0", headers=headers
    )
    assert zero_page.status_code == 422

    huge_page = await client.get(
        f"/api/v1/projects/{project_id}/papers?page=9999999999999999999999", headers=headers
    )
    assert huge_page.status_code == 422

    ok = await client.get(
        f"/api/v1/projects/{project_id}/papers?limit=500", headers=headers
    )
    assert ok.status_code == 200
    assert ok.json()["limit"] == 500

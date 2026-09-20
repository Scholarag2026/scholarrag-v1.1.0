import uuid

import pytest


async def get_auth_token(client):
    email = f"verify-{uuid.uuid4().hex[:8]}@example.com"
    res = await client.post("/api/v1/auth/register", json={
        "email": email, "password": "StrongPass123!",
        "name": "Verify User", "expertise_level": "researcher",
    })
    return res.json()["access_token"]


async def create_project(client, token):
    res = await client.post("/api/v1/projects", json={"title": "Verify Project"},
                            headers={"Authorization": f"Bearer {token}"})
    return res.json()["id"]


SAMPLE_PAPER = {
    "doi": "10.1038/s41586-019-1724-z",
    "title": "Deep learning for computational biology",
    "authors": [{"name": "John Smith"}],
    "year": 2019,
    "journal_name": "Nature",
    "citation_count": 150,
    "abstract": "Deep learning transforms biology.",
    "source_api": "openalex",
    "external_id": "W2741809807",
}

PAPER_NO_DOI = {
    "title": "A Paper Without DOI",
    "authors": [{"name": "Jane Doe"}],
    "year": 2022,
    "source_api": "manual",
}


@pytest.mark.asyncio
async def test_verify_references_returns_task_id(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    # Add a paper
    add_res = await client.post(
        f"/api/v1/projects/{project_id}/papers",
        json={"paper_data": SAMPLE_PAPER},
        headers={"Authorization": f"Bearer {token}"},
    )
    paper_id = add_res.json()["paper"]["id"]

    response = await client.post(
        f"/api/v1/projects/{project_id}/verify-references",
        json={"paper_ids": [paper_id]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data


@pytest.mark.asyncio
async def test_verify_all_papers_when_no_ids_given(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    # Add two papers
    await client.post(
        f"/api/v1/projects/{project_id}/papers",
        json={"paper_data": SAMPLE_PAPER},
        headers={"Authorization": f"Bearer {token}"},
    )
    paper2 = {**SAMPLE_PAPER, "doi": "10.1/other", "title": "Other Paper"}
    await client.post(
        f"/api/v1/projects/{project_id}/papers",
        json={"paper_data": paper2},
        headers={"Authorization": f"Bearer {token}"},
    )

    # Verify all (empty paper_ids)
    response = await client.post(
        f"/api/v1/projects/{project_id}/verify-references",
        json={"paper_ids": []},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert "task_id" in response.json()


@pytest.mark.asyncio
async def test_verify_references_unauthorized(client):
    response = await client.post(
        "/api/v1/projects/00000000-0000-0000-0000-000000000000/verify-references",
        json={"paper_ids": []},
    )
    assert response.status_code == 401 or response.status_code == 403

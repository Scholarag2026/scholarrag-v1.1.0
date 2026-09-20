import uuid

import pytest


async def get_auth_token(client):
    email = f"writer-{uuid.uuid4().hex[:8]}@example.com"
    res = await client.post("/api/v1/auth/register", json={
        "email": email, "password": "StrongPass123!",
        "name": "Writer User", "expertise_level": "researcher",
    })
    return res.json()["access_token"]


async def create_project(client, token):
    res = await client.post("/api/v1/projects", json={"title": "Writing Project"},
                            headers={"Authorization": f"Bearer {token}"})
    return res.json()["id"]


async def create_draft(client, token, project_id):
    res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "AI Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    return res.json()["id"]


SAMPLE_PAPER = {
    "doi": "10.1038/test-paper",
    "title": "Test Paper on Education",
    "authors": [{"name": "Alice Smith"}],
    "year": 2023,
    "journal_name": "Education Research",
    "abstract": "A study on learning outcomes.",
    "source_api": "openalex",
    "external_id": "W123",
}


@pytest.mark.asyncio
async def test_generate_returns_task_id(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)
    draft_id = await create_draft(client, token, project_id)

    response = await client.post(
        f"/api/v1/drafts/{draft_id}/generate",
        json={"section_type": "literature_review"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data


@pytest.mark.asyncio
async def test_generate_with_papers_in_library(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)
    draft_id = await create_draft(client, token, project_id)

    await client.post(
        f"/api/v1/projects/{project_id}/papers",
        json={"paper_data": SAMPLE_PAPER},
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.post(
        f"/api/v1/drafts/{draft_id}/generate",
        json={"section_type": "literature_review", "context": "Focus on learning outcomes"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert "task_id" in response.json()


@pytest.mark.asyncio
async def test_generate_accepts_a_section_title(client):
    """The endpoint accepts the optional ``section_title``
    field and does not reject the request for carrying it."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)
    draft_id = await create_draft(client, token, project_id)

    response = await client.post(
        f"/api/v1/drafts/{draft_id}/generate",
        json={"section_type": "literature_review", "section_title": "Peer feedback"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert "task_id" in response.json()


@pytest.mark.asyncio
async def test_generate_unauthorized(client):
    response = await client.post(
        "/api/v1/drafts/00000000-0000-0000-0000-000000000000/generate",
        json={"section_type": "literature_review"},
    )
    assert response.status_code == 401 or response.status_code == 403

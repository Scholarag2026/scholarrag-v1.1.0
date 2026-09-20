# backend/tests/test_bibtex.py
import pytest


async def get_auth_token(client):
    res = await client.post("/api/v1/auth/register", json={
        "email": "bib@example.com", "password": "StrongPass123!",
        "name": "Bib User", "expertise_level": "researcher",
    })
    return res.json()["access_token"]


async def create_project_with_paper(client, token):
    proj = await client.post("/api/v1/projects", json={"title": "BibTeX Project"},
                             headers={"Authorization": f"Bearer {token}"})
    project_id = proj.json()["id"]

    await client.post(
        f"/api/v1/projects/{project_id}/papers",
        json={"paper_data": {
            "doi": "10.1038/s41586-019-1724-z",
            "title": "Deep learning for computational biology",
            "authors": [{"name": "John Smith"}, {"name": "Jane Doe"}],
            "year": 2019,
            "journal_name": "Nature",
            "citation_count": 150,
            "source_api": "openalex",
        }},
        headers={"Authorization": f"Bearer {token}"},
    )
    return project_id


@pytest.mark.asyncio
async def test_export_bibtex(client):
    token = await get_auth_token(client)
    project_id = await create_project_with_paper(client, token)

    response = await client.get(
        f"/api/v1/projects/{project_id}/papers/export?format=bibtex",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-bibtex"

    content = response.text
    assert "@article{" in content
    assert "Deep learning for computational biology" in content
    assert "John Smith" in content
    assert "2019" in content
    assert "Nature" in content


@pytest.mark.asyncio
async def test_export_empty_library(client):
    token = await get_auth_token(client)
    proj = await client.post("/api/v1/projects", json={"title": "Empty"},
                             headers={"Authorization": f"Bearer {token}"})
    project_id = proj.json()["id"]

    response = await client.get(
        f"/api/v1/projects/{project_id}/papers/export?format=bibtex",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_import_bibtex(client):
    token = await get_auth_token(client)
    proj = await client.post("/api/v1/projects", json={"title": "Import Project"},
                             headers={"Authorization": f"Bearer {token}"})
    project_id = proj.json()["id"]

    bibtex_content = """@article{smith2020,
  title = {A Great Paper},
  author = {Smith, John and Doe, Jane},
  year = {2020},
  journal = {Nature},
  doi = {10.1234/test.2020},
}

@article{jones2021,
  title = {Another Paper},
  author = {Jones, Bob},
  year = {2021},
  journal = {Science},
}
"""
    response = await client.post(
        f"/api/v1/projects/{project_id}/papers/import",
        files={"file": ("refs.bib", bibtex_content, "application/x-bibtex")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["imported_count"] == 2


@pytest.mark.asyncio
async def test_import_empty_bibtex(client):
    token = await get_auth_token(client)
    proj = await client.post("/api/v1/projects", json={"title": "Empty Import"},
                             headers={"Authorization": f"Bearer {token}"})
    project_id = proj.json()["id"]

    response = await client.post(
        f"/api/v1/projects/{project_id}/papers/import",
        files={"file": ("empty.bib", "", "application/x-bibtex")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["imported_count"] == 0


@pytest.mark.asyncio
async def test_export_includes_more_than_one_page(client, db_session):
    """Export must not stop at the list helper's default page size of 50."""
    import uuid as _uuid

    from app.models.paper import Paper, SourceApi
    from app.models.project_paper import ProjectPaper

    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"bigbib-{_uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Big Bib",
            "expertise_level": "researcher",
        },
    )
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    proj = await client.post(
        "/api/v1/projects", json={"title": "Big Library"}, headers=headers
    )
    project_id = _uuid.UUID(proj.json()["id"])

    total = 55
    for i in range(total):
        paper = Paper(
            doi=f"10.9999/bulk.{i}",
            title=f"Bulk Paper {i}",
            authors=[{"name": f"Author {i}"}],
            year=2000 + (i % 20),
            journal_name="Bulk Journal",
            source_api=SourceApi.manual,
        )
        db_session.add(paper)
        await db_session.flush()
        db_session.add(ProjectPaper(project_id=project_id, paper_id=paper.id))
    await db_session.flush()

    response = await client.get(
        f"/api/v1/projects/{project_id}/papers/export?format=bibtex",
        headers=headers,
    )
    assert response.status_code == 200
    assert response.text.count("@article{") == total

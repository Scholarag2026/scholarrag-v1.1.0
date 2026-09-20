import io
import uuid

import pytest

from app.models.paper import Paper, SourceApi
from app.models.project_paper import ProjectPaper


async def get_auth_token(client):
    email = f"export-{uuid.uuid4().hex[:8]}@example.com"
    res = await client.post("/api/v1/auth/register", json={
        "email": email, "password": "StrongPass123!",
        "name": "Export User", "expertise_level": "researcher",
    })
    return res.json()["access_token"]


async def create_project(client, token, citation_style=None):
    payload = {"title": "Export Project"}
    if citation_style:
        payload["citation_style"] = citation_style
    res = await client.post("/api/v1/projects", json=payload,
                            headers={"Authorization": f"Bearer {token}"})
    return res.json()["id"]


CITED_CONTENT = {
    "type": "doc",
    "content": [
        {
            "type": "paragraph",
            "attrs": {
                "citationLinks": [
                    {
                        "sentence": "This finding was shown before (Smith, 2020).",
                        "citation_text": "(Smith, 2020)",
                        "keys": ["smith_2020"],
                    }
                ]
            },
            "content": [
                {"type": "text", "text": "This finding was shown before (Smith, 2020)."}
            ],
        }
    ],
}


async def add_cited_paper(db_session, project_id):
    paper = Paper(
        doi="10.1234/export.1",
        title="A Cited Paper",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        journal_name="Journal of Exports",
        source_api=SourceApi.manual,
    )
    db_session.add(paper)
    await db_session.flush()
    db_session.add(ProjectPaper(project_id=uuid.UUID(str(project_id)), paper_id=paper.id))
    await db_session.flush()


@pytest.mark.asyncio
async def test_export_empty_draft_docx(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Empty Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]

    response = await client.get(
        f"/api/v1/drafts/{draft_id}/export?format=docx",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert response.content[:2] == b"PK"


@pytest.mark.asyncio
async def test_export_draft_with_content_docx(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Content Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]

    tiptap_content = {
        "type": "doc",
        "content": [
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Literature Review"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "This is a test paragraph with some content."}]},
            {"type": "heading", "attrs": {"level": 2}, "content": [{"type": "text", "text": "Subsection"}]},
            {"type": "paragraph", "content": [
                {"type": "text", "text": "Some "},
                {"type": "text", "marks": [{"type": "bold"}], "text": "bold"},
                {"type": "text", "text": " and "},
                {"type": "text", "marks": [{"type": "italic"}], "text": "italic"},
                {"type": "text", "text": " text."},
            ]},
        ],
    }
    await client.put(
        f"/api/v1/drafts/{draft_id}",
        json={"content": tiptap_content},
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.get(
        f"/api/v1/drafts/{draft_id}/export?format=docx",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.content[:2] == b"PK"
    assert len(response.content) > 1000


@pytest.mark.asyncio
async def test_export_unsupported_format(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Export Test"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]

    response = await client.get(
        f"/api/v1/drafts/{draft_id}/export?format=html",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_export_ieee_project_renders_numbered_citation_docx(client, db_session):
    """Export applies the
    project's chosen citation style, computed once from the citation-link map, never
    written back to the stored draft."""
    import docx

    token = await get_auth_token(client)
    project_id = await create_project(client, token, citation_style="IEEE")
    await add_cited_paper(db_session, project_id)

    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "IEEE Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]
    await client.put(
        f"/api/v1/drafts/{draft_id}",
        json={"content": CITED_CONTENT},
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.get(
        f"/api/v1/drafts/{draft_id}/export?format=docx",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    document = docx.Document(io.BytesIO(response.content))
    texts = [p.text for p in document.paragraphs]
    assert any("[1]" in t for t in texts)
    assert "References" in texts

    # the stored draft itself is never rewritten
    detail = await client.get(
        f"/api/v1/drafts/{draft_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert detail.status_code == 200
    stored_content = detail.json()["draft"]["content"]
    assert stored_content["content"][0]["content"][0]["text"] == (
        "This finding was shown before (Smith, 2020)."
    )


@pytest.mark.asyncio
async def test_export_ieee_project_renders_numbered_citation_latex(client, db_session):
    token = await get_auth_token(client)
    project_id = await create_project(client, token, citation_style="IEEE")
    await add_cited_paper(db_session, project_id)

    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "IEEE Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]
    await client.put(
        f"/api/v1/drafts/{draft_id}",
        json={"content": CITED_CONTENT},
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.get(
        f"/api/v1/drafts/{draft_id}/export?format=latex",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert "[1]" in body
    assert "References" in body


@pytest.mark.asyncio
async def test_export_ieee_project_renders_valid_pdf(client, db_session):
    """PDF text is glyph-encoded by weasyprint and cannot cheaply be asserted on here
    (`test_export_service.py` and `test_citation_render.py` already cover, respectively,
    that `export_pdf` renders whatever HTML it is given and that `render_document`
    numbers this exact citation); this is the endpoint-level smoke test that the two are
    wired together without error for an IEEE project."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token, citation_style="IEEE")
    await add_cited_paper(db_session, project_id)

    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "IEEE Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]
    await client.put(
        f"/api/v1/drafts/{draft_id}",
        json={"content": CITED_CONTENT},
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.get(
        f"/api/v1/drafts/{draft_id}/export?format=pdf",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.content[:5] == b"%PDF-"


@pytest.mark.asyncio
async def test_export_ignores_paper_metadata_blob(client, db_session):
    """Export selects only the columns
    `render_document` reads, not the whole ORM row (which would pull `metadata`'s
    `fulltext_chunks` and `deep_analysis`). A paper carrying a large metadata blob must
    still export correctly and quickly, proving the narrower select still gives the
    renderer everything it needs."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token, citation_style="IEEE")

    paper = Paper(
        doi="10.1234/export.2",
        title="A Cited Paper With A Large Blob",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        journal_name="Journal of Exports",
        source_api=SourceApi.manual,
        metadata_={
            "fulltext_chunks": [{"section": "body", "text": "x" * 5000} for _ in range(20)],
            "deep_analysis": {"key_findings": "y" * 5000},
        },
    )
    db_session.add(paper)
    await db_session.flush()
    db_session.add(ProjectPaper(project_id=uuid.UUID(str(project_id)), paper_id=paper.id))
    await db_session.flush()

    content = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {
                            "sentence": "This finding was shown before (Smith, 2020).",
                            "citation_text": "(Smith, 2020)",
                            "keys": ["smith_2020"],
                        }
                    ]
                },
                "content": [
                    {"type": "text", "text": "This finding was shown before (Smith, 2020)."}
                ],
            }
        ],
    }
    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Blob Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]
    await client.put(
        f"/api/v1/drafts/{draft_id}",
        json={"content": content},
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.get(
        f"/api/v1/drafts/{draft_id}/export?format=latex",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert "[1]" in body
    assert "References" in body


@pytest.mark.asyncio
async def test_export_apa_project_text_unchanged(client, db_session):
    """Author-year styles (the APA default) never touch the in-text citation."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)  # default style is APA
    await add_cited_paper(db_session, project_id)

    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "APA Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]
    await client.put(
        f"/api/v1/drafts/{draft_id}",
        json={"content": CITED_CONTENT},
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.get(
        f"/api/v1/drafts/{draft_id}/export?format=latex",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert "(Smith, 2020)" in body
    assert "[1]" not in body
    assert "References" in body

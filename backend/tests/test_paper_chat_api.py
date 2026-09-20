"""Tests that paper chat narrows the library before building context."""

import os
import uuid

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.paper import Paper, SourceApi
from app.models.project_paper import ProjectPaper


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _fake_httpx(payload):
    fake_client = AsyncMock()
    fake_client.post.return_value = _FakeResponse(payload)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=fake_client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    module = MagicMock()
    module.AsyncClient.return_value = ctx
    return module


@pytest.mark.asyncio
async def test_chat_bounds_candidate_window_and_context(client, db_session):
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"chat-{uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Chat User",
            "expertise_level": "researcher",
        },
    )
    headers = {"Authorization": f"Bearer {res.json()['access_token']}"}
    proj = await client.post(
        "/api/v1/projects", json={"title": "Chat Library"}, headers=headers
    )
    project_id = uuid.UUID(proj.json()["id"])

    for i in range(250):
        paper = Paper(
            doi=f"10.5555/chat.{i}",
            title=f"Chat Paper {i}",
            authors=[{"name": f"Author {i}"}],
            year=2010 + (i % 10),
            abstract="An abstract.",
            source_api=SourceApi.manual,
            metadata_={"deep_analysis": {"themes": [f"theme{i}"]}},
        )
        db_session.add(paper)
        await db_session.flush()
        db_session.add(ProjectPaper(project_id=project_id, paper_id=paper.id))
    await db_session.flush()

    payload = {"choices": [{"message": {"content": "An answer (Author 1, 2011)."}}]}

    with patch(
        "app.api.paper_chat.select_papers_for_section", new_callable=AsyncMock
    ) as selector, patch("app.api.paper_chat.httpx", _fake_httpx(payload)):
        selector.return_value = []
        response = await client.post(
            f"/api/v1/projects/{project_id}/chat",
            json={"question": "What do these papers say?"},
            headers=headers,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "An answer (Author 1, 2011)."

    selector.assert_awaited_once()
    summaries = selector.await_args.args[1]
    assert len(summaries) == 200, "candidate window must be bounded to CHAT_CANDIDATE_LIMIT"
    assert set(summaries[0]) == {"id", "title", "year", "themes"}

    assert len(body["references"]) == 20, "context must be capped at CHAT_CONTEXT_LIMIT"


@pytest.mark.asyncio
async def test_chat_empty_library_short_circuits(client):
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"chat0-{uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Chat User",
            "expertise_level": "researcher",
        },
    )
    headers = {"Authorization": f"Bearer {res.json()['access_token']}"}
    proj = await client.post(
        "/api/v1/projects", json={"title": "Empty Library"}, headers=headers
    )
    project_id = proj.json()["id"]

    response = await client.post(
        f"/api/v1/projects/{project_id}/chat",
        json={"question": "anything?"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["references"] == []


@pytest.mark.asyncio
async def test_chat_context_and_references_carry_journal_and_doi(client, db_session):
    """The LLM can only cite what it is given: journal + DOI must reach the prompt
    and the structured references payload (user report: APA list missing both)."""
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"chatref-{uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Chat Ref User",
            "expertise_level": "researcher",
        },
    )
    headers = {"Authorization": f"Bearer {res.json()['access_token']}"}
    proj = await client.post(
        "/api/v1/projects",
        json={"title": "Ref Library", "citation_style": "IEEE"},
        headers=headers,
    )
    project_id = uuid.UUID(proj.json()["id"])

    paper = Paper(
        doi="10.1234/ref.1",
        title="A Paper With Full Metadata",
        authors=[{"name": "Ada Lovelace"}],
        year=2021,
        journal_name="Journal of Reproducible Results",
        abstract="An abstract.",
        source_api=SourceApi.manual,
    )
    db_session.add(paper)
    await db_session.flush()
    db_session.add(ProjectPaper(project_id=project_id, paper_id=paper.id))
    await db_session.flush()

    payload = {"choices": [{"message": {"content": "Answer (Lovelace, 2021)."}}]}
    fake = _fake_httpx(payload)

    with patch(
        "app.api.paper_chat.select_papers_for_section", new_callable=AsyncMock
    ) as selector, patch("app.api.paper_chat.httpx", fake):
        selector.return_value = []
        response = await client.post(
            f"/api/v1/projects/{project_id}/chat",
            json={"question": "What does the paper say?"},
            headers=headers,
        )

    assert response.status_code == 200
    body = response.json()
    ref = body["references"][0]
    assert ref["journal_name"] == "Journal of Reproducible Results"
    assert ref["doi"] == "10.1234/ref.1"

    # The project's IEEE style is rendered from the model's own author-year text and a
    # references section is appended from the paper's own metadata -- the model is
    # never told a style.
    assert "[1]" in body["answer"]
    assert "(Lovelace, 2021)" not in body["answer"]
    assert "## References" in body["answer"]
    assert "A. Lovelace" in body["answer"]
    assert "Journal of Reproducible Results" in body["answer"]
    assert body["unresolved_citations"] == []

    fake_client = fake.AsyncClient.return_value.__aenter__.return_value
    messages = fake_client.post.await_args.kwargs["json"]["messages"]
    sent = messages[1]["content"]
    assert "Journal: Journal of Reproducible Results" in sent
    assert "DOI: https://doi.org/10.1234/ref.1" in sent
    # Neither prompt names a style any more: rendering happens after the model answers.
    assert "IEEE" not in sent
    assert "IEEE" not in messages[0]["content"]
    assert "Do not add your own References section" in sent
    assert 'Do not add a "References" section yourself' in messages[0]["content"]


@pytest.mark.asyncio
async def test_chat_apa_project_keeps_text_and_gains_references(client, db_session):
    """An author-year (APA, the default) project's answer is byte-identical apart from
    the appended references section."""
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"chatapa-{uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Chat APA User",
            "expertise_level": "researcher",
        },
    )
    headers = {"Authorization": f"Bearer {res.json()['access_token']}"}
    proj = await client.post(
        "/api/v1/projects", json={"title": "APA Library"}, headers=headers
    )
    project_id = uuid.UUID(proj.json()["id"])

    paper = Paper(
        doi="10.1234/apa.1",
        title="An APA Paper",
        authors=[{"name": "Ada Lovelace"}],
        year=2021,
        journal_name="Journal of Reproducible Results",
        abstract="An abstract.",
        source_api=SourceApi.manual,
    )
    db_session.add(paper)
    await db_session.flush()
    db_session.add(ProjectPaper(project_id=project_id, paper_id=paper.id))
    await db_session.flush()

    original_answer = "Answer (Lovelace, 2021)."
    payload = {"choices": [{"message": {"content": original_answer}}]}

    with patch(
        "app.api.paper_chat.select_papers_for_section", new_callable=AsyncMock
    ) as selector, patch("app.api.paper_chat.httpx", _fake_httpx(payload)):
        selector.return_value = []
        response = await client.post(
            f"/api/v1/projects/{project_id}/chat",
            json={"question": "What does the paper say?"},
            headers=headers,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith(original_answer)
    assert "## References" in body["answer"]
    assert "Lovelace, A. (2021)" in body["answer"]
    assert body["unresolved_citations"] == []


@pytest.mark.asyncio
async def test_chat_citation_naming_no_library_paper_survives_and_is_unresolved(
    client, db_session
):
    """A citation the model wrote that names no paper in the library is left exactly as
    written and reported in `unresolved_citations`."""
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"chatmiss-{uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Chat Missing User",
            "expertise_level": "researcher",
        },
    )
    headers = {"Authorization": f"Bearer {res.json()['access_token']}"}
    proj = await client.post(
        "/api/v1/projects",
        json={"title": "Missing Library", "citation_style": "IEEE"},
        headers=headers,
    )
    project_id = uuid.UUID(proj.json()["id"])

    paper = Paper(
        doi="10.1234/miss.1",
        title="A Paper",
        authors=[{"name": "Ada Lovelace"}],
        year=2021,
        source_api=SourceApi.manual,
    )
    db_session.add(paper)
    await db_session.flush()
    db_session.add(ProjectPaper(project_id=project_id, paper_id=paper.id))
    await db_session.flush()

    original_answer = "An unsupported claim (Nguyen, 2019)."
    payload = {"choices": [{"message": {"content": original_answer}}]}

    with patch(
        "app.api.paper_chat.select_papers_for_section", new_callable=AsyncMock
    ) as selector, patch("app.api.paper_chat.httpx", _fake_httpx(payload)):
        selector.return_value = []
        response = await client.post(
            f"/api/v1/projects/{project_id}/chat",
            json={"question": "What does the paper say?"},
            headers=headers,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == original_answer
    assert body["unresolved_citations"] == ["(Nguyen, 2019)"]
    assert "## References" not in body["answer"]

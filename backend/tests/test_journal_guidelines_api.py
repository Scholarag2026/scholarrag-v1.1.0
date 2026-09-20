"""fetch-journal-guidelines must return 202 + task_id and do the work in the background."""

import os
import uuid

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from app.agents.journal_guidelines_agent import JournalGuidelines  # noqa: E402


async def _register(client):
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"guides-{uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPass123!",
            "name": "Guides User",
            "expertise_level": "researcher",
        },
    )
    return res.json()["access_token"]


async def _create_project(client, token):
    res = await client.post(
        "/api/v1/projects",
        json={"title": "Guides Project", "target_journal": "Computers & Education"},
        headers={"Authorization": f"Bearer {token}"},
    )
    return res.json()["id"]


@pytest.mark.asyncio
async def test_fetch_guidelines_returns_202_and_completes(client):
    token = await _register(client)
    project_id = await _create_project(client, token)

    extracted = JournalGuidelines(
        journal_name="Computers & Education",
        word_limit_total=8000,
        citation_style="APA 7th",
    )
    with patch(
        "app.services.journal_guidelines.get_journal_guidelines_agent",
        return_value=type("A", (), {"run": AsyncMock(return_value=type("R", (), {
            "output": extracted
        })())})(),
    ):
        response = await client.post(
            f"/api/v1/projects/{project_id}/fetch-journal-guidelines",
            json={"guidelines_text": "Manuscripts must not exceed 8000 words. Use APA 7th."},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 202
    task_id = response.json()["task_id"]
    uuid.UUID(task_id)

    task = await client.get(
        f"/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert task.status_code == 200
    body = task.json()
    assert body["status"] == "completed", body.get("error")
    assert body["result"]["word_limit_total"] == 8000

    project = await client.get(
        f"/api/v1/projects/{project_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert project.json()["target_journal_guidelines"]["word_limit_total"] == 8000


@pytest.mark.asyncio
async def test_fetch_guidelines_requires_target_journal(client):
    token = await _register(client)
    res = await client.post(
        "/api/v1/projects",
        json={"title": "No Journal"},
        headers={"Authorization": f"Bearer {token}"},
    )
    project_id = res.json()["id"]

    response = await client.post(
        f"/api/v1/projects/{project_id}/fetch-journal-guidelines",
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400

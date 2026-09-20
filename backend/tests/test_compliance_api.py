import uuid
from unittest.mock import AsyncMock, patch

import pytest


async def _run_compliance(client, token, draft_id):
    """POST check-compliance (202) then read the finished job result."""
    with patch(
        "app.agents.compliance_rules_agent.merge_compliance_rules",
        new_callable=AsyncMock,
        side_effect=RuntimeError("no LLM in tests"),
    ):
        response = await client.post(
            f"/api/v1/drafts/{draft_id}/check-compliance",
            json={},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    task = await client.get(
        f"/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert task.status_code == 200
    body = task.json()
    assert body["status"] == "completed", body.get("error")
    return body["result"]


async def get_auth_token(client):
    email = f"compliance-{uuid.uuid4().hex[:8]}@example.com"
    res = await client.post("/api/v1/auth/register", json={
        "email": email, "password": "StrongPass123!",
        "name": "Compliance User", "expertise_level": "researcher",
    })
    return res.json()["access_token"]


async def create_project(client, token):
    res = await client.post("/api/v1/projects", json={"title": "Compliance Project"},
                            headers={"Authorization": f"Bearer {token}"})
    return res.json()["id"]


@pytest.mark.asyncio
async def test_check_compliance_empty_draft(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Empty Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]

    data = await _run_compliance(client, token, draft_id)
    assert "overall_status" in data
    assert "checks" in data
    assert "total_word_count" in data
    assert data["total_word_count"] == 0


@pytest.mark.asyncio
async def test_check_compliance_with_content(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    create_res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Full Draft", "paper_type": "research_article"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = create_res.json()["id"]

    content = {
        "type": "doc",
        "content": [
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Introduction"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " ".join(["word"] * 1000)}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Literature Review"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " ".join(["word"] * 1000)}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Methods"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " ".join(["word"] * 500)}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Results"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " ".join(["word"] * 500)}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Discussion"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " ".join(["word"] * 500)}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Conclusion"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " ".join(["word"] * 200)}]},
        ],
    }
    await client.put(
        f"/api/v1/drafts/{draft_id}",
        json={"content": content},
        headers={"Authorization": f"Bearer {token}"},
    )

    data = await _run_compliance(client, token, draft_id)
    assert data["total_word_count"] > 3000
    assert len(data["sections_found"]) >= 5


@pytest.mark.asyncio
async def test_check_compliance_unauthorized(client):
    response = await client.post(
        "/api/v1/drafts/00000000-0000-0000-0000-000000000000/check-compliance",
        json={},
    )
    assert response.status_code == 401 or response.status_code == 403

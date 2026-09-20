"""POST /projects/{id}/smart-search: the WoS venue filter is optional.

v1.0.0 returned 400 whenever the Web of Science journal table was empty, which made
Smart Search unusable without Clarivate-licensed lists. The request now carries
``wos_filter`` ("auto" | "on" | "off"); only an explicit "on" with an empty table is
refused, with 409.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.services import smart_search as smart_search_module  # noqa: E402


async def _register(client, email: str) -> str:
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Smart Search User",
            "expertise_level": "researcher",
        },
    )
    assert res.status_code == 201, res.text
    return res.json()["access_token"]


async def _project(client, token: str) -> str:
    res = await client.post(
        "/api/v1/projects",
        json={"title": "Smart Search Project"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 201, res.text
    return res.json()["id"]


@pytest.fixture
def run_smart_search():
    """The background pipeline never runs in these tests; we only inspect its kwargs."""
    with patch.object(smart_search_module, "run_smart_search", new=AsyncMock()) as mock:
        yield mock


async def _post(client, token: str, project_id: str, body: dict):
    return await client.post(
        f"/api/v1/projects/{project_id}/smart-search",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
    )


def test_request_schema_defaults_to_auto():
    from app.schemas.smart_search import SmartSearchRequest

    assert SmartSearchRequest(query="q").wos_filter == "auto"
    assert SmartSearchRequest(query="q", wos_filter="off").wos_filter == "off"


@pytest.mark.asyncio
async def test_empty_journal_table_no_longer_blocks_smart_search(client, run_smart_search):
    token = await _register(client, "ss-auto@example.com")
    project_id = await _project(client, token)

    res = await _post(client, token, project_id, {"query": "teacher burnout"})

    assert res.status_code == 202, res.text
    assert "task_id" in res.json()
    run_smart_search.assert_awaited_once()
    kwargs = run_smart_search.await_args.kwargs
    assert kwargs["wos_filter"] == "auto"
    assert kwargs["query"] == "teacher burnout"
    assert kwargs["inclusion_criteria"] is None
    assert kwargs["exclusion_criteria"] is None


@pytest.mark.asyncio
async def test_wos_filter_on_with_empty_journal_table_returns_409(client, run_smart_search):
    token = await _register(client, "ss-on-empty@example.com")
    project_id = await _project(client, token)

    res = await _post(client, token, project_id, {"query": "q", "wos_filter": "on"})

    assert res.status_code == 409, res.text
    assert res.json()["detail"] == "WoS journal list not imported"
    run_smart_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_wos_filter_on_with_journals_imported_is_accepted(client, run_smart_search):
    token = await _register(client, "ss-on-full@example.com")
    project_id = await _project(client, token)

    with patch("app.api.smart_search.get_wos_journal_count", new=AsyncMock(return_value=42)):
        res = await _post(
            client, token, project_id,
            {
                "query": "q",
                "wos_filter": "on",
                "inclusion_criteria": ["empirical"],
                "exclusion_criteria": ["editorials"],
            },
        )

    assert res.status_code == 202, res.text
    kwargs = run_smart_search.await_args.kwargs
    assert kwargs["wos_filter"] == "on"
    assert kwargs["inclusion_criteria"] == ["empirical"]
    assert kwargs["exclusion_criteria"] == ["editorials"]


@pytest.mark.asyncio
async def test_wos_filter_off_is_passed_through(client, run_smart_search):
    token = await _register(client, "ss-off@example.com")
    project_id = await _project(client, token)

    res = await _post(client, token, project_id, {"query": "q", "wos_filter": "off"})

    assert res.status_code == 202, res.text
    assert run_smart_search.await_args.kwargs["wos_filter"] == "off"


@pytest.mark.asyncio
async def test_invalid_wos_filter_is_rejected(client, run_smart_search):
    token = await _register(client, "ss-invalid@example.com")
    project_id = await _project(client, token)

    res = await _post(client, token, project_id, {"query": "q", "wos_filter": "maybe"})

    assert res.status_code == 422
    run_smart_search.assert_not_awaited()

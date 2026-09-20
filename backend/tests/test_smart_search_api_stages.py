"""POST /projects/{id}/smart-search forwards the stage-aware criteria arrays.

``SmartSearchRequest.inclusion_criteria_stages``/``exclusion_criteria_stages`` and
``run_smart_search`` accept them, and the route itself must not drop both fields on the
floor: parsed and validated is not enough, they must also be forwarded. This file is
the route-level regression test for the wire contract: each stages list is either empty
or exactly parallel to its criteria list, values "abstract" or "full_text", else 422.
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
            "name": "Smart Search Stages User",
            "expertise_level": "researcher",
        },
    )
    assert res.status_code == 201, res.text
    return res.json()["access_token"]


async def _project(client, token: str) -> str:
    res = await client.post(
        "/api/v1/projects",
        json={"title": "Smart Search Stages Project"},
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


@pytest.mark.asyncio
async def test_parallel_stage_lists_reach_run_smart_search(client, run_smart_search):
    token = await _register(client, "ss-stages-parallel@example.com")
    project_id = await _project(client, token)

    res = await _post(
        client,
        token,
        project_id,
        {
            "query": "study strategy interventions",
            "inclusion_criteria": ["Administered the LASSI", "Reported an achievement outcome"],
            "exclusion_criteria": ["Book or book chapter"],
            "inclusion_criteria_stages": ["full_text", "abstract"],
            "exclusion_criteria_stages": ["abstract"],
        },
    )

    assert res.status_code == 202, res.text
    run_smart_search.assert_awaited_once()
    kwargs = run_smart_search.await_args.kwargs
    assert kwargs["inclusion_criteria_stages"] == ["full_text", "abstract"]
    assert kwargs["exclusion_criteria_stages"] == ["abstract"]
    assert kwargs["inclusion_criteria"] == [
        "Administered the LASSI",
        "Reported an achievement outcome",
    ]
    assert kwargs["exclusion_criteria"] == ["Book or book chapter"]


@pytest.mark.asyncio
async def test_empty_stage_lists_are_forwarded_as_none(client, run_smart_search):
    token = await _register(client, "ss-stages-empty@example.com")
    project_id = await _project(client, token)

    res = await _post(
        client,
        token,
        project_id,
        {
            "query": "study strategy interventions",
            "inclusion_criteria": ["Administered the LASSI"],
            "exclusion_criteria": [],
        },
    )

    assert res.status_code == 202, res.text
    run_smart_search.assert_awaited_once()
    kwargs = run_smart_search.await_args.kwargs
    # Neither field was sent: the schema default is [], and the route folds an empty
    # list to None so run_smart_search's own default (every criterion "abstract") applies.
    assert kwargs["inclusion_criteria_stages"] is None
    assert kwargs["exclusion_criteria_stages"] is None


@pytest.mark.asyncio
async def test_queries_override_and_publication_date_max_reach_run_smart_search(
    client, run_smart_search
):
    """Both replay fields are threaded to the background pipeline
    exactly as sent, so a replayed run reaches the same round loop the schema validated."""
    token = await _register(client, "ss-replay-fields@example.com")
    project_id = await _project(client, token)

    res = await _post(
        client,
        token,
        project_id,
        {
            "query": "study strategy interventions",
            "queries_override": [["round one query"], ["round two a", "round two b"]],
            "publication_date_max": "2026-09-08",
        },
    )

    assert res.status_code == 202, res.text
    run_smart_search.assert_awaited_once()
    kwargs = run_smart_search.await_args.kwargs
    assert kwargs["queries_override"] == [["round one query"], ["round two a", "round two b"]]
    assert str(kwargs["publication_date_max"]) == "2026-09-08"


@pytest.mark.asyncio
async def test_replay_fields_default_to_none(client, run_smart_search):
    token = await _register(client, "ss-replay-defaults@example.com")
    project_id = await _project(client, token)

    res = await _post(
        client, token, project_id, {"query": "study strategy interventions"},
    )

    assert res.status_code == 202, res.text
    run_smart_search.assert_awaited_once()
    kwargs = run_smart_search.await_args.kwargs
    assert kwargs["queries_override"] is None
    assert kwargs["publication_date_max"] is None


@pytest.mark.asyncio
async def test_mismatched_stage_length_is_rejected_with_422(client, run_smart_search):
    token = await _register(client, "ss-stages-mismatch@example.com")
    project_id = await _project(client, token)

    res = await _post(
        client,
        token,
        project_id,
        {
            "query": "study strategy interventions",
            "inclusion_criteria": ["Administered the LASSI", "Reported an achievement outcome"],
            "inclusion_criteria_stages": ["full_text"],
        },
    )

    assert res.status_code == 422, res.text
    run_smart_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_stage_value_is_rejected_with_422(client, run_smart_search):
    token = await _register(client, "ss-stages-badvalue@example.com")
    project_id = await _project(client, token)

    res = await _post(
        client,
        token,
        project_id,
        {
            "query": "study strategy interventions",
            "inclusion_criteria": ["Administered the LASSI"],
            "inclusion_criteria_stages": ["sometimes"],
        },
    )

    assert res.status_code == 422, res.text
    run_smart_search.assert_not_awaited()

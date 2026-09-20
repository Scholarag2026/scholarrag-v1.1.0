"""Shared OpenAlex relation helpers used by the graph build, expand and seed expansion.

Everything is mocked at the OpenAlexClient boundary — no network access.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.services.openalex_relations import (
    cap_works,
    fetch_relations,
    is_work_id,
    resolve_work_id,
)


def _work(w_id: str, *, refs: list[str] | None = None, citations: int = 0) -> dict:
    return {
        "doi": None,
        "title": f"Work {w_id}",
        "authors": [],
        "year": 2020,
        "journal_name": None,
        "journal_issn": None,
        "citation_count": citations,
        "abstract": None,
        "source_api": "openalex",
        "external_id": w_id,
        "full_text_url": None,
        "wos_collection": None,
        "wos_categories": None,
        "openalex_id": w_id,
        "referenced_works": list(refs or []),
    }


def _client(*, work=None, batch=None, cites=None, work_error=None) -> MagicMock:
    client = MagicMock()
    if work_error is not None:
        client.get_work = AsyncMock(side_effect=work_error)
    else:
        client.get_work = AsyncMock(return_value=work)
    client.get_works_batch = AsyncMock(return_value=list(batch or []))
    client.get_citations = AsyncMock(return_value=list(cites or []))
    return client


# --- id helpers ---


def test_is_work_id():
    assert is_work_id("W4381953137") is True
    assert is_work_id("  W123  ") is True
    assert is_work_id("W") is False
    assert is_work_id("649def34f8be52c8b66281af98ae884c09aef38b") is False
    assert is_work_id(None) is False


def test_resolve_work_id_prefers_a_stored_w_id():
    assert resolve_work_id("W4381953137", "10.1000/test") == "W4381953137"


def test_resolve_work_id_falls_back_to_doi():
    assert resolve_work_id(None, "10.1000/test") == "10.1000/test"


def test_resolve_work_id_ignores_a_semantic_scholar_paper_id():
    """A legacy S2 paperId is meaningless to OpenAlex — it must not be sent (issue #31)."""
    s2_paper_id = "649def34f8be52c8b66281af98ae884c09aef38b"
    assert resolve_work_id(s2_paper_id, None) is None
    assert resolve_work_id(s2_paper_id, "10.1000/test") == "10.1000/test"


def test_resolve_work_id_returns_none_without_identifiers():
    assert resolve_work_id(None, None) is None


# --- cap_works ---


def test_cap_works_keeps_the_most_cited():
    works = [_work(f"W{i}", citations=i * 10) for i in range(50)]
    capped = cap_works(works, 15)
    assert len(capped) == 15
    assert capped[0]["citation_count"] == 490
    assert capped[-1]["citation_count"] == 350


def test_cap_works_returns_all_when_under_the_limit():
    works = [_work(f"W{i}", citations=i) for i in range(5)]
    assert len(cap_works(works, 15)) == 5


def test_cap_works_tolerates_missing_citation_counts():
    works = [_work("W1"), {"openalex_id": "W2"}]
    assert len(cap_works(works, 5)) == 2


# --- fetch_relations ---


async def test_fetch_relations_gets_refs_and_cites_concurrently():
    client = _client(
        work=_work("W1", refs=["W2", "W3"]),
        batch=[_work("W2"), _work("W3")],
        cites=[_work("W9")],
    )

    refs, cites = await fetch_relations(client, "W1", max_refs=100, max_cites=50)

    client.get_work.assert_awaited_once_with("W1")
    client.get_works_batch.assert_awaited_once_with(["W2", "W3"])
    client.get_citations.assert_awaited_once_with("W1", per_page=50)
    assert [r["openalex_id"] for r in refs] == ["W2", "W3"]
    assert [c["openalex_id"] for c in cites] == ["W9"]


async def test_fetch_relations_caps_the_reference_id_list():
    client = _client(work=_work("W1", refs=[f"W{i}" for i in range(200)]))

    await fetch_relations(client, "W1", max_refs=100, max_cites=50)

    assert len(client.get_works_batch.await_args.args[0]) == 100


async def test_fetch_relations_returns_none_when_the_work_is_unknown():
    client = _client(work=None)

    assert await fetch_relations(client, "W404", max_refs=100, max_cites=50) is None
    client.get_works_batch.assert_not_awaited()
    client.get_citations.assert_not_awaited()


async def test_fetch_relations_skips_the_batch_call_when_there_are_no_references():
    client = _client(work=_work("W1", refs=[]), cites=[_work("W9")])

    refs, cites = await fetch_relations(client, "W1", max_refs=100, max_cites=50)

    assert refs == []
    assert len(cites) == 1
    client.get_works_batch.assert_not_awaited()


async def test_fetch_relations_propagates_http_errors():
    error = httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
    client = _client(work_error=error)

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_relations(client, "W1", max_refs=100, max_cites=50)


async def test_cites_rate_limit_returns_refs_only():
    """A 429 on the cites: query degrades to (refs, None) instead of failing the paper."""
    import httpx

    from app.services.openalex_relations import fetch_relations

    request = httpx.Request("GET", "https://api.openalex.org/works")
    response = httpx.Response(429, request=request, headers={"Retry-After": "33926"})
    err = httpx.HTTPStatusError("HTTP 429", request=request, response=response)

    client = MagicMock()
    client.get_work = AsyncMock(
        return_value={
            "openalex_id": "W1",
            "referenced_works": ["W2"],
            "title": "t",
            "doi": None,
        }
    )
    client.get_works_batch = AsyncMock(return_value=[{"openalex_id": "W2", "title": "r"}])
    client.get_citations = AsyncMock(side_effect=err)

    fetched = await fetch_relations(client, "W1", max_refs=10, max_cites=10)
    assert fetched is not None
    refs, cites = fetched
    assert [r["openalex_id"] for r in refs] == ["W2"]
    assert cites is None

"""Tests for the bounded concurrent query fan-out helper (T3 / SEARCH-SEQUENTIAL-QUERY-FANOUT)."""

import asyncio
from unittest.mock import patch

import pytest

from app.config import settings
from app.schemas.paper import PaperData
from app.services.search import SearchService, gather_search_results


def _paper(name: str) -> PaperData:
    return PaperData(
        doi=f"10.1/{name}",
        title=f"Paper {name}",
        authors=[{"name": "Test Author"}],
        year=2023,
        citation_count=1,
        source_api="openalex",
    )


@pytest.mark.asyncio
async def test_fanout_runs_more_than_one_query_at_a_time(monkeypatch):
    monkeypatch.setattr(settings, "search_fanout_concurrency", 6)
    state = {"active": 0, "peak": 0}

    async def _fake_search(self, query, **kwargs):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.05)
        state["active"] -= 1
        return [_paper(query)], {"openalex": 1}

    with patch.object(SearchService, "search", new=_fake_search):
        results, stopped, _failures = await gather_search_results(
            SearchService(), ["a", "b", "c", "d", "e", "f"],
        )

    assert state["peak"] == 6
    assert stopped is False
    assert len(results) == 6


@pytest.mark.asyncio
async def test_fanout_respects_the_concurrency_ceiling(monkeypatch):
    monkeypatch.setattr(settings, "search_fanout_concurrency", 2)
    state = {"active": 0, "peak": 0}

    async def _fake_search(self, query, **kwargs):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.05)
        state["active"] -= 1
        return [], {"openalex": 0}

    with patch.object(SearchService, "search", new=_fake_search):
        await gather_search_results(SearchService(), ["a", "b", "c", "d", "e", "f"])

    assert state["peak"] == 2


@pytest.mark.asyncio
async def test_fanout_preserves_input_order():
    async def _fake_search(self, query, **kwargs):
        # Later queries finish sooner — output must still follow input order.
        await asyncio.sleep(0.05 if query == "a" else 0.0)
        return [_paper(query)], {"openalex": 1}

    with patch.object(SearchService, "search", new=_fake_search):
        results, _, _ = await gather_search_results(SearchService(), ["a", "b", "c"])

    assert [q for q, _ in results] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_fanout_skips_failed_queries():
    async def _fake_search(self, query, **kwargs):
        if query == "boom":
            raise RuntimeError("provider exploded")
        return [_paper(query)], {"openalex": 1}

    with patch.object(SearchService, "search", new=_fake_search):
        results, stopped, _failures = await gather_search_results(
            SearchService(), ["a", "boom", "c"],
        )

    assert [q for q, _ in results] == ["a", "c"]
    assert stopped is False


@pytest.mark.asyncio
async def test_fanout_reports_failed_queries_with_the_error_text():
    """A query that raised is not silently dropped: the caller learns which one and why."""

    async def _fake_search(self, query, **kwargs):
        if query != "ok":
            raise RuntimeError(f"provider exploded on {query}")
        return [_paper(query)], {"openalex": 1}

    with patch.object(SearchService, "search", new=_fake_search):
        results, stopped, failures = await gather_search_results(
            SearchService(), ["boom1", "ok", "boom2"],
        )

    assert [q for q, _ in results] == ["ok"]
    assert stopped is False
    assert failures == [
        ("boom1", "RuntimeError: provider exploded on boom1"),
        ("boom2", "RuntimeError: provider exploded on boom2"),
    ]


def test_describe_search_failure_keeps_the_provider_message_and_hides_the_identity():
    import httpx

    from app.services.search import describe_search_failure

    request = httpx.Request(
        "GET",
        "https://api.openalex.org/works?search=x&mailto=someone%40example.org&api_key=sk-123",
    )
    response = httpx.Response(
        429,
        request=request,
        json={
            "error": "Rate limit exceeded",
            "message": "Insufficient budget. Resets at midnight UTC",
        },
    )
    exc = httpx.HTTPStatusError("boom", request=request, response=response)

    text = describe_search_failure(exc)

    assert text == (
        "HTTPStatusError: HTTP 429 Too Many Requests - Insufficient budget. Resets at midnight UTC"
    )
    assert "example.org" not in text and "sk-123" not in text

    plain = describe_search_failure(RuntimeError("timeout"))
    assert plain == "RuntimeError: timeout"
    leaked = describe_search_failure(
        RuntimeError("bad url https://x/?mailto=a%40b.c&api_key=k9&per_page=1")
    )
    assert leaked == "RuntimeError: bad url https://x/?mailto=***&api_key=***&per_page=1"


def test_describe_search_failure_caps_a_long_provider_message():
    """A verbose provider ``message`` body must not blow up a caller's fixed-width column
    (e.g. AnalysisJob.progress_message, String(500)) when several failures are joined into
    one status line."""
    import httpx

    from app.services.search import describe_search_failure

    request = httpx.Request("GET", "https://api.openalex.org/works?search=x")
    long_message = "Your query uses too many terms; " + ("narrow it down. " * 50)
    response = httpx.Response(
        429, request=request, json={"message": long_message},
    )
    exc = httpx.HTTPStatusError("boom", request=request, response=response)

    text = describe_search_failure(exc)

    assert len(text) < 250
    assert text.startswith("HTTPStatusError: HTTP 429 Too Many Requests - Your query uses")


@pytest.mark.asyncio
async def test_fanout_asks_the_service_to_raise_instead_of_swallowing():
    """The fan-out must see provider errors, so it opts out of the service's swallow."""
    seen: dict = {}

    async def _fake_search(self, query, **kwargs):
        seen.update(kwargs)
        return [], {"openalex": 0}

    with patch.object(SearchService, "search", new=_fake_search):
        await gather_search_results(SearchService(), ["a"])

    assert seen.get("raise_on_failure") is True


@pytest.mark.asyncio
async def test_fanout_stops_when_stop_check_returns_true(monkeypatch):
    monkeypatch.setattr(settings, "search_fanout_concurrency", 1)
    executed: list[str] = []

    async def _fake_search(self, query, **kwargs):
        executed.append(query)
        return [], {"openalex": 0}

    state = {"n": 0}

    async def _stop() -> bool:
        state["n"] += 1
        return state["n"] > 2

    with patch.object(SearchService, "search", new=_fake_search):
        results, stopped, _failures = await gather_search_results(
            SearchService(), ["a", "b", "c", "d"], stop_check=_stop,
        )

    assert executed == ["a", "b"]
    assert [q for q, _ in results] == ["a", "b"]
    assert stopped is True


@pytest.mark.asyncio
async def test_fanout_forwards_filters():
    seen: list[dict] = []

    async def _fake_search(self, query, **kwargs):
        seen.append(kwargs)
        return [], {"openalex": 0}

    with patch.object(SearchService, "search", new=_fake_search):
        await gather_search_results(
            SearchService(), ["a"], year_from=2020, year_to=2024, min_citations=5,
        )

    assert seen == [
        {
            "year_from": 2020,
            "year_to": 2024,
            "min_citations": 5,
            "to_publication_date": None,
            "raise_on_failure": True,
        }
    ]


@pytest.mark.asyncio
async def test_fanout_forwards_to_publication_date():
    """Forwarded to every query's own search call."""
    from datetime import date

    seen: list[dict] = []

    async def _fake_search(self, query, **kwargs):
        seen.append(kwargs)
        return [], {"openalex": 0}

    with patch.object(SearchService, "search", new=_fake_search):
        await gather_search_results(
            SearchService(), ["a", "b"], to_publication_date=date(2026, 9, 8),
        )

    assert all(kwargs["to_publication_date"] == date(2026, 9, 8) for kwargs in seen)


@pytest.mark.asyncio
async def test_fanout_with_no_queries_is_a_no_op():
    results, stopped, _failures = await gather_search_results(SearchService(), [])
    assert results == []
    assert stopped is False

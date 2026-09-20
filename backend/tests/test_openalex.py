# backend/tests/test_openalex.py
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.clients.openalex import OpenAlexClient


def _make_openalex_response(results: list[dict], total: int = 1) -> dict:
    return {
        "meta": {"count": total, "per_page": 25, "page": 1},
        "results": results,
    }


SAMPLE_WORK = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.1038/s41586-019-1724-z",
    "title": "Deep learning for computational biology",
    "authorships": [
        {"author": {"display_name": "John Smith"}},
        {"author": {"display_name": "Jane Doe"}},
    ],
    "publication_year": 2019,
    "primary_location": {
        "source": {
            "display_name": "Nature",
            "issn": ["0028-0836", "1476-4687"],
        }
    },
    "cited_by_count": 150,
    "abstract_inverted_index": {
        "Deep": [0],
        "learning": [1],
        "transforms": [2],
        "biology.": [3],
    },
}


@pytest.mark.asyncio
async def test_search_returns_papers():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_openalex_response([SAMPLE_WORK], total=1)

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    client = OpenAlexClient(http_client=mock_client)
    results = await client.search("deep learning")

    assert len(results) == 1
    paper = results[0]
    assert paper.title == "Deep learning for computational biology"
    assert paper.doi == "10.1038/s41586-019-1724-z"
    assert paper.year == 2019
    assert paper.journal_name == "Nature"
    assert paper.citation_count == 150
    assert paper.source_api == "openalex"
    assert len(paper.authors) == 2
    assert paper.authors[0]["name"] == "John Smith"
    assert paper.abstract == "Deep learning transforms biology."


@pytest.mark.asyncio
async def test_search_carries_openalex_type_and_is_paratext():
    """``search()`` (the Smart Search fan-out's own path, unlike ``get_work``/``get_works_batch``,
    which already carried these via ``parse_work_dict``) must also surface OpenAlex's
    ``type``/``is_paratext`` on the returned ``PaperData``, so ``run_smart_search`` can apply
    ``apply_type_demotion`` without a second request."""
    work = dict(SAMPLE_WORK, type="book", is_paratext=True)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_openalex_response([work], total=1)

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    client = OpenAlexClient(http_client=mock_client)
    results = await client.search("deep learning")

    assert len(results) == 1
    assert results[0].openalex_type == "book"
    assert results[0].is_paratext is True


@pytest.mark.asyncio
async def test_search_defaults_openalex_type_and_is_paratext_when_absent():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_openalex_response([SAMPLE_WORK], total=1)

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    client = OpenAlexClient(http_client=mock_client)
    results = await client.search("deep learning")

    assert results[0].openalex_type is None
    assert results[0].is_paratext is False


@pytest.mark.asyncio
async def test_search_with_filters():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_openalex_response([], total=0)

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    client = OpenAlexClient(http_client=mock_client)
    await client.search("test", year_from=2020, year_to=2024, min_citations=10)

    call_kwargs = mock_client.get.call_args
    params = call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
    assert "publication_year:2020-2024" in params.get("filter", "")
    filt = params.get("filter", "")
    assert "cited_by_count:>10" in filt or "cited_by_count:>9" in filt


@pytest.mark.asyncio
async def test_search_with_to_publication_date_filter():
    """A ``to_publication_date`` filter freezes the corpus a replayed
    run sees to whatever OpenAlex indexed on or before that date."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_openalex_response([], total=0)

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    client = OpenAlexClient(http_client=mock_client)
    await client.search("test", to_publication_date=date(2026, 9, 8))

    call_kwargs = mock_client.get.call_args
    params = call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
    assert "to_publication_date:2026-09-08" in params.get("filter", "")


@pytest.mark.asyncio
async def test_search_without_to_publication_date_omits_the_filter():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_openalex_response([], total=0)

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    client = OpenAlexClient(http_client=mock_client)
    await client.search("test")

    call_kwargs = mock_client.get.call_args
    params = call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
    assert "to_publication_date" not in params.get("filter", "")


@pytest.mark.asyncio
async def test_search_combines_to_publication_date_with_other_filters():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_openalex_response([], total=0)

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    client = OpenAlexClient(http_client=mock_client)
    await client.search(
        "test", year_from=2020, min_citations=10, to_publication_date=date(2026, 9, 8)
    )

    call_kwargs = mock_client.get.call_args
    params = call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
    filt = params.get("filter", "")
    assert "publication_year:2020-" in filt
    assert "cited_by_count:>10" in filt or "cited_by_count:>9" in filt
    assert "to_publication_date:2026-09-08" in filt


@pytest.mark.asyncio
async def test_search_empty_results():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_openalex_response([], total=0)

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    client = OpenAlexClient(http_client=mock_client)
    results = await client.search("xyznonexistent")
    assert results == []


@pytest.mark.asyncio
async def test_search_handles_missing_fields():
    """Papers with missing optional fields should still parse."""
    work = {
        "id": "https://openalex.org/W123",
        "doi": None,
        "title": "A paper without DOI",
        "authorships": [],
        "publication_year": None,
        "primary_location": None,
        "cited_by_count": 0,
        "abstract_inverted_index": None,
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_openalex_response([work])

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    client = OpenAlexClient(http_client=mock_client)
    results = await client.search("test")
    assert len(results) == 1
    assert results[0].doi is None
    assert results[0].abstract is None


# --- T2: wildcard sanitization (issue OPENALEX-WILDCARD-400 / decision D1) ---


def test_sanitize_query_strips_wildcards():
    from app.clients.openalex import sanitize_query

    assert sanitize_query("compar* AND contrast*") == "compar AND contrast"
    assert sanitize_query('"student*" learning') == '"student" learning'
    assert sanitize_query("teach?ng methods") == "teachng methods"


def test_sanitize_query_collapses_whitespace_and_trims():
    from app.clients.openalex import sanitize_query

    assert sanitize_query("  machine   learning  ") == "machine learning"
    assert sanitize_query("   ") == ""
    assert sanitize_query("***") == ""


def test_sanitize_query_caps_length_on_a_word_boundary():
    from app.clients.openalex import sanitize_query
    from app.config import settings

    long_query = "translation " * 100  # 1200 chars
    cleaned = sanitize_query(long_query)
    assert len(cleaned) <= settings.openalex_max_query_length
    assert not cleaned.endswith(" ")


async def test_search_sends_sanitized_query_and_select():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_openalex_response([], total=0)

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    client = OpenAlexClient(http_client=mock_client)
    await client.search("compar* education")

    params = mock_client.get.call_args.kwargs["params"]
    assert params["search"] == "compar education"
    assert "*" not in params["search"]
    assert params["select"].startswith("id,doi,title")
    assert "abstract_inverted_index" in params["select"]


async def test_search_skips_the_request_when_query_is_all_wildcards():
    mock_client = AsyncMock()
    client = OpenAlexClient(http_client=mock_client)

    assert await client.search("* ?") == []
    mock_client.get.assert_not_called()


# --- T2: polite pool ---


async def test_search_sends_mailto_and_user_agent(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "openalex_email", "team@example.test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_openalex_response([], total=0)

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    client = OpenAlexClient(http_client=mock_client)
    await client.search("deep learning")

    call = mock_client.get.call_args
    assert call.kwargs["params"]["mailto"] == "team@example.test"
    assert "mailto:team@example.test" in call.kwargs["headers"]["User-Agent"]


# --- T2: retry policy (issue S2-RETRY-PERMANENT-4XX applied to OpenAlex) ---


def _error_response(code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = code
    resp.is_success = False
    resp.reason_phrase = "Error"
    resp.url = "https://api.openalex.org/works"
    resp.request = MagicMock()
    resp.headers = {}
    return resp


async def test_search_fails_fast_on_permanent_4xx():
    mock_client = AsyncMock()
    mock_client.get.return_value = _error_response(400)

    client = OpenAlexClient(http_client=mock_client)
    with pytest.raises(httpx.HTTPStatusError):
        await client.search("anything")

    assert mock_client.get.await_count == 1


async def test_search_retries_transient_429(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "external_retry_min_wait", 0.0)
    monkeypatch.setattr(settings, "external_retry_max_wait", 0.0)

    ok = MagicMock()
    ok.status_code = 200
    ok.raise_for_status = MagicMock()
    ok.json.return_value = _make_openalex_response([SAMPLE_WORK], total=1)

    mock_client = AsyncMock()
    mock_client.get.side_effect = [_error_response(429), ok]

    client = OpenAlexClient(http_client=mock_client)
    results = await client.search("deep learning")

    assert len(results) == 1
    assert mock_client.get.await_count == 2


# --- T2: shared singletons (issue RATE-LIMITER-NOT-SHARED) ---


def test_clients_share_one_rate_limiter():
    a = OpenAlexClient()
    b = OpenAlexClient()
    assert a._limiter is b._limiter


def test_clients_share_one_pooled_http_client():
    from app.clients.openalex import get_shared_http_client

    assert OpenAlexClient()._get_client() is get_shared_http_client()
    assert OpenAlexClient()._get_client() is get_shared_http_client()

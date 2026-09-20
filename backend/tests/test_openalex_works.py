# backend/tests/test_openalex_works.py
"""Single-work / batch / citations reads on the OpenAlex client.

Request shapes were verified live against api.openalex.org on 2026-07-26:
  GET /works/W2741809807?select=...,referenced_works        -> 200
  GET /works/doi:10.7717/peerj.4375?select=...              -> 200
  GET /works?filter=openalex_id:W1|W2&select=...&per_page=50 -> 200
  GET /works?filter=cites:W2741809807&select=...            -> 200
  GET /works/W9999999999                                    -> 404
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.clients.openalex import (
    WORK_SELECT_FIELDS,
    OpenAlexClient,
    normalize_work_id,
    parse_work_dict,
    short_work_id,
    to_paper_data,
)

FULL_WORK = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.7717/peerj.4375",
    "title": "The state of OA",
    "authorships": [{"author": {"display_name": "Heather Piwowar"}}],
    "publication_year": 2018,
    "primary_location": {"source": {"display_name": "PeerJ", "issn": ["2167-8359"]}},
    "cited_by_count": 900,
    "abstract_inverted_index": {"Open": [0], "access": [1]},
    "open_access": {"oa_url": "https://peerj.com/articles/4375.pdf"},
    "referenced_works": [
        "https://openalex.org/W1560783210",
        "https://openalex.org/W1724212071",
    ],
    "type": "article",
    "is_paratext": False,
}


def _ok(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    return resp


def _not_found() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 404
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("404", request=MagicMock(), response=resp)
    )
    return resp


# --- pure helpers ---


def test_short_work_id():
    assert short_work_id("https://openalex.org/W123") == "W123"
    assert short_work_id("W123") == "W123"
    assert short_work_id(None) is None


def test_normalize_work_id():
    assert normalize_work_id("W123") == "W123"
    assert normalize_work_id("https://openalex.org/W123") == "W123"
    assert normalize_work_id("10.7717/peerj.4375") == "doi:10.7717/peerj.4375"
    assert normalize_work_id("https://doi.org/10.7717/peerj.4375") == "doi:10.7717/peerj.4375"
    assert normalize_work_id("doi:10.7717/peerj.4375") == "doi:10.7717/peerj.4375"


def test_parse_work_dict_shape():
    parsed = parse_work_dict(FULL_WORK)

    assert parsed["doi"] == "10.7717/peerj.4375"
    assert parsed["title"] == "The state of OA"
    assert parsed["year"] == 2018
    assert parsed["journal_name"] == "PeerJ"
    assert parsed["journal_issn"] == "2167-8359"
    assert parsed["citation_count"] == 900
    assert parsed["abstract"] == "Open access"
    assert parsed["source_api"] == "openalex"
    assert parsed["external_id"] == "W2741809807"
    assert parsed["full_text_url"] == "https://peerj.com/articles/4375.pdf"
    # T2 additions
    assert parsed["openalex_id"] == "W2741809807"
    assert parsed["referenced_works"] == ["W1560783210", "W1724212071"]
    assert parsed["openalex_type"] == "article"
    assert parsed["is_paratext"] is False


def test_parse_work_dict_without_references():
    parsed = parse_work_dict({"id": "https://openalex.org/W9", "title": "x"})
    assert parsed["referenced_works"] == []
    assert parsed["openalex_id"] == "W9"
    assert parsed["openalex_type"] is None
    assert parsed["is_paratext"] is False


def test_work_select_fields_requests_type_and_is_paratext():
    """Both are free (OpenAlex bills nothing for
    extra select fields), requested unconditionally so a screening harness can apply the
    narrow non-article-type demotion without a second request."""
    fields = WORK_SELECT_FIELDS.split(",")
    assert "type" in fields
    assert "is_paratext" in fields


def test_parse_work_dict_carries_a_paratext_flag_and_a_nonarticle_type():
    parsed = parse_work_dict(
        {
            "id": "https://openalex.org/W1",
            "title": "Front matter",
            "type": "paratext",
            "is_paratext": True,
        }
    )
    assert parsed["openalex_type"] == "paratext"
    assert parsed["is_paratext"] is True


def test_to_paper_data_round_trip():
    paper = to_paper_data(parse_work_dict(FULL_WORK))
    assert paper.title == "The state of OA"
    assert paper.external_id == "W2741809807"
    assert paper.source_api == "openalex"


# --- get_work ---


async def test_get_work_by_w_id_requests_references():
    mock_client = AsyncMock()
    mock_client.get.return_value = _ok(FULL_WORK)

    result = await OpenAlexClient(http_client=mock_client).get_work("W2741809807")

    url = mock_client.get.call_args.args[0]
    params = mock_client.get.call_args.kwargs["params"]
    assert url == "https://api.openalex.org/works/W2741809807"
    assert "referenced_works" in params["select"]
    assert result["openalex_id"] == "W2741809807"
    assert result["referenced_works"] == ["W1560783210", "W1724212071"]


async def test_get_work_by_doi_uses_the_doi_path_prefix():
    mock_client = AsyncMock()
    mock_client.get.return_value = _ok(FULL_WORK)

    await OpenAlexClient(http_client=mock_client).get_work("10.7717/peerj.4375")

    assert (
        mock_client.get.call_args.args[0]
        == "https://api.openalex.org/works/doi:10.7717/peerj.4375"
    )


async def test_get_work_returns_none_on_404():
    mock_client = AsyncMock()
    mock_client.get.return_value = _not_found()

    assert await OpenAlexClient(http_client=mock_client).get_work("W9999999999") is None


async def test_get_work_propagates_permanent_errors():
    resp = MagicMock()
    resp.status_code = 403
    resp.is_success = False
    resp.reason_phrase = "Forbidden"
    resp.url = "https://api.openalex.org/works/W1"
    resp.request = MagicMock()
    resp.headers = {}
    mock_client = AsyncMock()
    mock_client.get.return_value = resp

    with pytest.raises(httpx.HTTPStatusError):
        await OpenAlexClient(http_client=mock_client).get_work("W1")
    assert mock_client.get.await_count == 1


# --- get_works_batch ---


async def test_get_works_batch_builds_an_or_filter():
    mock_client = AsyncMock()
    mock_client.get.return_value = _ok({"results": [FULL_WORK]})

    results = await OpenAlexClient(http_client=mock_client).get_works_batch(
        ["https://openalex.org/W1", "W2", "W1", "", None]
    )

    params = mock_client.get.call_args.kwargs["params"]
    assert params["filter"] == "openalex_id:W1|W2"  # de-duped, order preserved
    assert params["per_page"] == 2
    assert "referenced_works" in params["select"]
    assert len(results) == 1
    assert results[0]["openalex_id"] == "W2741809807"


async def test_get_works_batch_chunks_at_the_configured_maximum(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "openalex_batch_size", 50)

    mock_client = AsyncMock()
    mock_client.get.return_value = _ok({"results": []})

    ids = [f"W{i}" for i in range(120)]
    await OpenAlexClient(http_client=mock_client).get_works_batch(ids)

    assert mock_client.get.await_count == 3  # 50 + 50 + 20
    first_filter = mock_client.get.call_args_list[0].kwargs["params"]["filter"]
    last_filter = mock_client.get.call_args_list[2].kwargs["params"]["filter"]
    assert len(first_filter.removeprefix("openalex_id:").split("|")) == 50
    assert len(last_filter.removeprefix("openalex_id:").split("|")) == 20


async def test_get_works_batch_makes_no_request_for_empty_input():
    mock_client = AsyncMock()
    assert await OpenAlexClient(http_client=mock_client).get_works_batch([]) == []
    mock_client.get.assert_not_called()


# --- get_citations ---


async def test_get_citations_uses_the_cites_filter():
    mock_client = AsyncMock()
    mock_client.get.return_value = _ok({"results": [FULL_WORK]})

    results = await OpenAlexClient(http_client=mock_client).get_citations(
        "https://openalex.org/W3009335665", per_page=25
    )

    params = mock_client.get.call_args.kwargs["params"]
    assert mock_client.get.call_args.args[0] == "https://api.openalex.org/works"
    assert params["filter"] == "cites:W3009335665"
    assert params["per_page"] == 25
    assert params["sort"] == "cited_by_count:desc"
    assert len(results) == 1
    assert results[0]["referenced_works"] == []


async def test_get_citations_clamps_per_page_to_the_api_maximum():
    mock_client = AsyncMock()
    mock_client.get.return_value = _ok({"results": []})

    await OpenAlexClient(http_client=mock_client).get_citations("W1", per_page=1000)

    assert mock_client.get.call_args.kwargs["params"]["per_page"] == 200


async def test_get_citations_rejects_a_doi():
    mock_client = AsyncMock()

    assert await OpenAlexClient(http_client=mock_client).get_citations("10.1/abc") == []
    mock_client.get.assert_not_called()


async def test_get_works_batch_falls_back_to_singles_on_429():
    """A quota-length 429 on the batch filter hydrates via single-work GETs instead."""
    import httpx

    from app.clients.openalex import OpenAlexClient

    request = httpx.Request("GET", "https://api.openalex.org/works")
    response = httpx.Response(429, request=request, headers={"Retry-After": "33926"})
    err = httpx.HTTPStatusError("HTTP 429", request=request, response=response)

    client = OpenAlexClient()
    calls: list[str] = []

    async def fake_get_json(path, params):
        calls.append(path)
        if path == "/works":  # the batch filter endpoint
            raise err
        wid = path.rsplit("/", 1)[-1]
        return {"id": f"https://openalex.org/{wid}", "title": f"work {wid}"}

    client._get_json = fake_get_json  # type: ignore[method-assign]

    works = await client.get_works_batch(["W1", "W2", "W1"])  # duplicate collapses
    assert [w["openalex_id"] for w in works] == ["W1", "W2"]
    assert calls[0] == "/works"
    assert calls[1:] == ["/works/W1", "/works/W2"]


async def test_get_json_sends_api_key_when_configured(monkeypatch):
    """The account API key must reach every request so usage bills the account."""
    from unittest.mock import AsyncMock, MagicMock

    from app.clients import openalex as oa
    from app.config import settings

    monkeypatch.setattr(settings, "openalex_api_key", "test-key-123")
    client = oa.OpenAlexClient()
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"ok": True}
    http = MagicMock()
    http.get = AsyncMock(return_value=resp)
    client._client = http
    client._limiter = MagicMock(acquire=AsyncMock())

    await client._get_json("/works/W1", {"select": "id"})
    _, kwargs = http.get.call_args
    assert kwargs["params"]["api_key"] == "test-key-123"

    monkeypatch.setattr(settings, "openalex_api_key", "")
    await client._get_json("/works/W1", {"select": "id"})
    _, kwargs = http.get.call_args
    assert "api_key" not in kwargs["params"]


async def test_get_works_batch_raises_when_every_single_fails():
    """All-HTTP-failure must propagate — an outage is not 'this paper has no refs'."""
    import httpx

    from app.clients.openalex import OpenAlexClient

    def _err(code):
        request = httpx.Request("GET", "https://api.openalex.org/works")
        response = httpx.Response(
            code, request=request, headers={"Retry-After": "33926"} if code == 429 else {}
        )
        return httpx.HTTPStatusError(f"HTTP {code}", request=request, response=response)

    client = OpenAlexClient()

    async def fake_get_json(path, params):
        raise _err(429)

    client._get_json = fake_get_json  # type: ignore[method-assign]
    with pytest.raises(httpx.HTTPStatusError):
        await client.get_works_batch(["W1", "W2"])


async def test_get_works_batch_skips_probe_while_filter_block_memo_active():
    """After one quota 429, later batch calls go straight to singles — no re-probe."""
    import httpx

    from app.clients.openalex import OpenAlexClient, filter_queries_blocked

    request = httpx.Request("GET", "https://api.openalex.org/works")
    response = httpx.Response(429, request=request, headers={"Retry-After": "33926"})
    err = httpx.HTTPStatusError("HTTP 429", request=request, response=response)

    client = OpenAlexClient()
    calls: list[str] = []

    async def fake_get_json(path, params):
        calls.append(path)
        if path == "/works":
            raise err
        wid = path.rsplit("/", 1)[-1]
        return {"id": f"https://openalex.org/{wid}", "title": wid}

    client._get_json = fake_get_json  # type: ignore[method-assign]

    await client.get_works_batch(["W1"])  # probe 429s, single succeeds, memo set
    assert filter_queries_blocked() is True
    calls.clear()
    await client.get_works_batch(["W2"])  # memo active: no probe at all
    assert calls == ["/works/W2"]


async def test_http_errors_redact_the_api_key(monkeypatch):
    """str(exc) reaches logs and user-visible job errors — the key must never ride along."""
    import httpx

    from app.clients import openalex as oa
    from app.config import settings

    monkeypatch.setattr(settings, "openalex_api_key", "SECRET-KEY-xyz")
    client = oa.OpenAlexClient()

    real_url = "https://api.openalex.org/works/W1?select=id&api_key=SECRET-KEY-xyz"

    async def fake_get(url, params=None, headers=None):
        request = httpx.Request("GET", real_url)
        return httpx.Response(429, request=request, headers={"Retry-After": "33926"})

    http = MagicMock()
    http.get = fake_get
    client._client = http
    client._limiter = MagicMock(acquire=AsyncMock())

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client._get_json("/works/W1", {"select": "id"})
    message = str(exc_info.value)
    assert "SECRET-KEY-xyz" not in message
    assert "api_key=***" in message

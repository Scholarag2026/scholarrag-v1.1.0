# backend/tests/test_crossref_client.py
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.clients.crossref import CrossRefClient, get_shared_http_client
from app.config import settings

CROSSREF_OK = {
    "status": "ok",
    "message": {
        "DOI": "10.1038/s41586-019-1724-z",
        "title": ["Deep learning for computational biology"],
        "author": [{"given": "John", "family": "Smith"}],
        "published-print": {"date-parts": [[2019]]},
        "container-title": ["Nature"],
        "ISSN": ["0028-0836"],
        "is-referenced-by-count": 150,
    },
}


def _ok(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    return resp


def _error(code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = code
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            f"HTTP {code}", request=MagicMock(), response=resp
        )
    )
    return resp


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(settings, "external_retry_min_wait", 0.0)
    monkeypatch.setattr(settings, "external_retry_max_wait", 0.0)


async def test_verify_doi_exists():
    mock_client = AsyncMock()
    mock_client.get.return_value = _ok(CROSSREF_OK)

    result = await CrossRefClient(http_client=mock_client).verify_doi(
        "10.1038/s41586-019-1724-z"
    )

    assert result is not None
    assert result["doi"] == "10.1038/s41586-019-1724-z"
    assert result["title"] == "Deep learning for computational biology"
    assert result["year"] == 2019
    assert result["journal_name"] == "Nature"
    assert result["issn"] == "0028-0836"
    assert result["citation_count"] == 150
    assert result["authors"] == [{"name": "John Smith"}]


async def test_verify_doi_not_found():
    mock_client = AsyncMock()
    mock_client.get.return_value = _error(404)

    assert await CrossRefClient(http_client=mock_client).verify_doi("10.1234/nope") is None
    assert mock_client.get.await_count == 1


async def test_verify_doi_does_not_retry_permanent_4xx():
    mock_client = AsyncMock()
    mock_client.get.return_value = _error(403)

    assert await CrossRefClient(http_client=mock_client).verify_doi("10.1/x") is None
    assert mock_client.get.await_count == 1


async def test_verify_doi_retries_transient_5xx():
    mock_client = AsyncMock()
    mock_client.get.side_effect = [_error(503), _ok(CROSSREF_OK)]

    result = await CrossRefClient(http_client=mock_client).verify_doi("10.1/x")

    assert result is not None
    assert mock_client.get.await_count == 2


async def test_verify_doi_network_error_returns_none():
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.ConnectError("Connection failed")

    assert await CrossRefClient(http_client=mock_client).verify_doi("10.1/x") is None
    assert mock_client.get.await_count == settings.external_retry_attempts


def test_clients_share_one_pooled_http_client():
    assert CrossRefClient()._get_client() is get_shared_http_client()
    assert CrossRefClient()._get_client() is get_shared_http_client()

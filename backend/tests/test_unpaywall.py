# backend/tests/test_unpaywall.py
"""Tests for the Unpaywall API client."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.clients.unpaywall import UnpaywallClient

# -- Sample Unpaywall API responses --

OA_RESPONSE = {
    "doi": "10.1038/s41586-019-1724-z",
    "is_oa": True,
    "best_oa_location": {
        "url_for_pdf": "https://europepmc.org/articles/pmc6919535?pdf=render",
        "url": "https://europepmc.org/articles/pmc6919535",
        "is_best": True,
    },
}

OA_RESPONSE_NO_PDF = {
    "doi": "10.1234/oa-no-pdf",
    "is_oa": True,
    "best_oa_location": {
        "url_for_pdf": None,
        "url": "https://archive.org/details/some-article",
        "is_best": True,
    },
}

NON_OA_RESPONSE = {
    "doi": "10.1016/j.cell.2020.01.001",
    "is_oa": False,
    "best_oa_location": None,
}

# `oa_locations` deliberately NOT in publishedVersion/acceptedVersion/submittedVersion
# order, so a test that finds them back in that order is actually exercising the
# client's own sort rather than an incidental match to the fixture's own order.
OA_RESPONSE_MULTI_LOCATIONS = {
    "doi": "10.1/multi",
    "is_oa": True,
    "best_oa_location": {
        "url_for_pdf": "https://example.org/published.pdf",
        "url": "https://example.org/published",
        "version": "publishedVersion",
    },
    "oa_locations": [
        {
            "url_for_pdf": None,
            "url": "https://example.org/submitted",
            "version": "submittedVersion",
        },
        {
            "url_for_pdf": "https://example.org/accepted.pdf",
            "url": "https://example.org/accepted",
            "version": "acceptedVersion",
        },
        {
            "url_for_pdf": "https://example.org/published.pdf",
            "url": "https://example.org/published",
            "version": "publishedVersion",
        },
        {
            "url_for_pdf": "https://example.org/other.pdf",
            "url": "https://example.org/other",
            "version": "some_other_version",
        },
    ],
}


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def _mock_404_response() -> MagicMock:
    """Create a mock 404 httpx Response."""
    resp = MagicMock()
    resp.status_code = 404
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "Not found", request=MagicMock(), response=resp,
        ),
    )
    return resp


@pytest.mark.asyncio
@patch("app.clients.unpaywall.settings")
async def test_lookup_oa_paper_returns_pdf_url(mock_settings):
    """An OA paper with url_for_pdf should return the PDF URL."""
    mock_settings.unpaywall_email = "test@example.com"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_response(OA_RESPONSE))

    client = UnpaywallClient(http_client=mock_client)
    result = await client.lookup("10.1038/s41586-019-1724-z")

    assert result == "https://europepmc.org/articles/pmc6919535?pdf=render"
    mock_client.get.assert_called_once_with(
        "/10.1038/s41586-019-1724-z",
        params={"email": "test@example.com"},
    )


@pytest.mark.asyncio
@patch("app.clients.unpaywall.settings")
async def test_lookup_oa_paper_falls_back_to_url(mock_settings):
    """When url_for_pdf is None, should fall back to url."""
    mock_settings.unpaywall_email = "test@example.com"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_response(OA_RESPONSE_NO_PDF))

    client = UnpaywallClient(http_client=mock_client)
    result = await client.lookup("10.1234/oa-no-pdf")

    assert result == "https://archive.org/details/some-article"


@pytest.mark.asyncio
@patch("app.clients.unpaywall.settings")
async def test_lookup_non_oa_paper_returns_none(mock_settings):
    """A non-OA paper should return None."""
    mock_settings.unpaywall_email = "test@example.com"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_response(NON_OA_RESPONSE))

    client = UnpaywallClient(http_client=mock_client)
    result = await client.lookup("10.1016/j.cell.2020.01.001")

    assert result is None


@pytest.mark.asyncio
@patch("app.clients.unpaywall.settings")
async def test_lookup_invalid_doi_returns_none(mock_settings):
    """A 404 (invalid DOI) should return None without raising."""
    mock_settings.unpaywall_email = "test@example.com"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_404_response())

    client = UnpaywallClient(http_client=mock_client)
    result = await client.lookup("10.9999/does-not-exist")

    assert result is None


@pytest.mark.asyncio
@patch("app.clients.unpaywall.settings")
async def test_lookup_without_email_returns_none(mock_settings):
    """If unpaywall_email is not configured, should return None immediately."""
    mock_settings.unpaywall_email = ""

    mock_client = AsyncMock()

    client = UnpaywallClient(http_client=mock_client)
    result = await client.lookup("10.1038/s41586-019-1724-z")

    assert result is None
    mock_client.get.assert_not_called()


# --- T2: retry policy + shared pooled client ---


def _error_response(code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = code
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            f"HTTP {code}", request=MagicMock(), response=resp
        )
    )
    return resp


@pytest.mark.asyncio
@patch("app.clients.unpaywall.settings")
async def test_lookup_does_not_retry_permanent_4xx(mock_settings):
    mock_settings.unpaywall_email = "test@example.com"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_error_response(403))

    result = await UnpaywallClient(http_client=mock_client).lookup("10.1/x")

    assert result is None
    assert mock_client.get.await_count == 1


@pytest.mark.asyncio
async def test_lookup_retries_transient_503(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "unpaywall_email", "test@example.com")
    monkeypatch.setattr(settings, "external_retry_min_wait", 0.0)
    monkeypatch.setattr(settings, "external_retry_max_wait", 0.0)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        side_effect=[_error_response(503), _mock_response(OA_RESPONSE)]
    )

    result = await UnpaywallClient(http_client=mock_client).lookup("10.1/x")

    assert result == "https://europepmc.org/articles/pmc6919535?pdf=render"
    assert mock_client.get.await_count == 2


def test_clients_share_one_pooled_http_client():
    from app.clients.unpaywall import BASE_URL, get_shared_http_client

    shared = get_shared_http_client()
    assert UnpaywallClient()._get_client() is shared
    assert UnpaywallClient()._get_client() is shared
    assert str(shared.base_url).rstrip("/") == BASE_URL


# --- F4 acquisition: every OA location, ordered publishedVersion > acceptedVersion >
# submittedVersion > anything else, each contributing url_for_pdf else url. ---


@pytest.mark.asyncio
@patch("app.clients.unpaywall.settings")
async def test_lookup_locations_orders_published_accepted_submitted_then_other(
    mock_settings,
):
    mock_settings.unpaywall_email = "test@example.com"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_response(OA_RESPONSE_MULTI_LOCATIONS))

    client = UnpaywallClient(http_client=mock_client)
    result = await client.lookup_locations("10.1/multi")

    assert result == [
        "https://example.org/published.pdf",
        "https://example.org/accepted.pdf",
        "https://example.org/submitted",
        "https://example.org/other.pdf",
    ]


@pytest.mark.asyncio
@patch("app.clients.unpaywall.settings")
async def test_lookup_locations_falls_back_to_best_oa_location_when_no_oa_locations_list(
    mock_settings,
):
    """A response shaped like the older single-location fixtures (no ``oa_locations``
    key) still yields the one URL Unpaywall did give, rather than an empty list."""
    mock_settings.unpaywall_email = "test@example.com"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_response(OA_RESPONSE))

    client = UnpaywallClient(http_client=mock_client)
    result = await client.lookup_locations("10.1038/s41586-019-1724-z")

    assert result == ["https://europepmc.org/articles/pmc6919535?pdf=render"]


@pytest.mark.asyncio
@patch("app.clients.unpaywall.settings")
async def test_lookup_locations_returns_empty_for_non_oa_paper(mock_settings):
    mock_settings.unpaywall_email = "test@example.com"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_response(NON_OA_RESPONSE))

    client = UnpaywallClient(http_client=mock_client)
    result = await client.lookup_locations("10.1016/j.cell.2020.01.001")

    assert result == []


@pytest.mark.asyncio
@patch("app.clients.unpaywall.settings")
async def test_lookup_locations_without_email_returns_empty_list_and_makes_no_call(
    mock_settings,
):
    mock_settings.unpaywall_email = ""

    mock_client = AsyncMock()

    client = UnpaywallClient(http_client=mock_client)
    result = await client.lookup_locations("10.1038/s41586-019-1724-z")

    assert result == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
@patch("app.clients.unpaywall.settings")
async def test_lookup_locations_invalid_doi_returns_empty_list(mock_settings):
    mock_settings.unpaywall_email = "test@example.com"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_404_response())

    client = UnpaywallClient(http_client=mock_client)
    result = await client.lookup_locations("10.9999/does-not-exist")

    assert result == []


@pytest.mark.asyncio
async def test_lookup_locations_retries_transient_503(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "unpaywall_email", "test@example.com")
    monkeypatch.setattr(settings, "external_retry_min_wait", 0.0)
    monkeypatch.setattr(settings, "external_retry_max_wait", 0.0)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        side_effect=[_error_response(503), _mock_response(OA_RESPONSE_MULTI_LOCATIONS)]
    )

    result = await UnpaywallClient(http_client=mock_client).lookup_locations("10.1/x")

    assert result[0] == "https://example.org/published.pdf"
    assert mock_client.get.await_count == 2


@pytest.mark.asyncio
@patch("app.clients.unpaywall.settings")
async def test_lookup_locations_skips_a_location_with_neither_pdf_nor_url(mock_settings):
    mock_settings.unpaywall_email = "test@example.com"
    response = {
        "doi": "10.1/gap",
        "is_oa": True,
        "best_oa_location": None,
        "oa_locations": [
            {"url_for_pdf": None, "url": None, "version": "publishedVersion"},
            {
                "url_for_pdf": "https://example.org/accepted.pdf",
                "url": None,
                "version": "acceptedVersion",
            },
        ],
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_response(response))

    client = UnpaywallClient(http_client=mock_client)
    result = await client.lookup_locations("10.1/gap")

    assert result == ["https://example.org/accepted.pdf"]

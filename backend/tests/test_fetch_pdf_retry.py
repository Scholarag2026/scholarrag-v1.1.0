# backend/tests/test_fetch_pdf_retry.py
"""F4 acquisition: `fetch_pdf_from_url` retries a blocked/wrong-content-type download.

A response that looks like a publisher interstitial rather than a PDF (HTML
content-type, or a body that does not start with the ``%PDF`` magic bytes), or an
HTTP 403, 429 or 5xx, is retried after 3 s and then after 10 s, the two retries using
the browser-standard headers ``demo/tools/verify_pdf.py`` documents under
``--browser-ua``. Every attempt is logged with the host, status and content-type it
received, and a URL where every attempt looked like an interstitial raises
``PublisherInterstitialError`` rather than silently returning ``None``.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

import httpx  # noqa: E402
import pytest  # noqa: E402

from app.services.fulltext import (  # noqa: E402
    PDF_FETCH_BROWSER_HEADERS,
    PublisherInterstitialError,
    fetch_pdf_from_url,
)

PDF_BYTES = b"%PDF-1.4 fake pdf body"


def _client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Every test in this module controls timing via the recorded sleep calls, not by
    actually waiting -- record the delays `fetch_pdf_from_url` asks for instead."""
    delays: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("app.services.fulltext.asyncio.sleep", _fake_sleep)
    return delays


@pytest.mark.asyncio
async def test_first_attempt_success_returns_bytes_without_browser_headers():
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=PDF_BYTES)

    result = await fetch_pdf_from_url(
        "https://example.org/paper.pdf", http_client=_client_for(handler)
    )

    assert result == PDF_BYTES
    assert len(seen_requests) == 1
    assert seen_requests[0].headers.get("user-agent", "") != PDF_FETCH_BROWSER_HEADERS[
        "User-Agent"
    ]


@pytest.mark.asyncio
async def test_html_interstitial_is_retried_after_3s_with_browser_headers_then_succeeds(
    no_real_sleep,
):
    seen_requests: list[httpx.Request] = []
    responses = [
        httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>blocked</html>"),
        httpx.Response(200, headers={"content-type": "application/pdf"}, content=PDF_BYTES),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return responses[len(seen_requests) - 1]

    result = await fetch_pdf_from_url(
        "https://example.org/paper.pdf", http_client=_client_for(handler)
    )

    assert result == PDF_BYTES
    assert len(seen_requests) == 2
    assert no_real_sleep == [3.0]
    assert seen_requests[0].headers.get("user-agent", "") != PDF_FETCH_BROWSER_HEADERS[
        "User-Agent"
    ]
    assert seen_requests[1].headers["user-agent"] == PDF_FETCH_BROWSER_HEADERS["User-Agent"]


@pytest.mark.asyncio
async def test_a_200_body_not_starting_with_pdf_is_also_retried():
    """A publisher interstitial can be served as a plain 200 with an HTML-less but
    still non-PDF body (e.g. a JSON "please log in" page) -- the %PDF magic-byte
    check catches this even when the content-type header lies."""
    seen_requests: list[httpx.Request] = []
    responses = [
        httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"not a pdf"),
        httpx.Response(200, headers={"content-type": "application/pdf"}, content=PDF_BYTES),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return responses[len(seen_requests) - 1]

    result = await fetch_pdf_from_url(
        "https://example.org/paper.pdf", http_client=_client_for(handler)
    )

    assert result == PDF_BYTES
    assert len(seen_requests) == 2


@pytest.mark.asyncio
async def test_403_then_429_then_success_retries_after_3s_and_10s(no_real_sleep):
    seen_requests: list[httpx.Request] = []
    responses = [
        httpx.Response(403, headers={"content-type": "text/html"}, content=b"forbidden"),
        httpx.Response(429, headers={"content-type": "text/html"}, content=b"rate limited"),
        httpx.Response(200, headers={"content-type": "application/pdf"}, content=PDF_BYTES),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return responses[len(seen_requests) - 1]

    result = await fetch_pdf_from_url(
        "https://example.org/paper.pdf", http_client=_client_for(handler)
    )

    assert result == PDF_BYTES
    assert len(seen_requests) == 3
    assert no_real_sleep == [3.0, 10.0]


@pytest.mark.asyncio
async def test_5xx_is_retried_and_recovers():
    seen_requests: list[httpx.Request] = []
    responses = [
        httpx.Response(503, headers={"content-type": "text/plain"}, content=b"unavailable"),
        httpx.Response(200, headers={"content-type": "application/pdf"}, content=PDF_BYTES),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return responses[len(seen_requests) - 1]

    result = await fetch_pdf_from_url(
        "https://example.org/paper.pdf", http_client=_client_for(handler)
    )

    assert result == PDF_BYTES
    assert len(seen_requests) == 2


@pytest.mark.asyncio
async def test_every_attempt_returning_html_raises_publisher_interstitial_error(
    no_real_sleep,
):
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<html>always blocked</html>"
        )

    with pytest.raises(PublisherInterstitialError):
        await fetch_pdf_from_url(
            "https://example.org/paper.pdf", http_client=_client_for(handler)
        )

    assert len(seen_requests) == 3
    assert no_real_sleep == [3.0, 10.0]


@pytest.mark.asyncio
async def test_a_transport_failure_is_not_retried_and_returns_none(no_real_sleep):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = await fetch_pdf_from_url(
        "https://example.org/paper.pdf", http_client=_client_for(handler)
    )

    assert result is None
    assert no_real_sleep == []


@pytest.mark.asyncio
async def test_a_404_html_page_is_not_retried_and_is_not_reported_as_interstitial(
    no_real_sleep,
):
    """A 404 is a missing document, not a publisher block --
    retrying it wastes 13 s per candidate, and `_pdf_response_looks_like_interstitial`
    being true for any non-PDF body would otherwise raise `PublisherInterstitialError`
    and mislabel a dead OA link as "blocked by the publisher"."""
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            404, headers={"content-type": "text/html"}, content=b"<html>not found</html>"
        )

    result = await fetch_pdf_from_url(
        "https://example.org/paper.pdf", http_client=_client_for(handler)
    )

    assert result is None
    assert len(seen_requests) == 1
    assert no_real_sleep == []


@pytest.mark.asyncio
async def test_a_410_html_page_is_not_retried_and_is_not_reported_as_interstitial(
    no_real_sleep,
):
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            410, headers={"content-type": "text/html"}, content=b"<html>gone</html>"
        )

    result = await fetch_pdf_from_url(
        "https://example.org/paper.pdf", http_client=_client_for(handler)
    )

    assert result is None
    assert len(seen_requests) == 1
    assert no_real_sleep == []


@pytest.mark.asyncio
async def test_every_attempt_is_logged_with_host_status_and_content_type(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<html>blocked</html>"
        )

    with caplog.at_level("INFO", logger="app.services.fulltext"):
        with pytest.raises(PublisherInterstitialError):
            await fetch_pdf_from_url(
                "https://example.org/paper.pdf", http_client=_client_for(handler)
            )

    attempt_logs = [r.message for r in caplog.records if "example.org" in r.message]
    assert len(attempt_logs) == 3
    for message in attempt_logs:
        assert "200" in message
        assert "text/html" in message

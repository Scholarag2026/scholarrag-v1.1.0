"""Program-wide retry policy: transient failures only (issue S2-RETRY-PERMANENT-4XX).

A permanent 4xx (400/401/403/404) must cost exactly one request. Retrying those is
what turned a dead credential into ~30s of backoff per call in production.
"""

import httpx
import pytest

from app.clients.http_retry import (
    is_transient_error,
    request_with_retry,
)
from app.config import settings


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.test/works")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"HTTP {code}", request=request, response=response)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Keep the retry tests instant."""
    monkeypatch.setattr(settings, "external_retry_min_wait", 0.0)
    monkeypatch.setattr(settings, "external_retry_max_wait", 0.0)


@pytest.mark.parametrize("code", [408, 425, 429, 500, 502, 503, 504])
def test_transient_status_codes_are_retryable(code):
    assert is_transient_error(_status_error(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 422])
def test_permanent_status_codes_are_not_retryable(code):
    assert is_transient_error(_status_error(code)) is False


def test_network_errors_are_retryable():
    assert is_transient_error(httpx.ConnectError("boom")) is True
    assert is_transient_error(httpx.ReadTimeout("boom")) is True
    assert is_transient_error(httpx.ConnectTimeout("boom")) is True


def test_unrelated_exceptions_are_not_retryable():
    assert is_transient_error(ValueError("nope")) is False


async def test_permanent_error_is_attempted_once():
    calls = {"n": 0}

    async def call() -> httpx.Response:
        calls["n"] += 1
        raise _status_error(403)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await request_with_retry(call)

    assert exc_info.value.response.status_code == 403
    assert calls["n"] == 1


async def test_transient_error_is_retried_then_succeeds():
    calls = {"n": 0}

    async def call() -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _status_error(429)
        return httpx.Response(200, request=httpx.Request("GET", "https://example.test/"))

    response = await request_with_retry(call)

    assert response.status_code == 200
    assert calls["n"] == 3


async def test_retry_budget_is_bounded_and_reraises():
    calls = {"n": 0}

    async def call() -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("upstream stalled")

    with pytest.raises(httpx.ReadTimeout):
        await request_with_retry(call)

    assert calls["n"] == settings.external_retry_attempts == 3


def _rate_limit_error(retry_after: str | None) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.test/works")
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    response = httpx.Response(429, request=request, headers=headers)
    return httpx.HTTPStatusError("HTTP 429", request=request, response=response)


def test_429_with_quota_length_retry_after_fails_fast():
    """A Retry-After of hours (observed live from OpenAlex) must not be retried."""
    from app.clients.http_retry import is_transient_error

    assert is_transient_error(_rate_limit_error("33926")) is False


def test_429_with_short_retry_after_is_retryable():
    from app.clients.http_retry import is_transient_error

    assert is_transient_error(_rate_limit_error("5")) is True
    assert is_transient_error(_rate_limit_error(None)) is True
    assert is_transient_error(_rate_limit_error("not-a-number")) is True


def test_retry_after_seconds_parsing():
    from app.clients.http_retry import retry_after_seconds

    assert retry_after_seconds(_rate_limit_error("12")) == 12.0
    assert retry_after_seconds(_rate_limit_error(None)) is None
    assert retry_after_seconds(_rate_limit_error("soon")) is None
    assert retry_after_seconds(_status_error(500)) is None


def test_retry_after_http_date_form_is_parsed():
    """RFC 7231 HTTP-date Retry-After must not be misread as 'short/transient'."""
    from datetime import datetime, timedelta, timezone

    from app.clients.http_retry import is_transient_error, retry_after_seconds

    future = datetime.now(timezone.utc) + timedelta(hours=9)
    stamp = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    err = _rate_limit_error(stamp)
    parsed = retry_after_seconds(err)
    assert parsed is not None and parsed > 8 * 3600
    assert is_transient_error(err) is False  # quota-length → fail fast


def test_transient_wait_honours_short_retry_after_exactly():
    from unittest.mock import MagicMock

    from app.clients.http_retry import _TransientWait

    wait = _TransientWait(multiplier=1, min=1.0, max=8.0)
    state = MagicMock()
    state.attempt_number = 2
    state.outcome.failed = True
    state.outcome.exception.return_value = _rate_limit_error("12")
    assert wait(state) == 12.0

    state.outcome.exception.return_value = _rate_limit_error("33926")
    # Beyond the cap the exponential fallback applies (bounded by max=8).
    assert wait(state) <= 8.0

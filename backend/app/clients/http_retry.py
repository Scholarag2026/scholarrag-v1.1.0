"""Shared retry policy for outbound calls to external APIs.

Program-wide standard: retry ONLY transient failures — 429, 5xx,
connect errors and read/connect timeouts — for at most
``settings.external_retry_attempts`` attempts with exponential backoff between
``external_retry_min_wait`` and ``external_retry_max_wait`` seconds. Every other 4xx
fails fast: retrying a permanent 400/401/403/404 buys nothing and, in production,
cost ~30s of backoff on every single call.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings

TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Seconds from a 429's ``Retry-After`` header, or ``None`` when absent/unparseable.

    Handles both the delta-seconds form and the RFC 7231 HTTP-date form.
    """
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        raw = exc.response.headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
            try:
                from email.utils import parsedate_to_datetime

                when = parsedate_to_datetime(raw)
                from datetime import datetime, timezone

                return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                return None
    return None


def is_transient_error(exc: BaseException) -> bool:
    """True when *exc* has a realistic chance of succeeding on a retry."""
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 429:
            wait_s = retry_after_seconds(exc)
            # A quota-length Retry-After (observed: ~9.4 HOURS from OpenAlex on
            # Zeabur's shared egress IP) cannot be waited out inside a request
            # or job — fail fast so callers can take their fallback paths.
            if wait_s is not None and wait_s > settings.external_retry_after_cap:
                return False
            return True
        return exc.response.status_code in TRANSIENT_STATUS_CODES
    return isinstance(exc, TRANSIENT_EXCEPTIONS)


class _TransientWait(wait_exponential):
    """Exponential backoff that honours a short 429 ``Retry-After`` exactly.

    Documented deviation: a server-stated Retry-After up to
    ``external_retry_after_cap`` (20s) is honoured verbatim, BYPASSING the
    ``external_retry_max_wait`` (8s) ceiling — waiting the time the server asked
    for beats blind backoff that is guaranteed to re-429. Consequences: one call
    can block up to attempts-1 times the cap, and test fixtures that zero
    min/max backoff do NOT suppress these waits (tests must avoid short
    Retry-After values through ``request_with_retry`` unless they mean to sleep).
    """

    def __call__(self, retry_state) -> float:  # type: ignore[override]
        outcome = retry_state.outcome
        if outcome is not None and outcome.failed:
            wait_s = retry_after_seconds(outcome.exception())
            if wait_s is not None and 0 <= wait_s <= settings.external_retry_after_cap:
                return wait_s
        return super().__call__(retry_state)


def transient_retry(attempts: int | None = None):
    """Build a tenacity decorator that retries transient failures only.

    The tenacity config is rebuilt on every call so that settings overrides
    (tests, env changes) are honoured.
    """
    return retry(
        retry=retry_if_exception(is_transient_error),
        wait=_TransientWait(
            multiplier=1,
            min=settings.external_retry_min_wait,
            max=settings.external_retry_max_wait,
        ),
        stop=stop_after_attempt(attempts or settings.external_retry_attempts),
        reraise=True,
    )


async def request_with_retry(
    call: Callable[[], Awaitable[httpx.Response]],
    *,
    attempts: int | None = None,
) -> httpx.Response:
    """Await *call* under the shared transient-only retry policy.

    Args:
        call: A zero-argument coroutine function performing one HTTP request and
            raising ``httpx.HTTPStatusError`` for statuses it wants retried.
        attempts: Override the configured attempt budget.

    Returns:
        The successful ``httpx.Response``.

    Raises:
        The last exception raised by *call* once the budget is exhausted, or
        immediately for any non-transient exception.
    """
    return await transient_retry(attempts)(call)()

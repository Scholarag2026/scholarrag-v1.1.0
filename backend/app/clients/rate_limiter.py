"""Async rate limiters for external API clients.

Limiters obtained through :func:`get_rate_limiter` are process-wide singletons, so
every job and every client instance shares one ceiling per upstream source
(issue RATE-LIMITER-NOT-SHARED). Note the ceiling is per *process*: the deployment
runs 2 uvicorn workers, so the effective global rate is 2x the configured value.
"""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """Spaces calls so that at most ``rate`` acquisitions happen per ``period``."""

    def __init__(self, rate: float, period: float = 1.0):
        self._rate = rate
        self._period = period
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._last_call = 0.0

    def _get_lock(self) -> asyncio.Lock:
        """Return a lock bound to the currently running event loop.

        asyncio primitives bind to the first loop that contends for them and then
        refuse to be used from any other loop. A process-wide singleton therefore
        has to rebuild its lock whenever the running loop changes.
        """
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def acquire(self) -> None:
        async with self._get_lock():
            now = time.monotonic()
            min_interval = self._period / self._rate
            elapsed = now - self._last_call
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_call = time.monotonic()


_LIMITERS: dict[str, AsyncRateLimiter] = {}


def get_rate_limiter(name: str, rate: float, period: float = 1.0) -> AsyncRateLimiter:
    """Return the process-wide limiter registered under *name*.

    The first caller decides ``rate``/``period``; later callers get that instance.
    """
    limiter = _LIMITERS.get(name)
    if limiter is None:
        limiter = AsyncRateLimiter(rate=rate, period=period)
        _LIMITERS[name] = limiter
    return limiter


def reset_rate_limiters() -> None:
    """Drop every registered limiter. Tests only."""
    _LIMITERS.clear()

"""AsyncRateLimiter must be shareable process-wide (issue RATE-LIMITER-NOT-SHARED).

Two properties are required:
  1. get_rate_limiter() returns the SAME instance for the same name, so N concurrent
     jobs share one ceiling instead of each believing it may use the full budget.
  2. A long-lived singleton must survive being awaited on more than one event loop.
     asyncio.Lock binds itself to the first loop that contends for it, so a naive
     instance-level lock raises "is bound to a different event loop" in the second job
     (and in the test suite, which creates one loop per test).
"""

import asyncio

from app.clients.rate_limiter import (
    AsyncRateLimiter,
    get_rate_limiter,
    reset_rate_limiters,
)


def test_get_rate_limiter_returns_one_instance_per_name():
    reset_rate_limiters()
    try:
        a = get_rate_limiter("openalex", 8)
        b = get_rate_limiter("openalex", 8)
        c = get_rate_limiter("crossref", 5)
        assert a is b
        assert a is not c
    finally:
        reset_rate_limiters()


def test_limiter_survives_multiple_event_loops():
    """Three concurrent acquires force lock contention, which binds the loop."""
    limiter = AsyncRateLimiter(rate=50)

    async def hammer() -> None:
        await asyncio.gather(*(limiter.acquire() for _ in range(3)))

    asyncio.run(hammer())
    asyncio.run(hammer())  # different loop — must not raise


async def test_acquire_spaces_calls_by_the_configured_interval():
    limiter = AsyncRateLimiter(rate=20)  # 50ms minimum interval
    await limiter.acquire()
    start = asyncio.get_running_loop().time()
    await limiter.acquire()
    elapsed = asyncio.get_running_loop().time() - start
    assert elapsed >= 0.04

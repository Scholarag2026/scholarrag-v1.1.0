"""Tests for the shared async engine's connection-pool configuration (#44 pool half)."""


def test_engine_pool_is_configured_from_settings():
    from app.config import settings
    from app.database import engine

    pool = engine.pool
    assert pool.size() == settings.db_pool_size
    assert pool._max_overflow == settings.db_max_overflow
    assert pool._timeout == settings.db_pool_timeout


def test_engine_uses_pre_ping():
    """Recycled/dead connections must be detected before a job borrows them."""
    from app.database import engine

    assert engine.pool._pre_ping is True


def test_pool_ceiling_is_safe_for_two_uvicorn_workers():
    """2 workers x (pool_size + max_overflow) must stay well under Postgres max_connections."""
    from app.config import settings

    assert (settings.db_pool_size + settings.db_max_overflow) * 2 <= 60


def test_pool_timeout_fails_fast():
    """A request must not silently block for 30s when the pool is exhausted."""
    from app.config import settings

    assert settings.db_pool_timeout <= 10

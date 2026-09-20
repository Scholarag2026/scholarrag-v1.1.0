"""Settings that bound the search pipelines (T3)."""

from app.config import Settings


def test_deep_search_has_a_wall_clock_cap():
    defaults = Settings()
    assert defaults.deep_search_max_time_minutes == 30.0


def test_search_fanout_concurrency_default():
    defaults = Settings()
    assert defaults.search_fanout_concurrency == 4
    assert 1 <= defaults.search_fanout_concurrency <= 8


def test_time_caps_accept_fractional_minutes():
    """Sub-minute budgets must be representable so the caps are testable."""
    tuned = Settings(
        smart_search_max_time_minutes=0.25,
        deep_search_max_time_minutes=0.5,
    )
    assert tuned.smart_search_max_time_minutes == 0.25
    assert tuned.deep_search_max_time_minutes == 0.5

"""Defaults for the T4 citation-graph build settings.

These bound the whole graph pipeline: concurrency, wall clock, per-paper fan-out and the
staleness window used by the build-graph 409 guard (decision D12c). A silent default change
would silently un-fix GRAPH-SEQUENTIAL-NO-CONCURRENCY or GRAPH-STALE-JOB-BLOCKS-REBUILD.
"""

from app.config import Settings


def _settings() -> Settings:
    """Settings built without reading a local .env, so defaults are deterministic."""
    return Settings(_env_file=None)


def test_build_concurrency_defaults():
    s = _settings()
    assert s.graph_build_concurrency == 4
    assert s.graph_build_max_minutes == 20


def test_per_paper_fanout_defaults():
    s = _settings()
    assert s.graph_max_refs_per_paper == 100
    assert s.graph_max_cites_per_paper == 50


def test_stale_job_window_default():
    """D12c: a running build with no progress for 10 minutes is abandoned, not a 409."""
    s = _settings()
    assert s.graph_stale_job_minutes == 10


def test_existing_graph_settings_are_untouched():
    s = _settings()
    assert s.graph_max_nodes == 80
    assert s.graph_expand_max_refs == 15
    assert s.graph_expand_max_cites == 15


def test_s2_rate_limit_delay_is_gone():
    """The graph loop is limiter-bound now; the sleep knob has no readers left."""
    assert not hasattr(_settings(), "s2_rate_limit_delay")

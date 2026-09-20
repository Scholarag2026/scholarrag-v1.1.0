"""Tests for quantitative analysis API routes."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")


def test_trigger_quantitative_route_exists():
    """Verify the POST trigger endpoint is registered."""
    from app.api.quantitative import router

    routes = [r.path for r in router.routes]
    assert "/datasets/{dataset_id}/analyze/quantitative" in routes


def test_get_quantitative_route_exists():
    """Verify the GET results endpoint is registered."""
    from app.api.quantitative import router

    routes = [r.path for r in router.routes]
    assert "/datasets/{dataset_id}/analysis/quantitative" in routes

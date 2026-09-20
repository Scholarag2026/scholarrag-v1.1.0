"""Tests for qualitative analysis API routes."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")


def test_trigger_qualitative_route_exists():
    """Verify the POST trigger endpoint is registered."""
    from app.api.qualitative import router

    routes = [r.path for r in router.routes]
    assert "/datasets/{dataset_id}/analyze/qualitative" in routes


def test_get_qualitative_route_exists():
    """Verify the GET results endpoint is registered."""
    from app.api.qualitative import router

    routes = [r.path for r in router.routes]
    assert "/datasets/{dataset_id}/analysis/qualitative" in routes


def test_update_codebook_route_exists():
    """Verify the PUT codebook endpoint is registered."""
    from app.api.qualitative import router

    routes = [r.path for r in router.routes]
    assert "/datasets/{dataset_id}/codebook" in routes


def test_update_coding_session_route_exists():
    """Verify the PUT coding-session endpoint is registered."""
    from app.api.qualitative import router

    routes = [r.path for r in router.routes]
    assert "/datasets/{dataset_id}/coding-session" in routes


def test_trigger_inter_coder_route_exists():
    """Verify the POST inter-coder-reliability endpoint is registered."""
    from app.api.qualitative import router

    routes = [r.path for r in router.routes]
    assert "/datasets/{dataset_id}/inter-coder-reliability" in routes


def test_get_inter_coder_route_exists():
    """Verify the GET inter-coder-reliability endpoint is registered."""
    from app.api.qualitative import router

    routes = [r.path for r in router.routes]
    assert "/datasets/{dataset_id}/inter-coder-reliability" in routes

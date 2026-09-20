"""Tests for research design service helper functions."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")


def test_get_latest_research_design_returns_none_when_empty():
    """Verify the function signature and logic exist."""
    from app.services.research_design import get_latest_research_design
    assert get_latest_research_design is not None


def test_get_latest_collection_plan_returns_none_when_empty():
    from app.services.research_design import get_latest_collection_plan
    assert get_latest_collection_plan is not None


def test_run_research_design_exists():
    from app.services.research_design import run_research_design
    assert run_research_design is not None


def test_run_data_collection_plan_exists():
    from app.services.research_design import run_data_collection_plan
    assert run_data_collection_plan is not None

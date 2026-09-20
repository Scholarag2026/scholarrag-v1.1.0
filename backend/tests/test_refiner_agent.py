"""Tests for the draft refiner agent."""
import os
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.schemas.refine import RefinedText


def test_refined_text_schema():
    data = RefinedText(refined="This is a refined paragraph.")
    assert data.refined == "This is a refined paragraph."


def test_refiner_agent_module_imports():
    from app.agents.refiner_agent import refine_text
    assert callable(refine_text)


def test_refine_route_exists():
    """Verify refine endpoint is registered."""
    from app.api.drafts import router
    routes = [r.path for r in router.routes]
    assert any("refine" in r for r in routes)

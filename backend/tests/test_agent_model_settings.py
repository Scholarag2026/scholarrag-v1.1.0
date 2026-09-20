"""Every agent must carry an explicit LLM timeout and retry budget.

Issue LLM-NO-TIMEOUT / decision D3: with no ``model_settings`` pydantic-ai uses a
600s per-attempt default and the OpenAI SDK retries twice, so a single stalled
DeepSeek call can block a job for ~1800s. ``analysis_max_retries`` was declared in
config but wired to nothing; it is now the Agent retry budget.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

import pytest  # noqa: E402

from app.config import settings  # noqa: E402

FAST = settings.llm_timeout_seconds
LONG = settings.llm_long_timeout_seconds


def _agent_factories() -> list[tuple[str, object, int]]:
    from app.agents.analysis_agent import get_analysis_agent
    from app.agents.citation_link_agent import get_citation_link_agent
    from app.agents.claim_verification_agent import get_claim_verification_agent
    from app.agents.compliance_rules_agent import get_compliance_rules_agent
    from app.agents.data_collection_agent import get_data_collection_agent
    from app.agents.field_foundations_agent import get_field_foundations_agent
    from app.agents.gap_agent import get_gap_agent
    from app.agents.journal_guidelines_agent import get_journal_guidelines_agent
    from app.agents.paper_metadata_agent import _get_agent as get_paper_metadata_agent
    from app.agents.paper_selector_agent import get_paper_selector_agent
    from app.agents.qualitative_agent import get_qualitative_coding_agent
    from app.agents.quantitative_agent import (
        get_quantitative_interpretation_agent,
        get_quantitative_plan_agent,
    )
    from app.agents.query_expansion_agent import get_query_expansion_agent
    from app.agents.query_generator_agent import get_query_generator_agent
    from app.agents.refiner_agent import _get_agent as get_refiner_agent
    from app.agents.relevance_screener_agent import get_relevance_screener_agent
    from app.agents.research_design_agent import get_research_design_agent
    from app.agents.scope_refiner_agent import get_scope_refiner_agent
    from app.agents.writing_rules_agent import get_writing_rules_agent

    return [
        ("analysis", get_analysis_agent, LONG),
        ("citation_link", get_citation_link_agent, FAST),
        ("claim_verification", get_claim_verification_agent, LONG),
        ("compliance_rules", get_compliance_rules_agent, FAST),
        ("data_collection", get_data_collection_agent, LONG),
        ("field_foundations", get_field_foundations_agent, FAST),
        ("gap", get_gap_agent, LONG),
        ("journal_guidelines", get_journal_guidelines_agent, FAST),
        ("paper_metadata", get_paper_metadata_agent, FAST),
        ("paper_selector", get_paper_selector_agent, FAST),
        ("qualitative_coding", get_qualitative_coding_agent, LONG),
        ("quantitative_plan", get_quantitative_plan_agent, LONG),
        ("quantitative_interpretation", get_quantitative_interpretation_agent, LONG),
        ("query_expansion", get_query_expansion_agent, FAST),
        ("query_generator", get_query_generator_agent, FAST),
        ("refiner", get_refiner_agent, FAST),
        ("relevance_screener", get_relevance_screener_agent, FAST),
        ("research_design", get_research_design_agent, LONG),
        ("scope_refiner", get_scope_refiner_agent, FAST),
        ("writing_rules", get_writing_rules_agent, FAST),
    ]


_CASES = _agent_factories()


def test_model_config_tiers():
    from app.agents.model_config import (
        AGENT_RETRIES,
        FAST_MODEL_SETTINGS,
        LONG_MODEL_SETTINGS,
    )

    assert FAST_MODEL_SETTINGS["timeout"] == settings.llm_timeout_seconds == 60
    assert LONG_MODEL_SETTINGS["timeout"] == settings.llm_long_timeout_seconds == 300
    assert AGENT_RETRIES == settings.analysis_max_retries == 2


def test_all_agent_factories_are_covered():
    """Guard against a new agent being added without a timeout."""
    import pathlib

    agents_dir = pathlib.Path(__file__).resolve().parents[1] / "app" / "agents"
    skip = {"__init__.py", "model_config.py", "deep_analysis_agent.py"}
    modules = sorted(p.name for p in agents_dir.glob("*.py") if p.name not in skip)
    covered = {name for name, _, _ in _CASES}
    # 19 modules, quantitative_agent contributing
    # two factories
    assert len(modules) == 19
    assert len(covered) == 20


@pytest.mark.parametrize(
    "name,factory,expected_timeout", _CASES, ids=[c[0] for c in _CASES]
)
def test_agent_has_explicit_timeout_and_retries(name, factory, expected_timeout):
    agent = factory()
    assert agent.model_settings is not None, f"{name}: no model_settings"
    assert agent.model_settings["timeout"] == expected_timeout, f"{name}: wrong timeout"
    # pydantic-ai 1.107 stores Agent(retries=N) on these private attributes.
    assert agent._max_tool_retries == settings.analysis_max_retries
    assert agent._max_output_retries == settings.analysis_max_retries

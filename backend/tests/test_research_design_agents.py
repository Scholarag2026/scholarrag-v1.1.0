"""Tests for Research Design and Data Collection agents."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")


def test_research_design_agent_exists():
    from app.agents.research_design_agent import get_research_design_agent
    agent = get_research_design_agent()
    assert agent is not None


def test_format_research_design_prompt():
    from app.agents.research_design_agent import format_research_design_prompt
    result = format_research_design_prompt(
        research_question="How does X affect Y?",
        project_description="EdTech in higher education",
        target_journal="BJET",
        gap_summary="The field lacks studies on X in Y context.",
    )
    assert "How does X affect Y?" in result
    assert "EdTech" in result
    assert "BJET" in result
    assert "lacks studies" in result


def test_format_research_design_prompt_no_gap():
    from app.agents.research_design_agent import format_research_design_prompt
    result = format_research_design_prompt(
        research_question="RQ1",
        project_description=None,
        target_journal=None,
        gap_summary=None,
    )
    assert "RQ1" in result


def test_data_collection_agent_exists():
    from app.agents.data_collection_agent import get_data_collection_agent
    agent = get_data_collection_agent()
    assert agent is not None


def test_format_collection_prompt():
    from app.agents.data_collection_agent import format_collection_prompt
    design_dict = {
        "methodology": {"approach": "qualitative", "design_type": "case study"},
        "instruments": [{"name": "Interview Protocol", "type": "interview"}],
        "sampling": {"sample_size": 30, "target_population": "teachers"},
    }
    result = format_collection_prompt(
        design=design_dict,
        project_description="EdTech study",
    )
    assert "qualitative" in result
    assert "Interview Protocol" in result
    assert "30" in result

"""Tests for the Qualitative Coding Agent."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")


def test_qualitative_coding_agent_exists():
    from app.agents.qualitative_agent import get_qualitative_coding_agent
    agent = get_qualitative_coding_agent()
    assert agent is not None


def test_format_coding_prompt_basic():
    from app.agents.qualitative_agent import format_coding_prompt
    result = format_coding_prompt(
        text_segments=["I feel empowered by technology", "The system is hard to use"],
        research_questions=["How do students perceive EdTech?"],
    )
    assert "I feel empowered" in result
    assert "hard to use" in result
    assert "How do students perceive" in result


def test_format_coding_prompt_with_codebook_and_methodology():
    from app.agents.qualitative_agent import format_coding_prompt
    codebook = {"codes": [{"id": "c1", "name": "Empowerment"}], "themes": []}
    result = format_coding_prompt(
        text_segments=["Segment A"],
        research_questions=["RQ1"],
        existing_codebook=codebook,
        methodology="grounded theory",
    )
    assert "Empowerment" in result
    assert "grounded theory" in result

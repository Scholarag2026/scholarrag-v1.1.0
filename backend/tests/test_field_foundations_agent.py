"""Tests for Field Foundations agent."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")


def test_field_foundations_agent_exists():
    from app.agents.field_foundations_agent import get_field_foundations_agent

    agent = get_field_foundations_agent()
    assert agent is not None


def test_format_foundations_prompt_basic():
    from app.agents.field_foundations_agent import format_foundations_prompt

    result = format_foundations_prompt(
        topic="Educational Technology",
    )
    assert "Educational Technology" in result
    assert "foundational works" in result.lower()


def test_format_foundations_prompt_with_questions_and_titles():
    from app.agents.field_foundations_agent import format_foundations_prompt

    result = format_foundations_prompt(
        topic="Educational Technology",
        research_questions=[
            "How does gamification affect learning?",
            "What role does AI play in personalized learning?",
        ],
        existing_titles=[
            "Situated Cognition and the Culture of Learning",
            "Constructivism and Learning",
        ],
    )
    assert "Educational Technology" in result
    assert "gamification" in result
    assert "AI" in result
    assert "Situated Cognition" in result
    assert "do NOT duplicate" in result


def test_format_foundations_prompt_no_questions_no_titles():
    from app.agents.field_foundations_agent import format_foundations_prompt

    result = format_foundations_prompt(
        topic="Applied Linguistics",
        research_questions=None,
        existing_titles=None,
    )
    assert "Applied Linguistics" in result
    # Should not contain "Research questions" or "already in" sections
    assert "Research questions:" not in result
    assert "already in" not in result

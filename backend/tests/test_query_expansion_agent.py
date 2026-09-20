"""Tests for the Query Expansion Agent."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")


def test_expansion_prompt_forbids_wildcards_and_names_openalex():
    from app.agents.query_expansion_agent import EXPANSION_PROMPT

    text = EXPANSION_PROMPT.lower()
    assert "openalex" in text
    assert "wildcard" in text
    assert "truncation" in text


def test_query_expansion_agent_exists():
    from app.agents.query_expansion_agent import get_query_expansion_agent

    agent = get_query_expansion_agent()
    assert agent is not None


def test_format_expansion_prompt_basic():
    from app.agents.query_expansion_agent import format_expansion_prompt

    result = format_expansion_prompt(
        original_query="machine learning in education",
        paper_titles=["Deep Learning for Student Assessment", "AI-Powered Tutoring Systems"],
    )
    assert "machine learning in education" in result
    assert "Deep Learning for Student Assessment" in result
    assert "AI-Powered Tutoring Systems" in result


def test_format_expansion_prompt_with_abstracts():
    from app.agents.query_expansion_agent import format_expansion_prompt

    result = format_expansion_prompt(
        original_query="EdTech in higher education",
        paper_titles=["Paper A", "Paper B"],
        paper_abstracts=["This study examines the impact of technology", None],
    )
    assert "EdTech in higher education" in result
    assert "Paper A" in result
    assert "Paper B" in result
    assert "This study examines the impact of technology" in result


def test_format_expansion_prompt_truncates_to_ten():
    from app.agents.query_expansion_agent import format_expansion_prompt

    titles = [f"Paper {i}" for i in range(15)]
    result = format_expansion_prompt(
        original_query="test query",
        paper_titles=titles,
    )
    assert "Paper 9" in result
    assert "Paper 10" not in result

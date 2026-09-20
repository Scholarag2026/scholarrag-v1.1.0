"""D6: the three declarations of 'required sections' must not contradict each other."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

RESEARCH_ARTICLE_SECTIONS = [
    "abstract",
    "introduction",
    "literature review",
    "methods",
    "results",
    "discussion",
    "implications",
    "conclusion",
]

LITERATURE_REVIEW_SECTIONS = [
    "abstract",
    "introduction",
    "literature review",
    "discussion",
    "conclusion",
]


def _sections_from_prompt_line(text: str, marker: str) -> list[str]:
    """Extract the comma-separated section list that follows *marker* in *text*."""
    line = next(line for line in text.splitlines() if marker in line)
    tail = line.split(marker, 1)[1]
    tail = tail.rstrip().rstrip(".")
    return [part.strip() for part in tail.split(",") if part.strip()]


def test_merge_prompt_research_article_includes_implications():
    from app.agents.compliance_rules_agent import MERGE_PROMPT

    sections = _sections_from_prompt_line(MERGE_PROMPT, "A research article needs:")
    assert sections == RESEARCH_ARTICLE_SECTIONS


def test_merge_prompt_literature_review_matches_golden_standard():
    from app.agents.compliance_rules_agent import MERGE_PROMPT

    sections = _sections_from_prompt_line(MERGE_PROMPT, "A literature review article needs:")
    assert sections == LITERATURE_REVIEW_SECTIONS


def test_golden_standards_text_matches_merge_prompt():
    from app.agents.compliance_rules_agent import GOLDEN_STANDARDS_TEXT

    research = _sections_from_prompt_line(
        GOLDEN_STANDARDS_TEXT["research_article"], "- Required sections:"
    )
    review = _sections_from_prompt_line(
        GOLDEN_STANDARDS_TEXT["literature_review"], "- Required sections:"
    )
    assert research == RESEARCH_ARTICLE_SECTIONS
    assert review == LITERATURE_REVIEW_SECTIONS


def test_compliance_fallback_matches_merge_prompt():
    from app.services.compliance import GOLDEN_STANDARDS

    assert GOLDEN_STANDARDS["research_article"]["required_sections"] == RESEARCH_ARTICLE_SECTIONS
    assert GOLDEN_STANDARDS["literature_review"]["required_sections"] == LITERATURE_REVIEW_SECTIONS

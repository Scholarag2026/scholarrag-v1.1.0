"""Tests for the deterministic coverage/meta-evaluation flagging
``app.agents.citation_link_agent._flag_sentence_coverage`` adds to every validated
citation link.

No model call, no database: pure-function tests over ``validate_citation_links``.
"""
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.agents.citation_link_agent import (  # noqa: E402
    CitationLink,
    CitationLinkMap,
    validate_citation_links,
)


def _norm(text: str) -> str:
    return " ".join(text.split())


def test_a_sentence_with_uncovered_residue_is_flagged_incomplete():
    sentence = (
        "Liu and Wu (2019) add that 60% of students preferred teacher feedback, "
        "underscoring the persistent value of human judgement."
    )
    content = sentence
    link_map = CitationLinkMap(
        links=[
            CitationLink(
                paragraph_index=0,
                sentence=sentence,
                citation_text="(2019)",
                keys=["liu_2019"],
                proposition="60% of students preferred teacher feedback",
            )
        ]
    )
    validated = validate_citation_links(link_map, content)
    assert len(validated) == 1
    assert validated[0]["coverage_incomplete"] is True
    assert any("underscoring" in span for span in validated[0]["residue_spans"])
    assert validated[0]["meta_evaluation"] is False


def test_a_fully_covered_sentence_is_not_flagged():
    sentence = "Smith (2020) found that recall improved by 12 points."
    link_map = CitationLinkMap(
        links=[
            CitationLink(
                paragraph_index=0,
                sentence=sentence,
                citation_text="(2020)",
                keys=["smith_2020"],
                proposition="recall improved by 12 points",
            )
        ]
    )
    validated = validate_citation_links(link_map, sentence)
    assert validated[0]["coverage_incomplete"] is False
    assert validated[0]["residue_spans"] == []


def test_two_links_on_the_same_sentence_are_flagged_together():
    """The dedup-defect shape (design rows 28-30): a sentence with two propositions
    against the same key. Each proposition alone leaves the other uncovered, so both
    links of the pair must see the union, not be checked in isolation."""
    sentence = (
        "Bonilla Lopez et al. (2018) reported that learners' cognitive-load estimates "
        "were significantly lower when processing direct corrections, while self-"
        "correcting with no feedback available imposed significantly lower load."
    )
    link_map = CitationLinkMap(
        links=[
            CitationLink(
                paragraph_index=0,
                sentence=sentence,
                citation_text="(2018)",
                keys=["lopez_2018"],
                proposition=(
                    "learners' cognitive-load estimates were significantly lower when "
                    "processing direct corrections"
                ),
            ),
            CitationLink(
                paragraph_index=0,
                sentence=sentence,
                citation_text="(2018)",
                keys=["lopez_2018"],
                proposition=(
                    "self-correcting with no feedback available imposed significantly "
                    "lower load"
                ),
            ),
        ]
    )
    validated = validate_citation_links(link_map, sentence)
    assert len(validated) == 2
    # Together the two propositions cover the sentence (bar attribution/connector
    # frames), so neither link is flagged incomplete.
    assert all(v["coverage_incomplete"] is False for v in validated)


def test_comparative_meta_evaluation_is_flagged_on_the_proposition():
    sentence = "Revision behaviour is treated most directly by Yallop et al. (2021)."
    link_map = CitationLinkMap(
        links=[
            CitationLink(
                paragraph_index=0,
                sentence=sentence,
                citation_text="(2021)",
                keys=["yallop_2021"],
                proposition=sentence,
            )
        ]
    )
    validated = validate_citation_links(link_map, sentence)
    assert validated[0]["meta_evaluation"] is True

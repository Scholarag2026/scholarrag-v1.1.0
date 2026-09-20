"""Pure-function helpers shared by the deep-analysis job and the write job's own
grounding step: no network, no database.
"""

from app.services.deep_analysis import (
    format_authors,
    full_text_from_chunks,
    is_full_text_analysis,
)


def test_full_text_from_chunks_joins_text_fields_in_order():
    chunks = [
        {"section": "intro", "text": "First part."},
        {"section": "results", "text": "Second part."},
    ]
    assert full_text_from_chunks(chunks) == "First part.\n\nSecond part."


def test_full_text_from_chunks_skips_entries_without_text():
    chunks = [{"section": "intro"}, {"text": "Kept."}, "not a dict"]
    assert full_text_from_chunks(chunks) == "Kept."


def test_full_text_from_chunks_tolerates_none():
    assert full_text_from_chunks(None) == ""


def test_format_authors_joins_dicts_by_name():
    assert format_authors([{"name": "Jane Smith"}, {"name": "Amy Brown"}]) == (
        "Jane Smith, Amy Brown"
    )


def test_format_authors_joins_plain_strings():
    assert format_authors(["Jane Smith", "Amy Brown"]) == "Jane Smith, Amy Brown"


def test_format_authors_tolerates_a_mix():
    assert format_authors(["Jane Smith", {"name": "Amy Brown"}]) == "Jane Smith, Amy Brown"


def test_format_authors_tolerates_non_list():
    assert format_authors(None) == ""
    assert format_authors("Solo Author") == "Solo Author"


def test_is_full_text_analysis_false_for_none_and_empty():
    assert is_full_text_analysis(None) is False
    assert is_full_text_analysis({}) is False


def test_is_full_text_analysis_false_for_the_abstract_only_placeholder():
    from app.agents.deep_analysis_agent import build_abstract_only_analysis

    placeholder = build_abstract_only_analysis("T", "A", 2020, "An abstract.")
    assert is_full_text_analysis(placeholder) is False


def test_is_full_text_analysis_true_when_any_substantive_field_is_populated():
    assert is_full_text_analysis({"key_findings": "x", "methodology": "A survey."}) is True
    assert is_full_text_analysis({"key_findings": "x", "themes": ["a theme"]}) is True

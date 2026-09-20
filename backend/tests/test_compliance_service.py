import pytest

from app.services.compliance import (
    _check_abstract_length,
    _check_sections_present,
    _check_word_count,
    _count_words_in_tiptap,
    _extract_sections_from_tiptap,
    check_compliance,
)


def test_count_words_empty():
    assert _count_words_in_tiptap(None) == 0
    assert _count_words_in_tiptap({}) == 0


def test_count_words_simple():
    content = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Hello world this is a test"}]},
        ],
    }
    assert _count_words_in_tiptap(content) == 6


def test_count_words_with_headings():
    content = {
        "type": "doc",
        "content": [
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Introduction"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "This is the introduction text."}]},
        ],
    }
    assert _count_words_in_tiptap(content) == 6  # heading words + paragraph words


def test_extract_sections():
    content = {
        "type": "doc",
        "content": [
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Introduction"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Some text."}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Literature Review"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "More text."}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Methods"}]},
        ],
    }
    sections = _extract_sections_from_tiptap(content)
    assert "introduction" in sections
    assert "literature review" in sections
    assert "methods" in sections


def test_check_abstract_length_good():
    standards = {"word_limit_abstract_min": 150, "word_limit_abstract_max": 300}
    result = _check_abstract_length(250, standards)
    assert result.status == "pass"


def test_check_abstract_length_too_short():
    standards = {"word_limit_abstract_min": 150, "word_limit_abstract_max": 300}
    result = _check_abstract_length(100, standards)
    assert result.status == "warning"


def test_check_abstract_length_too_long():
    standards = {"word_limit_abstract_min": 150, "word_limit_abstract_max": 300}
    result = _check_abstract_length(350, standards)
    assert result.status == "warning"


def test_check_abstract_length_borderline():
    standards = {"word_limit_abstract_min": 150, "word_limit_abstract_max": 300}
    result = _check_abstract_length(225, standards)
    assert result.status == "pass"


def test_check_abstract_length_zero():
    standards = {"word_limit_abstract_min": 150, "word_limit_abstract_max": 300}
    result = _check_abstract_length(0, standards)
    assert result.status == "fail"


def test_check_sections_all_present():
    sections = ["introduction", "literature review", "methods", "results", "discussion", "conclusion"]
    standards = {"required_sections": ["introduction", "literature review", "methods", "results", "discussion", "conclusion"]}
    result = _check_sections_present(sections, standards)
    assert result.status == "pass"


def test_check_sections_missing():
    sections = ["introduction", "literature review"]
    standards = {"required_sections": ["introduction", "literature review", "methods", "results", "discussion", "conclusion"]}
    result = _check_sections_present(sections, standards)
    assert result.status == "fail" or result.status == "warning"


def test_check_word_count_good():
    standards = {"word_limit_min": 5000, "word_limit_max": 8000}
    result = _check_word_count(6000, standards)
    assert result.status == "pass"


def test_check_word_count_too_short():
    standards = {"word_limit_min": 5000, "word_limit_max": 8000}
    result = _check_word_count(3000, standards)
    assert result.status == "warning"


@pytest.mark.asyncio
async def test_full_compliance_check():
    content = {
        "type": "doc",
        "content": [
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Introduction"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " ".join(["word"] * 500)}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Literature Review"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " ".join(["word"] * 1000)}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Methods"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " ".join(["word"] * 500)}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Results"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " ".join(["word"] * 500)}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Discussion"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " ".join(["word"] * 500)}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Conclusion"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " ".join(["word"] * 200)}]},
        ],
    }
    report = await check_compliance(content, "research_article")
    assert report.total_word_count > 3000
    assert len(report.checks) >= 3
    assert len(report.sections_found) >= 5

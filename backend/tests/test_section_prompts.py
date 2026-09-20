from app.services.writing import SECTION_PROMPTS, get_section_prompt


def test_literature_review_prompt_exists():
    prompt = get_section_prompt("literature_review")
    assert prompt is not None
    assert "synthesize" in prompt.lower() or "SYNTHESIZE" in prompt


def test_introduction_prompt_exists():
    prompt = get_section_prompt("introduction")
    assert prompt is not None
    assert "purpose" in prompt.lower() or "gap" in prompt.lower() or "GAP" in prompt


def test_methods_prompt_exists():
    prompt = get_section_prompt("methods")
    assert prompt is not None
    assert "methodology" in prompt.lower() or "justify" in prompt.lower()


def test_results_prompt_exists():
    prompt = get_section_prompt("results")
    assert prompt is not None
    assert "past tense" in prompt.lower()


def test_discussion_prompt_exists():
    prompt = get_section_prompt("discussion")
    assert prompt is not None
    assert "interpret" in prompt.lower() or "theory" in prompt.lower()


def test_implications_prompt_exists():
    prompt = get_section_prompt("implications")
    assert prompt is not None
    assert "theoretical" in prompt.lower() or "practical" in prompt.lower()


def test_conclusion_prompt_exists():
    prompt = get_section_prompt("conclusion")
    assert prompt is not None
    assert "summary" in prompt.lower() or "limitation" in prompt.lower()


def test_abstract_prompt_exists():
    prompt = get_section_prompt("abstract")
    assert prompt is not None
    assert "250" in prompt or "word" in prompt.lower()


def test_unknown_section_returns_default():
    prompt = get_section_prompt("unknown_section")
    assert prompt is not None  # Should return a generic academic writing prompt


def test_all_section_types_registered():
    expected = {
        "introduction", "literature_review", "methods",
        "results", "discussion", "implications",
        "conclusion", "abstract",
    }
    assert expected.issubset(set(SECTION_PROMPTS.keys()))

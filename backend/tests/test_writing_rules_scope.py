"""The writer's base rules no longer force a citation onto every
sentence or instruct the model to spread citations across papers. Measurement traced
22 of 40 claims' unsupported status to exactly those two rules:
the old "EVERY factual claim MUST include an in-text citation" line and the
"CITATION DISTRIBUTION RULES" block that told the model to distribute citations broadly
across papers rather than to whichever paper's material actually supports a clause.
"""

from app.services.writing import _BASE_RULES, get_section_prompt


def test_the_old_citation_distribution_block_is_gone():
    assert "CITATION DISTRIBUTION RULES" not in _BASE_RULES
    assert "at most 2-3 times" not in _BASE_RULES
    assert "Distribute citations broadly" not in _BASE_RULES
    assert "Never cite the same paper in consecutive sentences" not in _BASE_RULES
    assert "Only foundational/seminal works" not in _BASE_RULES


def test_the_old_every_claim_must_cite_line_is_gone():
    assert "EVERY factual claim MUST include an in-text citation" not in _BASE_RULES


def test_the_new_scoped_citation_rules_are_present():
    # Cite a paper only for a proposition stated in the material supplied for it.
    assert "only for a proposition" in _BASE_RULES
    assert "material supplied for that paper" in _BASE_RULES
    # A multi-citation sentence: each cited paper supports its own clause; a
    # field-wide or other-papers claim is never attributed to one paper.
    assert "each cited paper must support the clause it is attached to" in _BASE_RULES
    assert "never attributed to a paper" in _BASE_RULES
    # An unsupported statement is written without a citation and flagged.
    assert "written without a citation and flagged [NEEDS CITATION]" in _BASE_RULES


def test_only_cite_papers_provided_and_author_year_format_are_kept():
    assert "ONLY cite papers provided in the context below" in _BASE_RULES
    assert "NEVER invent citations" in _BASE_RULES
    assert "(Author, Year) or Author (Year)" in _BASE_RULES
    assert "APA 7th" in _BASE_RULES


def test_every_section_prompt_inherits_the_new_rules():
    for section_type in (
        "literature_review", "introduction", "methods", "results",
        "discussion", "implications", "conclusion", "abstract",
    ):
        prompt = get_section_prompt(section_type)
        assert "CITATION DISTRIBUTION RULES" not in prompt
        assert "EVERY factual claim MUST include an in-text citation" not in prompt
        assert "only for a proposition" in prompt

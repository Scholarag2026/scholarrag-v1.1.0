"""Tests for demo/check_delivered.py.

Two kinds of test: a fixture-driven unit test for each of the seven rules (hand-built,
minimal data, no I/O beyond ``tmp_path``), and a historical-regression test that runs
the checker over every tracked run directory under ``demo/output/`` and pins down
exactly which of the known, already-documented defects it must still catch (and which
directories it must report as unable to check certain rules, for a file that
directory never wrote).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import skip_unless_run_dir

from check_delivered import (
    ABBREVIATIONS_NOT_SENTENCE_FINAL,
    RULE_CLAIM_TEXT_NOT_IN_DRAFT,
    RULE_COMPARATIVE_META_EVALUATION,
    RULE_DANGLING_FRAMING_SENTENCE,
    RULE_DUPLICATE_HEADING,
    RULE_EMPTY_CITED_SECTION,
    RULE_ENUMERATION_OPENER_SHORT_OF_COUNT,
    RULE_EVIDENCE_NOT_VERBATIM,
    RULE_FIXTURE_UNSUPPORTED_PRESENT,
    RULE_FIXTURE_VERIFIED_MISSING,
    RULE_HEADING_SHAPED_PARAGRAPH,
    RULE_HEADING_WITH_NO_BODY,
    RULE_NON_TITLE_HEADING_IN_SECTION,
    RULE_TITLE_ECHO_IN_BODY,
    RULE_NEEDS_CITATION_MARKER,
    RULE_PARAGRAPH_HAS_NO_CITATION,
    RULE_PARTICIPLE_AFTER_ATTRIBUTION,
    RULE_PASSAGE_NOT_LOCATED,
    RULE_ROW_NOT_VERIFIED,
    RULE_SENTENCE_EXCEEDS_VERIFIED_CLAIM,
    RULE_TRUNCATED_SENTENCE,
    RULE_UNCITED_FINDING_SURVIVED,
    RULE_UNCLASSIFIED_SENTENCE,
    RULE_UNRESOLVED_CITATION,
    RULE_VERB_INITIAL_SENTENCE,
    CheckResult,
    SectionFiles,
    _is_frame,
    _Residue,
    check_body_sentence_classification,
    check_citation_coverage_resolved,
    check_comparative_meta_evaluation,
    check_dangling_framing,
    check_delivered_evidence_rows,
    check_duplicate_leading_heading,
    check_empty_cited_section,
    check_final_report_rows,
    check_fixture_claims,
    check_heading_has_no_body,
    check_heading_shaped_paragraph,
    check_needs_citation_marker,
    check_non_title_heading,
    check_paragraph_has_citation,
    check_participle_after_attribution,
    check_run_directory,
    check_section,
    check_sentence_covered_by_verified_claims,
    check_title_echo_in_body,
    check_truncated_sentences,
    check_verb_initial_sentence,
    discover_extra_section_indices,
    draft_heading_texts,
    draft_paragraph_texts,
    is_heading_shaped_paragraph,
    is_title_echo,
    is_verbatim_under_guard_fold,
    load_section_files,
    main,
    normalize_sentence,
    split_sentences,
    truncation_reasons,
)

DEMO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers to build a minimal Tiptap draft document
# ---------------------------------------------------------------------------


def _heading(text: str) -> dict:
    return {"type": "heading", "attrs": {"level": 2}, "content": [{"type": "text", "text": text}]}


def _paragraph(text: str) -> dict:
    return {"type": "paragraph", "content": [{"type": "text", "text": text}]}


def _draft(*nodes: dict) -> dict:
    return {"type": "doc", "content": list(nodes)}


def _uncited(sentence: str, tag: str, paragraph_index: int = 0) -> dict:
    return {"sentence": sentence, "tag": tag, "paragraph_index": paragraph_index}


def _final_row(
    claim_text: str,
    *,
    status: str = "verified",
    claim_sentence: str | None = None,
    evidence_quote: str | None = None,
    evidence_quotes: list[str] | None = None,
) -> dict:
    row = {"claim_text": claim_text, "status": status}
    if claim_sentence is not None:
        row["claim_sentence"] = claim_sentence
    if evidence_quote is not None:
        row["evidence_quote"] = evidence_quote
    if evidence_quotes is not None:
        row["evidence_quotes"] = evidence_quotes
    return row


def _claim_report(rows: list[dict], *, citation_coverage: dict | None = "MISSING") -> dict:
    report: dict = {"final_report": {"verifications": rows, "verified_count": len(rows)}}
    if citation_coverage != "MISSING":
        report["citation_coverage"] = citation_coverage
    return report


# ---------------------------------------------------------------------------
# sentence-level helpers
# ---------------------------------------------------------------------------


def test_normalize_sentence_strips_needs_citation_marker_and_collapses_whitespace():
    assert normalize_sentence("A claim  [NEEDS CITATION]   here.") == "A claim here."
    assert normalize_sentence(None) == ""


def test_split_sentences_splits_on_terminal_punctuation():
    assert split_sentences("First one. Second one! Third one?") == [
        "First one.",
        "Second one!",
        "Third one?",
    ]


def test_split_sentences_does_not_split_a_narrative_citations_own_year():
    text = "Smith et al. (2020) found an effect. A new sentence follows."
    assert split_sentences(text) == [
        "Smith et al. (2020) found an effect.",
        "A new sentence follows.",
    ]


def test_split_sentences_does_not_split_at_the_vs_abbreviation():
    text = "ChatGPT beat Grammarly (94-98% vs. 85%), but recall fell. Next sentence."
    sentences = split_sentences(text)
    assert sentences == [
        "ChatGPT beat Grammarly (94-98% vs. 85%), but recall fell.",
        "Next sentence.",
    ]


def test_abbreviations_not_sentence_final_carries_vs():
    assert "vs" in ABBREVIATIONS_NOT_SENTENCE_FINAL


def test_truncation_reasons_flags_a_sentence_ending_at_the_vs_abbreviation():
    reasons = truncation_reasons("Grammarly's precision was higher (94-98% vs.")
    assert any("abbreviation" in r for r in reasons)


def test_truncation_reasons_flags_a_lowercase_leading_fragment():
    reasons = truncation_reasons("85%), but recall rose only from 10% to 55%.")
    assert any("lower-case" in r for r in reasons)


def test_truncation_reasons_does_not_flag_a_percentage_led_sentence():
    """A real false positive: the sentence's own first real
    word after its leading percentage ("of") happens to be lower-case, but the
    sentence is a complete, grammatical statement, not a fragment cut from
    anything -- unlike "85%), but recall rose ...", it carries no stray, unmatched
    closing bracket before its own first letter."""
    sentence = (
        "88% of 33 students submitted to the automated system more than five "
        "times, and students reported spending more time revising after "
        "automated feedback than after peer or teacher feedback (Zhang & "
        "Hyland, 2021)."
    )
    assert truncation_reasons(sentence) == []


def test_truncation_reasons_does_not_flag_a_bare_digit_led_sentence():
    assert truncation_reasons("3 of the 5 studies used a mixed-methods design.") == []


def test_truncation_reasons_does_not_flag_a_parenthesised_citation_opener():
    assert truncation_reasons(
        "(as reported by Zhang, 2021) revision behaviour varied widely."
    ) == []


def test_truncation_reasons_does_not_flag_a_quotation_mark_opener():
    assert truncation_reasons(
        '"generalisation beyond this sample is limited," the authors note.'
    ) == []


def test_truncation_reasons_still_flags_a_percentage_fragment_with_a_stray_closer():
    """The exact shape `_begins_with_lowercase_fragment_reason` was written for:
    a stray, unmatched closing bracket right after the leading percentage is the
    tell of a fragment cut from inside a parenthetical, not a percentage opening a
    fresh sentence -- the percentage exemption never applies here."""
    reasons = truncation_reasons("94-98% vs. 85%), but recall rose only from 10% to 55%.")
    assert any("lower-case" in r for r in reasons)


def test_truncation_reasons_empty_for_a_clean_sentence():
    assert truncation_reasons("Smith (2020) found a large effect.") == []


def test_truncation_reasons_check_lowercase_start_false_suppresses_that_signal():
    reasons = truncation_reasons(
        "found that most students preferred feedback.", check_lowercase_start=False
    )
    assert reasons == []


def test_truncation_reasons_still_flags_abbreviation_when_lowercase_check_is_off():
    reasons = truncation_reasons(
        "exceeded Grammarly's precision (94-98% vs.", check_lowercase_start=False
    )
    assert any("abbreviation" in r for r in reasons)


# ---------------------------------------------------------------------------
# draft (Tiptap) helpers
# ---------------------------------------------------------------------------


def test_draft_paragraph_texts_reads_paragraph_nodes_in_order():
    draft = _draft(_heading("Title"), _paragraph("One."), _paragraph("Two."))
    assert draft_paragraph_texts(draft) == ["One.", "Two."]


def test_draft_heading_texts_reads_heading_nodes_in_order():
    draft = _draft(_heading("Title"), _heading("Title"), _paragraph("Body."))
    assert draft_heading_texts(draft) == ["Title", "Title"]


def test_draft_paragraph_texts_of_none_is_empty():
    assert draft_paragraph_texts(None) == []


# ---------------------------------------------------------------------------
# Rule 3: no duplicate leading heading
# ---------------------------------------------------------------------------


def test_check_duplicate_leading_heading_flags_two_identical_adjacent_headings():
    draft = _draft(
        _heading("Literature Review"), _heading("Literature Review"), _paragraph("Body.")
    )
    violations = check_duplicate_leading_heading(draft, "protocol section")
    assert len(violations) == 1
    assert violations[0].rule == RULE_DUPLICATE_HEADING


def test_check_duplicate_leading_heading_ignores_a_single_heading():
    draft = _draft(_heading("Literature Review"), _paragraph("Body."))
    assert check_duplicate_leading_heading(draft, "s") == []


def test_check_duplicate_leading_heading_flags_two_different_headings_at_the_start():
    """The section's own raw first two content nodes both
    being heading nodes is always wrong, whatever their wording -- this is the shape
    a run's own draft_content_N.json can carry (H2 then a differently worded H1),
    which a text-equality-only rule alone would let through unflagged."""
    draft = _draft(_heading("Literature Review"), _heading("Sub-heading"), _paragraph("Body."))
    violations = check_duplicate_leading_heading(draft, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_DUPLICATE_HEADING


def test_check_duplicate_leading_heading_ignores_two_different_headings_not_at_the_start():
    """A legitimate sub-heading later in the body, not immediately following the
    section's own opening heading, is not a duplicate of it."""
    draft = _draft(
        _heading("Literature Review"),
        _paragraph("Framing prose."),
        _heading("Sub-heading"),
        _paragraph("Body."),
    )
    assert check_duplicate_leading_heading(draft, "s") == []


def test_check_duplicate_leading_heading_is_case_and_whitespace_insensitive():
    draft = _draft(_heading("Literature   Review"), _heading("literature review"), _paragraph("B."))
    assert len(check_duplicate_leading_heading(draft, "s")) == 1


# ---------------------------------------------------------------------------
# Rule 2: no [NEEDS CITATION] marker, no unresolved citation
# ---------------------------------------------------------------------------


def test_check_needs_citation_marker_flags_a_surviving_marker():
    draft = _draft(_paragraph("A claim with no source [NEEDS CITATION]."))
    violations = check_needs_citation_marker(draft, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_NEEDS_CITATION_MARKER


def test_check_needs_citation_marker_passes_a_clean_draft():
    draft = _draft(_paragraph("A fully cited claim (Smith, 2020)."))
    assert check_needs_citation_marker(draft, "s") == []


def test_check_citation_coverage_resolved_skips_when_coverage_is_none():
    assert check_citation_coverage_resolved({"citation_coverage": None}, "s") == []
    assert check_citation_coverage_resolved(None, "s") == []


def test_check_citation_coverage_resolved_passes_when_unresolved_is_zero():
    report = {"citation_coverage": {"unresolved": 0}}
    assert check_citation_coverage_resolved(report, "s") == []


def test_check_citation_coverage_resolved_flags_a_nonzero_unresolved_count():
    report = {"citation_coverage": {"unresolved": 2}}
    violations = check_citation_coverage_resolved(report, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_UNRESOLVED_CITATION


# ---------------------------------------------------------------------------
# Rule 4: every final report row verified and verbatim in the draft
# ---------------------------------------------------------------------------


def test_check_final_report_rows_passes_a_verified_row_found_in_the_draft():
    draft = _draft(_paragraph("Smith (2020) found a large effect."))
    report = _claim_report([_final_row("found a large effect")])
    assert check_final_report_rows(draft, report, "s") == []


def test_check_final_report_rows_flags_a_row_that_is_not_verified():
    draft = _draft(_paragraph("Smith (2020) found a large effect."))
    report = _claim_report([_final_row("found a large effect", status="needs_nuance")])
    violations = check_final_report_rows(draft, report, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_ROW_NOT_VERIFIED


def test_check_final_report_rows_flags_claim_text_missing_from_the_draft():
    draft = _draft(_paragraph("Completely unrelated paragraph text."))
    report = _claim_report([_final_row("found a large effect")])
    violations = check_final_report_rows(draft, report, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_CLAIM_TEXT_NOT_IN_DRAFT


def test_check_final_report_rows_returns_empty_when_final_report_is_absent():
    draft = _draft(_paragraph("Anything."))
    assert check_final_report_rows(draft, {"verifications": []}, "s") == []


# ---------------------------------------------------------------------------
# New rule: empty_cited_section
# ---------------------------------------------------------------------------


def test_check_empty_cited_section_flags_a_hollow_section_with_a_real_citation_audit():
    """Real shape: the writer's own text
    carried three real citations, but the final report has no verified row at all."""
    writing_result = {"citation_audit": {"total": 3}, "loop_stats": {}}
    report = _claim_report([])
    violations = check_empty_cited_section(writing_result, report, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_EMPTY_CITED_SECTION


def test_check_empty_cited_section_passes_a_section_with_no_citations_at_all():
    """An honest section that had nothing to cite is not a violation -- the writer's
    own citation audit found nothing, so an empty final report is not a loss."""
    writing_result = {"citation_audit": {"total": 0}, "loop_stats": {}}
    report = _claim_report([])
    assert check_empty_cited_section(writing_result, report, "s") == []


def test_check_empty_cited_section_passes_a_section_with_a_verified_row():
    writing_result = {"citation_audit": {"total": 1}, "loop_stats": {}}
    report = _claim_report([_final_row("found a large effect")])
    assert check_empty_cited_section(writing_result, report, "s") == []


def test_check_empty_cited_section_flags_the_regeneration_backstop_flag():
    """The empty-section regeneration backstop fired and even its own second attempt
    finalized to zero cited sentences -- a violation regardless of what the final
    report itself carries (there may be none at all, on a build too old to carry
    one)."""
    writing_result = {
        "citation_audit": {"total": 2},
        "loop_stats": {"empty_section_after_regeneration": True},
    }
    violations = check_empty_cited_section(writing_result, None, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_EMPTY_CITED_SECTION


def test_check_empty_cited_section_is_silent_when_final_report_is_absent_entirely():
    """``verifications`` being ``None`` (no ``final_report`` on this build at all,
    and no regeneration flag either) is not itself a violation of this rule -- that
    gap is already reported elsewhere as a missing-file notice."""
    writing_result = {"citation_audit": {"total": 3}, "loop_stats": {}}
    assert check_empty_cited_section(writing_result, {}, "s") == []


# ---------------------------------------------------------------------------
# New rule: title_echo_in_body
# ---------------------------------------------------------------------------


def test_is_title_echo_matches_the_shared_fixture_file():
    """``demo/fixtures/title_echo_cases.json`` is the single committed fixture both
    this module's own copy and `app.services.fulltext`'s copy read (`backend/tests/
    test_title_echo_fixture.py` checks the app-side copy), so the two cannot diverge
    silently."""
    fixture_path = DEMO / "fixtures" / "title_echo_cases.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert cases, "expected the shared title_echo_cases.json fixture to be non-empty"
    for case in cases:
        assert is_title_echo(case["paragraph"], case["references"]) == case["expected"], case


def test_check_title_echo_in_body_flags_a_demoted_heading_restating_the_section_title():
    draft = _draft(
        _heading("Metalinguistic codes as written corrective feedback"),
        _paragraph("Metalinguistic Coded Feedback in L2 Writing"),
    )
    violations = check_title_echo_in_body(draft, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_TITLE_ECHO_IN_BODY


def test_check_title_echo_in_body_passes_a_genuine_body_paragraph():
    draft = _draft(
        _heading("Learner engagement with written corrective feedback"),
        _paragraph(
            "Research on this topic has moved from whether feedback works to how "
            "learners engage with it, and Mao et al. (2024) synthesised 50 studies."
        ),
    )
    assert check_title_echo_in_body(draft, "s") == []


def test_check_title_echo_in_body_ignores_a_markdown_heading_shaped_paragraph_node():
    """An older saved run's own restated markdown title, stored as a plain
    "paragraph" node from before this product promoted that shape to a real heading
    node, already reads as a heading to a reader -- `_paragraph_is_heading_shaped`
    excludes it here exactly as rule 8a's own `_paragraph_descriptors` already does."""
    draft = _draft(
        _heading("Effects of written corrective feedback"),
        _paragraph("## Effects of Written Corrective Feedback on L2 Writing Accuracy"),
    )
    assert check_title_echo_in_body(draft, "s") == []


def test_check_title_echo_in_body_returns_empty_with_no_headings_at_all():
    draft = _draft(_paragraph("Metalinguistic Coded Feedback in L2 Writing"))
    assert check_title_echo_in_body(draft, "s") == []


# ---------------------------------------------------------------------------
# New rule: heading_shaped_paragraph
# ---------------------------------------------------------------------------


def test_is_heading_shaped_paragraph_matches_the_shared_fixture_file():
    """``demo/fixtures/heading_shaped_paragraphs.json`` is the single committed
    fixture both this module's own copy and `app.services.sentence_coverage`'s copy
    read (`backend/tests/test_heading_shaped_paragraph_fixture.py` checks the
    app-side copy), so the two cannot diverge silently."""
    fixture_path = DEMO / "fixtures" / "heading_shaped_paragraphs.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert cases, "expected the shared heading_shaped_paragraphs.json fixture to be non-empty"
    for case in cases:
        assert is_heading_shaped_paragraph(case["text"]) == case["expected"], case


def test_check_heading_shaped_paragraph_flags_the_bare_sub_heading_fragment():
    """The exact defect an independent audit found: unlike rule 15
    (`title_echo_in_body`), this needs no heading or title to compare against at
    all -- the fragment is caught by its own shape."""
    draft = _draft(
        _heading("Automated written corrective feedback"),
        _paragraph("Comparative Scope and Accuracy"),
        _paragraph("Zhang and Hyland (2018) found substantial variation (Zhang, 2018)."),
        _heading("Revision Behaviour and Engagement"),
    )
    violations = check_heading_shaped_paragraph(draft, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_HEADING_SHAPED_PARAGRAPH
    assert violations[0].sentence == "Comparative Scope and Accuracy"


def test_check_heading_shaped_paragraph_does_not_flag_ordinary_short_sentences():
    draft = _draft(
        _heading("s"),
        _paragraph("Uptake was uneven."),
        _paragraph("Findings diverge."),
        _paragraph("In contrast, results differed (Mao et al., 2024)."),
    )
    assert check_heading_shaped_paragraph(draft, "s") == []


def test_check_heading_shaped_paragraph_does_not_flag_the_protocol_fixtures():
    draft = _draft(
        _heading("s"),
        _paragraph(
            "A systematic search for the review identified 50 empirical studies "
            "meeting its inclusion criteria, revealing four major themes: teacher "
            "feedback practices in L2 writing classrooms, L2 learner responses to "
            "feedback, stakeholders' beliefs and perspectives about feedback, and "
            "feedback-related motivation and emotions (Mao et al., 2024)."
        ),
        _paragraph(
            "Screening yielded the same 50 eligible studies for the review, "
            "sorted into two major themes: writing accuracy outcomes and learner "
            "motivation only (Mao et al., 2024)."
        ),
    )
    assert check_heading_shaped_paragraph(draft, "s") == []


def test_check_heading_shaped_paragraph_fires_at_any_position_not_only_the_first():
    """Rule 15 only ever looks at a paragraph against a title or a standing
    heading; this rule needs neither, so a heading-shaped fragment delivered
    anywhere in the section -- not only as its first body paragraph -- is still a
    violation."""
    draft = _draft(
        _heading("s"),
        _paragraph("Zhang and Hyland (2018) found substantial variation (Zhang, 2018)."),
        _paragraph("Comparative Scope and Accuracy"),
    )
    violations = check_heading_shaped_paragraph(draft, "s")
    assert len(violations) == 1
    assert violations[0].sentence == "Comparative Scope and Accuracy"


# ---------------------------------------------------------------------------
# Rule 6: no sentence truncated at an abbreviation
# ---------------------------------------------------------------------------


def test_check_truncated_sentences_flags_a_truncated_final_report_claim_sentence():
    draft = _draft(_paragraph("ChatGPT beat Grammarly (94-98% vs. 85%), but recall fell."))
    report = _claim_report(
        [
            _final_row(
                "beat Grammarly's precision (94-98% vs.",
                claim_sentence="ChatGPT beat Grammarly's precision (94-98% vs.",
            )
        ]
    )
    violations = check_truncated_sentences(draft, report, "s")
    assert any(v.rule == RULE_TRUNCATED_SENTENCE for v in violations)


def test_check_truncated_sentences_does_not_flag_a_lowercase_start_in_claim_text_alone():
    """claim_text is routinely a narrower proposition (e.g. "found that ...") and
    legitimately opens lower-case; only claim_sentence is held to that signal."""
    draft = _draft(_paragraph("Smith (2020) found that most students improved."))
    report = _claim_report([_final_row("found that most students improved")])
    assert check_truncated_sentences(draft, report, "s") == []


def test_check_truncated_sentences_flags_a_lowercase_start_in_claim_sentence():
    draft = _draft(_paragraph("orphaned continuation, filler filler filler filler."))
    report = _claim_report(
        [_final_row("filler", claim_sentence="orphaned continuation, filler filler filler filler.")]
    )
    violations = check_truncated_sentences(draft, report, "s")
    assert any("lower-case" in v.detail for v in violations)


def test_check_truncated_sentences_checks_draft_sentences_directly_too():
    draft = _draft(_paragraph("Precision was higher (94-98% vs. 85%), recall fell."))
    # No claim_report data at all -- the draft sentence itself is well formed once
    # split on the abbreviation-aware boundary, so this must report no violation.
    assert check_truncated_sentences(draft, None, "s") == []


def test_check_truncated_sentences_does_not_flag_a_correct_sentence_with_an_internal_abbreviation():
    """A whole, correct sentence containing "U.S." (not in the fixed abbreviation
    list) is split
    into two fragments by this module's own splitter; the second fragment legitimately
    opens lower-case ("intensive English program, ...") and must not be reported."""
    draft = _draft(
        _paragraph(
            "Surveying 70 students and 16 teachers in a U.S. intensive English "
            "program, Liu and Wu (2019) found that most preferred direct correction."
        )
    )
    assert check_truncated_sentences(draft, None, "s") == []


def test_check_truncated_sentences_does_not_double_report_a_shared_final_report_field():
    """``claim_text`` and ``claim_sentence`` are the same string on a row whose
    proposition IS its whole sentence -- reported once, from the stronger
    (``claim_sentence``) check, not twice for the same underlying text."""
    draft = _draft(_paragraph("An unrelated paragraph, present only so the draft is non-empty."))
    same_text = "Text (94-98% vs."
    report = _claim_report([_final_row(same_text, claim_sentence=same_text)])
    violations = check_truncated_sentences(draft, report, "s")
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Rule 1: every body sentence classified
# ---------------------------------------------------------------------------


def test_check_body_sentence_classification_passes_a_framing_sentence():
    draft = _draft(_paragraph("This review covers three themes."))
    writing_result = {
        "uncited_sentences": [_uncited("This review covers three themes.", "framing")]
    }
    violations, notices = check_body_sentence_classification(draft, writing_result, None, "s")
    assert violations == []
    assert notices == []


def test_check_body_sentence_classification_passes_a_sentence_verified_in_the_final_report():
    draft = _draft(_paragraph("Smith (2020) found a large effect on accuracy."))
    writing_result = {"uncited_sentences": []}
    report = _claim_report(
        [
            _final_row(
                "found a large effect",
                claim_sentence="Smith (2020) found a large effect on accuracy.",
            )
        ]
    )
    violations, _ = check_body_sentence_classification(draft, writing_result, report, "s")
    assert violations == []


def test_check_body_sentence_classification_flags_an_uncited_finding_that_survived():
    draft = _draft(_paragraph("This unpublished finding was never cited."))
    writing_result = {
        "uncited_sentences": [_uncited("This unpublished finding was never cited.", "finding")]
    }
    violations, _ = check_body_sentence_classification(draft, writing_result, None, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_UNCITED_FINDING_SURVIVED


def test_check_body_sentence_classification_flags_a_wholly_unclassified_sentence():
    """A sentence the citation-link call returns in neither list at all."""
    draft = _draft(_paragraph("Mixed-effect linear models showed a durable advantage."))
    writing_result = {"uncited_sentences": []}
    violations, _ = check_body_sentence_classification(draft, writing_result, None, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_UNCLASSIFIED_SENTENCE
    assert "Mixed-effect" in violations[0].sentence


def test_check_body_sentence_classification_never_flags_a_bold_heading_shaped_sentence():
    """A whole-line bold sub-heading a citation-link call, given no
    structural context, tagged "finding" (or left unclassified) must never be reported
    -- `_build_section_tiptap_nodes` promotes exactly this shape to a real heading node
    whenever it is the whole of its own block."""
    draft = _draft(_paragraph("**Key Findings**"))
    writing_result = {
        "uncited_sentences": [_uncited("**Key Findings**", "finding")]
    }
    violations, _ = check_body_sentence_classification(draft, writing_result, None, "s")
    assert violations == []


def test_check_body_sentence_classification_reads_the_drafts_own_attrs_for_a_later_heal():
    """``writing_result.json`` is a one-time snapshot from
    generation; a sentence a LATER heal (the standalone verify-and-heal action, or a
    revision's own second citation-link pass) classifies correctly is read from the
    saved draft's own ``attrs.uncitedSentences``, not reported as a stale defect just
    because ``writing_result.json`` still shows it unclassified."""
    sentence = "Mixed-effect models showed a durable advantage four weeks later."
    draft = _draft({
        "type": "paragraph",
        "content": [{"type": "text", "text": sentence}],
        "attrs": {"uncitedSentences": [_uncited(sentence, "framing")]},
    })
    writing_result = {"uncited_sentences": []}
    violations, _ = check_body_sentence_classification(draft, writing_result, None, "s")
    assert violations == []


def test_check_body_sentence_classification_flags_a_never_classified_finding_distinctly():
    """An entry still carrying ``"unclassified": True`` (a
    citation-link retry that failed or never mentioned the sentence, kept by finalize
    rather than removed) is reported under a message that
    blames the retry, not one that says finalize should have removed it -- the two
    point a reader at different subsystems to fix."""
    sentence = "Cognitive load was lower for one feedback type on grammatical errors."
    draft = _draft(_paragraph(sentence))
    writing_result = {
        "uncited_sentences": [
            {**_uncited(sentence, "finding"), "unclassified": True},
        ]
    }
    violations, _ = check_body_sentence_classification(draft, writing_result, None, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_UNCITED_FINDING_SURVIVED
    assert "finalize should have removed it" not in violations[0].detail
    assert "citation-link retry" in violations[0].detail


def test_check_body_sentence_classification_flags_a_bold_sentence_carrying_a_citation():
    """A whole-line bold run carrying a citation is a claim
    sentence the writer happened to bold, not a sub-heading -- this checker's own
    bold-heading exemption must not swallow it either."""
    sentence = "**Direct corrections outperformed metalinguistic codes by 32% (Smith, 2020).**"
    draft = _draft(_paragraph(sentence))
    writing_result = {"uncited_sentences": [_uncited(sentence, "finding")]}
    violations, _ = check_body_sentence_classification(draft, writing_result, None, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_UNCITED_FINDING_SURVIVED


def test_check_body_sentence_classification_notices_when_uncited_sentences_is_absent():
    """An older writing_result.json shape (before the field existed): flagging every
    uncited sentence would be a flood of false positives with no ground truth, so
    this is a notice, not a check."""
    draft = _draft(_paragraph("Some framing prose with no citation at all."))
    writing_result = {"citation_links": []}
    violations, notices = check_body_sentence_classification(draft, writing_result, None, "s")
    assert violations == []
    assert len(notices) == 1
    assert "uncited_sentences" in notices[0]


def test_check_body_sentence_classification_notices_when_writing_result_is_none():
    violations, notices = check_body_sentence_classification(
        _draft(_paragraph("Text.")), None, None, "s"
    )
    assert violations == []
    assert len(notices) == 1


# ---------------------------------------------------------------------------
# Rule 7: fixture claims
# ---------------------------------------------------------------------------


def _protocol_claims():
    return [
        {
            "id": "supported-1",
            "text": "Fixture supports this claim.",
            "expected_status": "verified",
        },
        {
            "id": "unsupported-1",
            "text": "Fixture rejects this other claim.",
            "expected_status": "unsupported",
        },
    ]


def test_check_fixture_claims_passes_when_verified_present_and_unsupported_absent():
    draft = _draft(_paragraph("Fixture supports this claim."))
    report = _claim_report([_final_row("Fixture supports this claim.")])
    violations, notices = check_fixture_claims(_protocol_claims(), draft, report, "s")
    assert violations == []
    assert notices == []


def test_check_fixture_claims_flags_a_missing_verified_fixture():
    draft = _draft(_paragraph("Nothing relevant."))
    report = _claim_report([])
    violations, _ = check_fixture_claims(_protocol_claims(), draft, report, "s")
    assert any(v.rule == RULE_FIXTURE_VERIFIED_MISSING for v in violations)


def test_check_fixture_claims_passes_a_recorded_guard_demotion():
    """A "verified" fixture absent from the final report still passes
    when its own raw (pre-heal) verification row records the one guard-demotion shape
    that is not model drift -- the model itself said "verified" and the frozen
    ``quote_not_verbatim`` guard, alone, demoted it -- exactly what `run_demo.py`'s own
    printed table already calls "GUARD DEMOTION" rather than a failure."""
    draft = _draft(_paragraph("Nothing relevant."))
    report = {
        "final_report": {"verifications": [], "verified_count": 0},
        "verifications": [
            {
                "claim_text": "Fixture supports this claim.",
                "status": "needs_nuance",
                "model_status": "verified",
                "machine_reasons": ["quote_not_verbatim"],
            }
        ],
    }
    violations, _ = check_fixture_claims(_protocol_claims(), draft, report, "s")
    assert violations == []


def test_check_fixture_claims_still_flags_a_needs_nuance_row_with_another_reason():
    """The guard-demotion allowance is narrow: a second reason alongside
    ``quote_not_verbatim``, or a ``model_status`` other than "verified", is ordinary
    model drift, not the one shape the frozen guard forces -- still a violation."""
    draft = _draft(_paragraph("Nothing relevant."))
    report = {
        "final_report": {"verifications": [], "verified_count": 0},
        "verifications": [
            {
                "claim_text": "Fixture supports this claim.",
                "status": "needs_nuance",
                "model_status": "verified",
                "machine_reasons": ["quote_not_verbatim", "assertion_status_inconsistent"],
            }
        ],
    }
    violations, _ = check_fixture_claims(_protocol_claims(), draft, report, "s")
    assert any(v.rule == RULE_FIXTURE_VERIFIED_MISSING for v in violations)


def test_check_fixture_claims_still_flags_a_genuinely_unsupported_fixture():
    """No raw row at all, or a raw row whose own status is not a guard demotion
    (``unsupported``, the model's own real verdict): still a violation, exactly as
    before this rule was reconciled with `run_demo.py`'s own reading."""
    draft = _draft(_paragraph("Nothing relevant."))
    report = {
        "final_report": {"verifications": [], "verified_count": 0},
        "verifications": [
            {
                "claim_text": "Fixture supports this claim.",
                "status": "unsupported",
                "model_status": "unsupported",
                "machine_reasons": ["assertion_status_inconsistent"],
            }
        ],
    }
    violations, _ = check_fixture_claims(_protocol_claims(), draft, report, "s")
    assert any(v.rule == RULE_FIXTURE_VERIFIED_MISSING for v in violations)


def test_check_fixture_claims_flags_an_unsupported_fixture_still_in_the_draft():
    draft = _draft(
        _paragraph("Fixture supports this claim."), _paragraph("Fixture rejects this other claim.")
    )
    report = _claim_report([_final_row("Fixture supports this claim.")])
    violations, _ = check_fixture_claims(_protocol_claims(), draft, report, "s")
    assert any(v.rule == RULE_FIXTURE_UNSUPPORTED_PRESENT for v in violations)


def test_check_fixture_claims_returns_empty_when_no_protocol_claims_given():
    assert check_fixture_claims(None, _draft(), None, "s") == ([], [])
    assert check_fixture_claims([], _draft(), None, "s") == ([], [])


def test_check_fixture_claims_notices_when_final_report_is_absent():
    draft = _draft(_paragraph("Anything."))
    violations, notices = check_fixture_claims(
        _protocol_claims(), draft, {"verifications": []}, "s"
    )
    assert violations == []
    assert len(notices) == 1
    assert "final_report" in notices[0]


# ---------------------------------------------------------------------------
# Rule 8: the deterministic coherence pass
# ---------------------------------------------------------------------------


def test_check_paragraph_has_citation_flags_a_non_opening_uncited_paragraph():
    draft = _draft(
        _heading("Lit review"),
        _paragraph("Tutoring works well (Smith, 2020)."),
        _heading("Gaps"),
        _paragraph("These gaps motivate the present investigation."),
    )
    violations = check_paragraph_has_citation(draft, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_PARAGRAPH_HAS_NO_CITATION
    assert violations[0].detail.startswith("paragraph 1:")


def test_check_paragraph_has_citation_keeps_the_opening_paragraph_even_uncited():
    draft = _draft(
        _heading("Lit review"),
        _paragraph("This review synthesises the project library."),
    )
    assert check_paragraph_has_citation(draft, "s") == []


def test_check_paragraph_has_citation_recognises_a_narrative_citation():
    """A narrative citation ("Zhang and Hyland (2021) found...") is a citation, even
    with no bracketed author name -- a purely bracket-shaped detector would
    misclassify this as uncited."""
    draft = _draft(
        _heading("Lit review"),
        _paragraph("Tutoring works well (Smith, 2020)."),
        _heading("More"),
        _paragraph("Zhang and Hyland (2021) found further gains."),
    )
    assert check_paragraph_has_citation(draft, "s") == []


def test_check_paragraph_has_citation_keeps_a_paragraph_with_an_unclassified_sentence():
    """The guarantee that a sentence a citation-link retry could not classify is
    never silently dropped outranks rule 8a here, mirroring
    `app.services.fulltext.finalize_generated_section`'s own exemption (see also
    `test_resolve_and_finalize_keep_orphaned_sentences_from_writing_result_on_retry_
    failure`): a genuine, coherent closing synthesis paragraph can carry no
    citation of its own."""
    draft = _draft(
        _heading("Lit review"),
        _paragraph("Tutoring works well (Smith, 2020)."),
        {
            "type": "paragraph",
            "attrs": {
                "uncitedSentences": [
                    {
                        "sentence": "No study in the library compares all three sources.",
                        "tag": "finding",
                        "unclassified": True,
                    }
                ]
            },
            "content": [
                {
                    "type": "text",
                    "text": "No study in the library compares all three sources.",
                }
            ],
        },
    )
    assert check_paragraph_has_citation(draft, "s") == []


def test_check_paragraph_has_citation_ignores_a_markdown_or_bold_heading_paragraph():
    """An older saved run can carry a heading as a plain paragraph node, either
    as a literal "#" line or a whole-line bold run -- neither is a body paragraph."""
    draft = _draft(
        _paragraph("## Restated title"),
        _paragraph("**Sub-heading Restated**"),
        _paragraph("Tutoring works well (Smith, 2020)."),
    )
    assert check_paragraph_has_citation(draft, "s") == []


def test_check_dangling_framing_flags_an_enumeration_opener_short_of_its_count():
    draft = _draft(
        _heading("Lit review"),
        _paragraph(
            "Tutoring works well (Smith, 2020). Three gaps emerge. "
            "These gaps motivate the present review's contribution."
        ),
    )
    violations = check_dangling_framing(draft, "s")
    rules = [v.rule for v in violations]
    assert rules == [
        RULE_ENUMERATION_OPENER_SHORT_OF_COUNT,
        RULE_DANGLING_FRAMING_SENTENCE,
    ]


def test_check_dangling_framing_keeps_a_self_resolving_colon_enumeration():
    """"Three gaps remain: A; B; and C." resolves its own count in the same sentence,
    whatever else follows it in the paragraph -- not a shortfall."""
    draft = _draft(
        _heading("Lit review"),
        _paragraph(
            "Three gaps remain: short designs; small corpora; and no engagement "
            "tracking. These gaps motivate the present review's focus."
        ),
    )
    assert check_dangling_framing(draft, "s") == []


def test_check_dangling_framing_keeps_a_self_resolving_colon_free_ordinal_list():
    """A colon-free list resolves the SAME count one sentence
    later ("First, ...; second, ...; third, ..."), which is one surviving sentence, not
    three -- mirrors ``app.services.fulltext._next_fragment_resolves_enumeration_as_a_
    list``."""
    draft = _draft(
        _heading("Lit review"),
        _paragraph(
            "Three gaps emerge. First, designs are short; second, corpora are small; "
            "third, engagement is untracked (Smith, 2020)."
        ),
    )
    assert check_dangling_framing(draft, "s") == []


def test_check_dangling_framing_ignores_a_plain_factual_verb():
    """"are"/"were"/"exist(s/ed)"/"remain(s/ed)" are generic
    existential verbs a plain factual sentence uses just as often as a real
    announcement, so they never trigger 8b at all."""
    draft = _draft(
        _heading("Lit review"),
        _paragraph(
            "Tutoring works well (Smith, 2020). Two limitations remain unaddressed "
            "in the literature."
        ),
    )
    assert check_dangling_framing(draft, "s") == []


def test_check_dangling_framing_does_not_flag_a_cited_paragraph_right_after_a_heading():
    """A paragraph's own first sentence, with the node before it a heading, opening
    with a discourse connective and an unresolved deictic, but carrying its own
    citation, is not flagged: the check is narrowed to exactly what the app-side
    fix's own rule (c) can guarantee (an UNCITED fragment, trouble only from an
    in-paragraph 8b cascade), since removing a cited sentence would delete supported
    content and no deterministic pass can do that. This shape is accepted residue,
    tracked separately rather than gated on this rule's exit code."""
    draft = _draft(
        _heading("Lit review"),
        _paragraph("Tutoring works well (Smith, 2020)."),
        _heading("Sub-section"),
        _paragraph(
            "However, the study's own authors caution that it was conducted within "
            "a single instructional setting (Bonilla Lopez et al., 2018)."
        ),
    )
    assert check_dangling_framing(draft, "s") == []


def test_check_dangling_framing_does_not_flag_a_benign_second_sentence_after_a_heading():
    """The one thing rule 8c must not do: treat a paragraph's OWN, present first
    sentence as no antecedent for a "however"/"this" that plainly answers it, just
    because the paragraph itself is heading-adjacent."""
    draft = _draft(
        _heading("Lit review"),
        _paragraph("Tutoring works well (Smith, 2020)."),
        _heading("Sub-section"),
        _paragraph(
            "Luo et al. (2025) found that precision remained high. However, "
            "detection accuracy does not guarantee developmental benefit."
        ),
    )
    assert check_dangling_framing(draft, "s") == []


def test_check_dangling_framing_8c_never_sets_trouble_before_itself():
    """`trouble_before` is set only by an 8b firing
    (`if fired_b: trouble_before = True`), never by an 8c firing on the sentence
    before -- so a THIRD sentence, right after one 8c already flagged, is judged purely
    on whatever 8b did earlier in the paragraph, not on the 8c violation next to it.
    "Three gaps emerge." announces 3 and only two sentences follow it (a genuine 8b
    shortfall, which keeps trouble alive for the rest of the paragraph); "However,
    prior work never addressed them." is flagged by 8c; "Complementing this, Mao et al.
    (2024) found further evidence." is cited, so 8c's own uncited-only gate keeps it
    unflagged regardless of what 8c did to the sentence before it."""
    draft = _draft(
        _paragraph(
            "Three gaps emerge. However, prior work never addressed them. "
            "Complementing this, Mao et al. (2024) found further evidence."
        ),
    )
    violations = check_dangling_framing(draft, "s")
    rules = sorted(v.rule for v in violations)
    assert rules == [RULE_DANGLING_FRAMING_SENTENCE, RULE_ENUMERATION_OPENER_SHORT_OF_COUNT]
    assert all("sentence 2" not in v.detail for v in violations)


def test_check_dangling_framing_does_not_flag_the_bracketed_accented_surname_shape():
    """`_paragraph_has_rendered_citation` does not require an ASCII author-name shape
    inside the brackets -- "(Li & Hebert, 2023)" is a citation whatever the surname
    looks like, so the sentence carrying it correctly sets ``trouble_before`` to
    False for what follows it, unlike an unrecognised citation would: the paragraph's
    real, present, cited first sentence is a genuine antecedent for "Notably, ..."
    right after it."""
    draft = _draft(
        _heading("Lit review"),
        _paragraph("Tutoring works well (Smith, 2020)."),
        _heading("Sub-section"),
        _paragraph(
            "In an online study, 12 writers incorporated peer feedback (Li & Hebert, "
            "2023). Notably, students ignored peer comments that conflicted with "
            "their own judgements (Li & Hebert, 2023)."
        ),
    )
    assert check_dangling_framing(draft, "s") == []


# ---------------------------------------------------------------------------
# Rule 9: a heading with no body
# ---------------------------------------------------------------------------


def test_check_heading_has_no_body_flags_a_trailing_heading():
    draft = _draft(
        _heading("Lit review"),
        _paragraph("Tutoring works well (Smith, 2020)."),
        _heading("Synthesis and Gaps"),
    )
    violations = check_heading_has_no_body(draft, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_HEADING_WITH_NO_BODY
    assert "node 2" in violations[0].detail


def test_check_heading_has_no_body_flags_two_adjacent_headings():
    draft = _draft(
        _heading("Lit review"),
        _paragraph("Tutoring works well (Smith, 2020)."),
        _heading("Design"),
        _heading("Synthesis and Gaps"),
        _paragraph("These gaps motivate the present investigation (Jones, 2019)."),
    )
    violations = check_heading_has_no_body(draft, "s")
    assert len(violations) == 1
    assert "node 2" in violations[0].detail


def test_check_heading_has_no_body_never_flags_the_documents_own_first_node_even_adjacent():
    """The document's own opening node (index 0) is exempt even when a second heading
    follows it immediately with no body between them -- only the second heading
    (index 1), itself the section's own last node here, is flagged."""
    draft = _draft(_heading("Lit review"), _heading("Sub-section"))
    violations = check_heading_has_no_body(draft, "s")
    assert len(violations) == 1
    assert "node 1" in violations[0].detail


def test_check_heading_has_no_body_never_flags_the_documents_own_first_node():
    """Mirrors ``app.services.fulltext._heading_no_body_keep_mask``'s own exemption: a
    section reduced to nothing still keeps its own title."""
    draft = _draft(_heading("Lit review"))
    assert check_heading_has_no_body(draft, "s") == []


def test_check_heading_has_no_body_keeps_a_heading_with_a_real_body():
    draft = _draft(
        _heading("Lit review"),
        _paragraph("Tutoring works well (Smith, 2020)."),
        _heading("Synthesis and Gaps"),
        _paragraph("These gaps motivate the present investigation (Smith, 2020)."),
    )
    assert check_heading_has_no_body(draft, "s") == []


# ---------------------------------------------------------------------------
# New rule: a generated section carries only its own title heading -- companion
# to rule 9, independent of whether the extra heading has a body of its own.
# ---------------------------------------------------------------------------


def test_check_non_title_heading_flags_a_second_heading_with_a_real_body():
    """Unlike rule 9 (`check_heading_has_no_body`), this rule fires even when the
    extra heading has a perfectly good body underneath it -- a heading this
    product's own writer is never supposed to deliver any more, whatever it
    introduces."""
    draft = _draft(
        _heading("Lit review"),
        _heading("Effects of Feedback Types on Accuracy and Revision"),
        _paragraph("Tutoring works well (Smith, 2020)."),
    )
    violations = check_non_title_heading(draft, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_NON_TITLE_HEADING_IN_SECTION
    assert "node 1" in violations[0].detail


def test_check_non_title_heading_flags_every_extra_heading_not_only_the_first():
    draft = _draft(
        _heading("Lit review"),
        _paragraph("Tutoring works well (Smith, 2020)."),
        _heading("Design"),
        _paragraph("Sources were selected for their relevance (Jones, 2019)."),
    )
    violations = check_non_title_heading(draft, "s")
    assert len(violations) == 1
    assert "node 2" in violations[0].detail


def test_check_non_title_heading_never_flags_the_documents_own_first_node():
    draft = _draft(_heading("Lit review"), _paragraph("Tutoring works well (Smith, 2020)."))
    assert check_non_title_heading(draft, "s") == []


def test_check_non_title_heading_passes_run_8_section_3_once_the_second_heading_is_gone():
    """Run 8's own delivered ``draft_content_3.json`` failed this shape (the
    section-title heading directly followed by the writer's own leading
    sub-heading); the fixed app-side pipeline never delivers it again, so a draft
    carrying only the title heading passes."""
    draft = _draft(
        _heading("Learner engagement with written corrective feedback"),
        _paragraph(
            "Immediate accuracy during revision improved under both direct and "
            "coded feedback (Bonilla López et al., 2018)."
        ),
    )
    assert check_non_title_heading(draft, "extra section 3") == []


# ---------------------------------------------------------------------------
# Rule 5: delivered_evidence.json
# ---------------------------------------------------------------------------


def _evidence_row(**overrides):
    base = {
        "row": 1,
        "section_title": "protocol section",
        "sentence": "Learners improved after feedback.",
        "passage_located": True,
        "source_passage": "The intervention produced a gain in vocabulary retention.",
        "evidence_quotes": ["a gain in vocabulary retention"],
    }
    base.update(overrides)
    return base


def test_check_delivered_evidence_rows_passes_a_verbatim_quote():
    assert check_delivered_evidence_rows([_evidence_row()]) == []


def test_check_delivered_evidence_rows_matches_under_the_guard_fold_despite_typography():
    row = _evidence_row(
        source_passage="The intervention produced a gain in vocabulary–retention overall.",
        evidence_quotes=["a gain in vocabulary retention"],
    )
    assert check_delivered_evidence_rows([row]) == []


def test_check_delivered_evidence_rows_flags_passage_located_false():
    row = _evidence_row(passage_located=False)
    violations = check_delivered_evidence_rows([row])
    assert len(violations) == 1
    assert violations[0].rule == RULE_PASSAGE_NOT_LOCATED


def test_check_delivered_evidence_rows_flags_a_quote_not_found_in_the_passage():
    row = _evidence_row(evidence_quotes=["a fact that is nowhere in the passage"])
    violations = check_delivered_evidence_rows([row])
    assert len(violations) == 1
    assert violations[0].rule == RULE_EVIDENCE_NOT_VERBATIM


def test_is_verbatim_under_guard_fold_exact_and_folded_and_absent():
    assert is_verbatim_under_guard_fold("a gain", "There was a gain overall.")
    assert is_verbatim_under_guard_fold("co-operative", "a co–operative approach")
    assert not is_verbatim_under_guard_fold("nowhere at all", "unrelated text")
    assert not is_verbatim_under_guard_fold("", "text")
    assert not is_verbatim_under_guard_fold("quote", "")


# ---------------------------------------------------------------------------
# File loading and orchestration
# ---------------------------------------------------------------------------


def test_load_section_files_records_missing_files(tmp_path: Path):
    (tmp_path / "draft_content.json").write_text(
        json.dumps(_draft(_paragraph("Hi."))), encoding="utf-8"
    )
    files = load_section_files(tmp_path, index=None)
    assert files.draft_content is not None
    assert files.writing_result is None
    assert files.claim_report is None
    assert set(files.missing) == {"writing_result.json", "claim_report.json"}
    assert files.name == "protocol section"


def test_load_section_files_uses_the_indexed_filenames_for_an_extra_section(tmp_path: Path):
    (tmp_path / "draft_content_2.json").write_text(json.dumps(_draft()), encoding="utf-8")
    files = load_section_files(tmp_path, index=2)
    assert files.draft_content is not None
    assert files.name == "extra section 2"
    assert files.missing == ["writing_result_2.json", "claim_report_2.json"]


def test_discover_extra_section_indices_finds_every_indexed_draft(tmp_path: Path):
    for name in ("draft_content_2.json", "draft_content_3.json", "draft_content.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert discover_extra_section_indices(tmp_path) == [2, 3]


def test_discover_extra_section_indices_empty_when_none_present(tmp_path: Path):
    assert discover_extra_section_indices(tmp_path) == []


def test_check_section_notices_missing_files_and_skips_when_draft_is_absent(tmp_path: Path):
    files = SectionFiles(name="protocol section", index=None, missing=["draft_content.json"])
    violations, notices = check_section(files)
    assert violations == []
    assert len(notices) == 1


# ---------------------------------------------------------------------------
# check_run_directory: end-to-end over a synthetic run directory
# ---------------------------------------------------------------------------


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_check_run_directory_over_a_clean_synthetic_run_has_no_violations(tmp_path: Path):
    draft = _draft(
        _heading("Literature Review"),
        _paragraph("This review covers one theme."),
        _paragraph("Smith (2020) found a large effect on accuracy."),
    )
    writing_result = {
        "uncited_sentences": [_uncited("This review covers one theme.", "framing")],
        "citation_links": [
            {
                "sentence": "Smith (2020) found a large effect on accuracy.",
                "keys": ["smith_2020"],
                # Rule 10: matches the final-report row's own
                # `claim_text` below (the whole sentence content, so the sentence is
                # fully covered and this run stays clean).
                "proposition": "found a large effect on accuracy",
            }
        ],
    }
    report = _claim_report(
        [
            _final_row(
                "found a large effect on accuracy",
                claim_sentence="Smith (2020) found a large effect on accuracy.",
            )
        ],
        citation_coverage={"unresolved": 0},
    )
    _write_json(tmp_path / "draft_content.json", draft)
    _write_json(tmp_path / "writing_result.json", writing_result)
    _write_json(tmp_path / "claim_report.json", report)
    _write_json(
        tmp_path / "delivered_evidence.json",
        {
            "rows": [
                {
                    "row": 1,
                    "section_title": "Literature Review",
                    "sentence": "Smith (2020) found a large effect on accuracy.",
                    "passage_located": True,
                    "source_passage": "The study found a large effect on accuracy overall.",
                    "evidence_quotes": ["found a large effect on accuracy"],
                }
            ],
            "unlocated_rows": 0,
        },
    )

    result = check_run_directory(tmp_path)
    assert isinstance(result, CheckResult)
    assert result.ok
    assert result.violations == []
    assert result.sections_checked == ["protocol section"]


def test_check_run_directory_notices_an_entirely_empty_protocol_section(tmp_path: Path):
    """An empty or failed run directory (none of the protocol section's three files
    present) must not print "OK ... (0 section(s) checked)" and exit 0 with no notice
    at all: this module's own stated rule is that a missing file is named rather than
    silently skipped."""
    result = check_run_directory(tmp_path)
    assert result.ok
    assert result.sections_checked == []
    assert any("draft_content.json not found" in n for n in result.notices)
    assert any("writing_result.json not found" in n for n in result.notices)
    assert any("claim_report.json not found" in n for n in result.notices)
    assert any(
        "protocol section" in n and "no rules could be checked" in n for n in result.notices
    )


def test_check_run_directory_reports_missing_delivered_evidence_as_a_notice_not_a_violation(
    tmp_path: Path,
):
    _write_json(tmp_path / "draft_content.json", _draft(_paragraph("Text.")))
    _write_json(
        tmp_path / "writing_result.json", {"uncited_sentences": [_uncited("Text.", "framing")]}
    )
    _write_json(tmp_path / "claim_report.json", {"final_report": {"verifications": []}})
    result = check_run_directory(tmp_path)
    assert result.violations == []
    assert any("delivered_evidence.json not found" in n for n in result.notices)


def test_check_run_directory_discovers_and_checks_extra_sections(tmp_path: Path):
    _write_json(
        tmp_path / "draft_content.json", _draft(_paragraph("Smith (2020) found an effect."))
    )
    _write_json(
        tmp_path / "writing_result.json",
        {
            "uncited_sentences": [],
            "citation_links": [{"sentence": "Smith (2020) found an effect."}],
        },
    )
    _write_json(
        tmp_path / "claim_report.json",
        _claim_report(
            [_final_row("found an effect", claim_sentence="Smith (2020) found an effect.")]
        ),
    )
    # Extra section 2 carries a genuine orphaned sentence.
    _write_json(
        tmp_path / "draft_content_2.json",
        _draft(_paragraph("An orphaned empirical claim appears here.")),
    )
    _write_json(tmp_path / "writing_result_2.json", {"uncited_sentences": []})
    _write_json(tmp_path / "claim_report_2.json", {"final_report": {"verifications": []}})

    result = check_run_directory(tmp_path)
    assert result.sections_checked == ["protocol section", "extra section 2"]
    assert any(
        v.rule == RULE_UNCLASSIFIED_SENTENCE and v.section == "extra section 2"
        for v in result.violations
    )


def test_check_run_directory_checks_fixture_claims_only_on_the_protocol_section(tmp_path: Path):
    _write_json(tmp_path / "draft_content.json", _draft(_paragraph("Fixture supports this claim.")))
    _write_json(tmp_path / "writing_result.json", {"uncited_sentences": []})
    _write_json(
        tmp_path / "claim_report.json",
        _claim_report([_final_row("Fixture supports this claim.")]),
    )
    protocol = {"claims": _protocol_claims()}
    result = check_run_directory(tmp_path, protocol=protocol)
    assert any(v.rule == RULE_FIXTURE_VERIFIED_MISSING for v in result.violations) is False
    assert any(v.rule == RULE_FIXTURE_UNSUPPORTED_PRESENT for v in result.violations) is False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_exits_0_for_a_clean_run_directory(tmp_path: Path, capsys):
    _write_json(tmp_path / "draft_content.json", _draft(_paragraph("Text.")))
    _write_json(
        tmp_path / "writing_result.json", {"uncited_sentences": [_uncited("Text.", "framing")]}
    )
    _write_json(tmp_path / "claim_report.json", {"final_report": {"verifications": []}})
    rc = main([str(tmp_path), "--protocol", str(tmp_path / "no-protocol.json")])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_main_exits_2_and_prints_the_rule_name_on_a_violation(tmp_path: Path, capsys):
    _write_json(
        tmp_path / "draft_content.json", _draft(_paragraph("An orphaned claim with no citation."))
    )
    _write_json(tmp_path / "writing_result.json", {"uncited_sentences": []})
    rc = main([str(tmp_path), "--protocol", str(tmp_path / "no-protocol.json")])
    err = capsys.readouterr().err
    assert rc == 2
    assert RULE_UNCLASSIFIED_SENTENCE in err


# ---------------------------------------------------------------------------
# Historical regression: every tracked run directory under demo/output/
# ---------------------------------------------------------------------------

HISTORICAL_RUNS = [
    "20260909-071345",
    "20260910-095226",
    "20260911-042940",
    "20260911-062145",
    "20260911-152912",
    "20260911-171918",
]


def _historical_result(name: str) -> CheckResult:
    protocol = json.loads((DEMO / "protocol.json").read_text(encoding="utf-8"))
    run_dir = skip_unless_run_dir(DEMO / "output" / name)
    return check_run_directory(run_dir, protocol=protocol)


@pytest.mark.parametrize("name", HISTORICAL_RUNS)
def test_historical_run_directory_is_checkable_without_raising(name: str):
    _historical_result(name)


def test_historical_run_20260909_071345_predates_most_fields_and_says_so():
    """Rule 8a (`RULE_PARAGRAPH_HAS_NO_CITATION`) catches this run's own closing
    synthesis paragraph, which carries no citation at all and is not the section's
    own opening paragraph -- the exact shape the coherence pass
    (`app.services.fulltext`'s rule (a)) removes on every run written after it.

    This run also predates the backend's own promotion of a markdown "#"/"##" line to
    a real heading node, so its own draft opens with a real heading (the section
    title) immediately followed by a markdown "# Literature Review" line (itself
    heading-shaped, `_paragraph_is_heading_shaped`, exempt from rule 15 for exactly
    that reason) with a second markdown "##" heading line right after it and no body
    paragraph between them -- genuinely a heading with no body of its own, caught by
    rule 9 (`RULE_HEADING_WITH_NO_BODY`). Rule 16 (`RULE_HEADING_SHAPED_PARAGRAPH`)
    is not exempt the same way -- it asks about content shape, not markup, and every
    one of these five stored "#"/"###" lines genuinely reads as a heading (title
    case, no verb, no citation, no sentence-final punctuation) whatever markup
    survived in its own text -- so this run's every markdown-heading-shaped
    paragraph node is now also reported under that rule.

    This run also predates the writer's own bare-form-only ``[NEEDS CITATION]``
    marker convention: two of its sentences still carry the marker with an
    explanatory clause inside the brackets (``[NEEDS CITATION for specific
    validation evidence]``, ``[NEEDS CITATION for specific engagement
    outcomes]``), a shape the exact-form checker rule never saw before it was
    widened to match the same clause-carrying variant the backend's own
    ``_NEEDS_CITATION_RE`` strips. Both are now correctly reported under rule 2
    (`RULE_NEEDS_CITATION_MARKER`), a defect this run always had that only the
    widened rule can see."""
    result = _historical_result("20260909-071345")
    assert [v.rule for v in result.violations] == [
        RULE_HEADING_SHAPED_PARAGRAPH,
        RULE_HEADING_SHAPED_PARAGRAPH,
        RULE_HEADING_SHAPED_PARAGRAPH,
        RULE_HEADING_SHAPED_PARAGRAPH,
        RULE_HEADING_SHAPED_PARAGRAPH,
        RULE_NEEDS_CITATION_MARKER,
        RULE_NEEDS_CITATION_MARKER,
        RULE_PARAGRAPH_HAS_NO_CITATION,
        RULE_HEADING_WITH_NO_BODY,
    ]
    heading_shaped = [
        v.sentence for v in result.violations if v.rule == RULE_HEADING_SHAPED_PARAGRAPH
    ]
    assert heading_shaped == [
        "# Literature Review",
        "### Differential Efficacy of Direct Corrections and Metalinguistic Codes",
        "### Learner Engagement, Affect, and Revision Processes",
        "### Automated Feedback, Teacher Practices, and the Role of Artificial Intelligence",
        "### Synthesis and Research Gaps",
    ]
    joined_notices = " ".join(result.notices)
    assert "uncited_sentences" in joined_notices
    assert "final_report" in joined_notices
    assert "delivered_evidence.json not found" in joined_notices


def test_historical_run_20260911_042940_catches_the_truncated_vs_claim():
    """The sentence extractor cut a cited claim in two at the "vs." abbreviation."""
    result = _historical_result("20260911-042940")
    truncated = [v for v in result.violations if v.rule == RULE_TRUNCATED_SENTENCE]
    assert truncated, "expected a truncated_sentence violation on 20260911-042940"
    assert any(v.sentence and v.sentence.rstrip().endswith("vs.") for v in truncated)


def test_historical_run_20260911_171918_catches_the_orphaned_empirical_sentence():
    """An uncited, unverified empirical claim reached the delivered text with no
    citation and no verification row at all -- the citation-link call returned it in
    neither list."""
    result = _historical_result("20260911-171918")
    unclassified = [v for v in result.violations if v.rule == RULE_UNCLASSIFIED_SENTENCE]
    assert any(v.sentence and "Mixed-effect linear models" in v.sentence for v in unclassified)


def test_historical_run_20260911_171918_and_20260911_152912_carry_the_duplicate_heading():
    """Present on every promoted baseline checked -- still visible in the tracked
    history."""
    for name in ("20260911-171918", "20260911-152912"):
        result = _historical_result(name)
        assert any(v.rule == RULE_DUPLICATE_HEADING for v in result.violations), name


def test_historical_run_20260911_062145_carries_the_known_guard_demotion_gap():
    """test_expected_baseline.py's own comment: on this baseline the frozen
    quote_not_verbatim guard demoted supported-1 to needs_nuance, so it does not
    survive into the final report -- rule 7 (which does not special-case a guard
    demotion, unlike run_demo.py's own claim_problems/is_guard_demotion) reports it."""
    result = _historical_result("20260911-062145")
    assert any(v.rule == RULE_FIXTURE_VERIFIED_MISSING for v in result.violations)


def test_historical_run_20260910_095226_has_no_evidence_file_and_says_so():
    result = _historical_result("20260910-095226")
    assert any("delivered_evidence.json not found" in n for n in result.notices)


def test_historical_runs_with_delivered_evidence_pass_rule_5():
    """Rule 5 independently finds passage_located true on every row and
    unlocated_rows 0 on 20260911-171918 (its own delivered_evidence.json is committed
    history)."""
    result = _historical_result("20260911-171918")
    assert not any(
        v.rule in (RULE_EVIDENCE_NOT_VERBATIM, RULE_PASSAGE_NOT_LOCATED) for v in result.violations
    )


# ---------------------------------------------------------------------------
# Rules 10 (sentence_exceeds_verified_claim) and 11 (comparative_meta_evaluation).
# ---------------------------------------------------------------------------


def _row(
    claim_text, citation, *, status="verified", claim_sentence=None,
    evidence_quote=None, evidence_quotes=None,
):
    row = {"claim_text": claim_text, "citation": citation, "status": status}
    if claim_sentence is not None:
        row["claim_sentence"] = claim_sentence
    if evidence_quote is not None:
        row["evidence_quote"] = evidence_quote
    if evidence_quotes is not None:
        row["evidence_quotes"] = evidence_quotes
    return row


def _clink(sentence, proposition, citation_text, keys=("k_2020",)):
    return {
        "sentence": sentence,
        "proposition": proposition,
        "citation_text": citation_text,
        "keys": list(keys),
    }


def test_check_sentence_covered_flags_a_residue_no_verified_claim_covers():
    sentence = (
        "Liu and Wu (2019) add that 60% of students preferred teacher feedback, "
        "underscoring the persistent value of human judgement."
    )
    draft = _draft(_paragraph(sentence))
    writing_result = {
        "citation_links": [_clink(sentence, "60% of students preferred teacher feedback", "(2019)")]
    }
    report = _claim_report(
        [_row("60% of students preferred teacher feedback", "(2019)", claim_sentence=sentence)]
    )
    violations, notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    # The per-proposition half always reports its own match rate,
    # so a section with one writer proposition that matched reports it here.
    assert notices == ["s: 1 of 1 writer propositions matched a verdict"]
    assert len(violations) == 1
    assert violations[0].rule == RULE_SENTENCE_EXCEEDS_VERIFIED_CLAIM
    assert "underscoring" in violations[0].detail


def test_check_sentence_covered_keeps_a_residue_whose_numeral_is_in_an_evidence_quote():
    """The numeral-coverage relaxation: a residue whose
    only problem is a numeral no proposition covers is not a violation when the
    same number is verbatim in one of the sentence's own verified claims' evidence
    quotes."""
    sentence = (
        "In a study of 139 EFL writers, most preferred direct correction "
        "(Bonilla López et al., 2018)."
    )
    claim = "most preferred direct correction"
    draft = _draft(_paragraph(sentence))
    writing_result = {
        "citation_links": [
            _clink(sentence, claim, "(Bonilla López et al., 2018)", keys=("lopez_2018",))
        ]
    }
    report = _claim_report([
        _row(
            claim, "(Bonilla López et al., 2018)", claim_sentence=sentence,
            evidence_quote="A total of 139 low-intermediate EFL writers took part.",
        )
    ])
    violations, _notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    assert violations == []


def test_check_sentence_covered_still_flags_a_residue_numeral_in_no_evidence_quote():
    sentence = (
        "In a study of 139 EFL writers, most preferred direct correction "
        "(Bonilla López et al., 2018)."
    )
    claim = "most preferred direct correction"
    draft = _draft(_paragraph(sentence))
    writing_result = {
        "citation_links": [
            _clink(sentence, claim, "(Bonilla López et al., 2018)", keys=("lopez_2018",))
        ]
    }
    report = _claim_report([
        _row(
            claim, "(Bonilla López et al., 2018)", claim_sentence=sentence,
            evidence_quote="Learners reported a strong preference for explicit feedback.",
        )
    ])
    violations, _notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    assert len(violations) == 1
    assert violations[0].rule == RULE_SENTENCE_EXCEEDS_VERIFIED_CLAIM


def test_check_sentence_covered_matches_a_number_word_residue_against_a_digit_quote():
    """"seventy" and "70" normalise to the same key (`numeral_keys_in_text`)."""
    sentence = "In a survey of seventy students, most preferred immediate feedback (Liu, 2019)."
    claim = "most preferred immediate feedback"
    draft = _draft(_paragraph(sentence))
    writing_result = {
        "citation_links": [_clink(sentence, claim, "(Liu, 2019)", keys=("liu_2019",))]
    }
    report = _claim_report([
        _row(
            claim, "(Liu, 2019)", claim_sentence=sentence,
            evidence_quote="70 students completed the end-of-term survey.",
        )
    ])
    violations, _notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    assert violations == []


def test_check_sentence_covered_passes_a_fully_covered_sentence():
    sentence = "Smith (2020) found that recall improved by 12 points."
    draft = _draft(_paragraph(sentence))
    writing_result = {
        "citation_links": [_clink(sentence, "recall improved by 12 points", "(2020)")]
    }
    report = _claim_report(
        [_row("recall improved by 12 points", "(2020)", claim_sentence=sentence)]
    )
    violations, _notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    assert violations == []


def test_check_sentence_covered_passes_a_permitted_colon_frame():
    sentence = "This similarly documented selective uptake: teachers preferred global comments."
    draft = _draft(_paragraph(sentence))
    writing_result = {
        "citation_links": [_clink(sentence, "teachers preferred global comments", "(2019)")]
    }
    report = _claim_report(
        [_row("teachers preferred global comments", "(2019)", claim_sentence=sentence)]
    )
    violations, _notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    assert violations == []


def test_check_sentence_covered_passes_the_supported_1_delivered_shape():
    """The demo protocol fixture ``supported-1`` (`demo/protocol.json`) as the run
    actually rendered it. The citation-link step's own proposition began at the
    sentence's own verb,
    leaving "A systematic search for the review" as a leading residue directly abutting
    it with no comma or colon boundary -- the sentence's bare grammatical subject, not a
    separable embellishment, and the claim itself verified. This rule must not flag it."""
    sentence = (
        "A systematic search for the review identified 50 empirical studies meeting "
        "its inclusion criteria, revealing four major themes: teacher feedback "
        "practices in L2 writing classrooms, L2 learner responses to feedback, "
        "stakeholders' beliefs and perspectives about feedback, and feedback-related "
        "motivation and emotions (Mao et al., 2024)."
    )
    claim = (
        "identified 50 empirical studies meeting its inclusion criteria, revealing "
        "four major themes: teacher feedback practices in L2 writing classrooms, L2 "
        "learner responses to feedback, stakeholders' beliefs and perspectives about "
        "feedback, and feedback-related motivation and emotions"
    )
    draft = _draft(_paragraph(sentence))
    writing_result = {
        "citation_links": [_clink(sentence, claim, "(Mao et al., 2024)", keys=("mao_2024",))]
    }
    report = _claim_report(
        [_row(claim, "(Mao et al., 2024)", claim_sentence=sentence)]
    )
    violations, _notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    assert violations == []


def test_check_sentence_covered_flags_a_leading_residue_ending_in_a_coordinating_conjunction():
    """Unlike the bare-subject shape above, a leading
    residue that ends in a coordinating conjunction is a whole independent clause, not
    a subject abutting its own verb, and must still be flagged even though it opens
    the sentence and carries no internal punctuation of its own."""
    sentence = (
        "Teacher feedback remains the gold standard for second language writing "
        "instruction and ChatGPT's error-flagging improved considerably with prompt "
        "sophistication (Luo et al., 2025)."
    )
    claim = "ChatGPT's error-flagging improved considerably with prompt sophistication"
    draft = _draft(_paragraph(sentence))
    writing_result = {
        "citation_links": [_clink(sentence, claim, "(Luo et al., 2025)", keys=("luo_2025",))]
    }
    report = _claim_report([_row(claim, "(Luo et al., 2025)", claim_sentence=sentence)])
    violations, _notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    assert len(violations) == 1
    assert "gold standard" in violations[0].detail


def test_check_sentence_covered_flags_a_leading_residue_ending_in_a_subordinator():
    """A leading residue ending in a subordinator ("because") is a
    dependent clause, never a bare subject, exactly like the coordinating-conjunction
    shape above."""
    sentence = (
        "Automated feedback cannot replace a teacher because Grammarly caught most "
        "surface errors (Koltovskaia, 2022)."
    )
    claim = "Grammarly caught most surface errors"
    draft = _draft(_paragraph(sentence))
    writing_result = {
        "citation_links": [_clink(sentence, claim, "(Koltovskaia, 2022)", keys=("koltovskaia_2022",))]
    }
    report = _claim_report([_row(claim, "(Koltovskaia, 2022)", claim_sentence=sentence)])
    violations, _notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    assert len(violations) == 1
    assert "cannot replace a teacher" in violations[0].detail


def test_check_sentence_covered_flags_an_over_budget_leading_residue_with_no_boundary_punctuation():
    """Unbounded on length, this branch also
    accepted a leading residue that ends in neither a coordinating conjunction nor a
    subordinator, as long as it was short enough to abut the verb with no punctuation
    at all -- this one carries five content tokens, over the same budget every other
    frame shape answers to."""
    sentence = (
        "While teachers remain indispensable to writing instruction Grammarly caught "
        "most surface errors (Koltovskaia, 2022)."
    )
    claim = "Grammarly caught most surface errors"
    draft = _draft(_paragraph(sentence))
    writing_result = {
        "citation_links": [_clink(sentence, claim, "(Koltovskaia, 2022)", keys=("koltovskaia_2022",))]
    }
    report = _claim_report([_row(claim, "(Koltovskaia, 2022)", claim_sentence=sentence)])
    violations, _notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    assert len(violations) == 1
    assert "indispensable" in violations[0].detail


def test_check_sentence_covered_flags_a_dedup_defect_proposition_with_no_verified_row():
    """The dedup-defect shape (design section 1, rows 28-30, 40): a citation-link
    proposition the writer reported has no matching final-report row at all -- the
    second half of rule 10."""
    sentence = (
        "Bonilla Lopez et al. (2018) reported that learners' cognitive-load estimates "
        "were significantly lower when processing direct corrections, while self-"
        "correcting with no feedback available imposed significantly lower load."
    )
    draft = _draft(_paragraph(sentence))
    writing_result = {
        "citation_links": [
            _clink(
                sentence,
                "learners' cognitive-load estimates were significantly lower when "
                "processing direct corrections",
                "(2018)",
            ),
            _clink(
                sentence,
                "self-correcting with no feedback available imposed significantly "
                "lower load",
                "(2018)",
            ),
        ]
    }
    report = _claim_report(
        [
            _row(
                "learners' cognitive-load estimates were significantly lower when "
                "processing direct corrections",
                "(2018)",
                claim_sentence=sentence,
            )
        ]
    )
    violations, _notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    unverified = [v for v in violations if "no verified row" in v.detail]
    assert len(unverified) == 1
    assert "self-correcting" in unverified[0].detail


def test_check_sentence_covered_returns_empty_without_writing_result_or_final_report():
    draft = _draft(_paragraph("Smith (2020) found an effect."))
    assert check_sentence_covered_by_verified_claims(draft, None, None, "s") == ([], [])
    assert check_sentence_covered_by_verified_claims(draft, {"citation_links": []}, None, "s") == (
        [],
        [],
    )


def test_check_comparative_meta_evaluation_flags_the_treated_most_directly_phrase():
    sentence = "Revision behaviour is treated most directly by Yallop et al. (2021)."
    draft = _draft(_paragraph(sentence))
    report = _claim_report([_row(sentence, "(2021)", claim_sentence=sentence)])
    violations = check_comparative_meta_evaluation(draft, report, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_COMPARATIVE_META_EVALUATION
    assert "treated most directly" in violations[0].detail


def test_check_comparative_meta_evaluation_flags_the_strongest_evidence_phrase():
    sentence = "The strongest direct evidence comes from Bonilla Lopez et al. (2018)."
    draft = _draft(_paragraph(sentence))
    report = _claim_report([_row(sentence, "(2018)", claim_sentence=sentence)])
    violations = check_comparative_meta_evaluation(draft, report, "s")
    assert len(violations) == 1
    assert "strongest direct evidence" in violations[0].detail


def test_check_comparative_meta_evaluation_reports_one_violation_per_row():
    """A row-at-a-time check: a match on `claim_sentence` short-circuits the same
    row's own `claim_text`."""
    sentence = "The strongest direct evidence comes from Bonilla Lopez et al. (2018)."
    draft = _draft(_paragraph(sentence))
    report = _claim_report(
        [_row("strongest direct evidence", "(2018)", claim_sentence=sentence)]
    )
    violations = check_comparative_meta_evaluation(draft, report, "s")
    assert len(violations) == 1


def test_check_comparative_meta_evaluation_passes_a_clean_sentence():
    sentence = "Recall improved dramatically compared with the baseline (Smith, 2020)."
    draft = _draft(_paragraph(sentence))
    report = _claim_report([_row(sentence, "(2020)", claim_sentence=sentence)])
    assert check_comparative_meta_evaluation(draft, report, "s") == []


# ---------------------------------------------------------------------------
# Rule 11's own lexicon was superlative-only and could not see a comparative
# ranking of one study above the rest -- a real delivered run carried two such
# sentences, one cited and one uncited, and neither tripped rule 11.
# ---------------------------------------------------------------------------


def test_check_comparative_meta_evaluation_flags_the_stronger_evidence_phrase():
    """An uncaught cited sentence, a bare comparative applied to a literature noun."""
    sentence = (
        "Li and Hébert (2023) provide stronger evidence: a paired t-test on 12 "
        "L2 writers showed paper ratings improved by a mean of 1.1 points."
    )
    draft = _draft(_paragraph(sentence))
    report = _claim_report([_row(sentence, "(2023)", claim_sentence=sentence)])
    violations = check_comparative_meta_evaluation(draft, report, "s")
    assert len(violations) == 1
    assert "stronger evidence" in violations[0].detail


def test_check_comparative_meta_evaluation_flags_the_most_controlled_comparison_phrase():
    """An uncaught uncited framing sentence, so only the delivered-draft half of the
    rule sees it -- no `claim_report` row names it."""
    sentence = "Experimental work offers the most controlled comparison."
    draft = _draft(_paragraph(sentence))
    report = _claim_report([])
    violations = check_comparative_meta_evaluation(draft, report, "s")
    assert len(violations) == 1
    assert "most controlled comparison" in violations[0].detail


@pytest.mark.parametrize(
    "sentence",
    [
        "This is the least convincing evidence available.",
        "This body of work offers more robust support.",
    ],
)
def test_check_comparative_meta_evaluation_flags_least_and_more_plus_adjective(sentence):
    draft = _draft(_paragraph(sentence))
    report = _claim_report([])
    violations = check_comparative_meta_evaluation(draft, report, "s")
    assert len(violations) == 1


#: At least fifteen ordinary comparative sentences about the STUDIED PHENOMENON, not
#: the literature itself, drawn verbatim from real delivered draft sentences --
#: identical to `backend/tests/test_sentence_coverage.py`'s own
#: ``ORDINARY_COMPARATIVE_SENTENCES_FROM_DELIVERED_DRAFTS`` list, so both copies of
#: the guard are checked against the same real-text grid.
ORDINARY_COMPARATIVE_SENTENCES_FROM_DELIVERED_DRAFTS = [
    "First, the field remains dominated by qualitative designs, and most naturalistic "
    "WCF studies are not theoretically motivated (Mao et al., 2024).",
    "Teacher feedback in the same case addressed more error types (16) than AWE "
    "feedback (8), though assignment differences may have contributed (Zhang and "
    "Hyland, 2018).",
    "Zhang and Hyland (2018) found that teacher feedback addressed more error types "
    "than AWE feedback, although assignment differences may have confounded this "
    "comparison.",
    "In the same integrated-feedback study, 82% of students approved of peer "
    "feedback as supportive and less critical, and 91% endorsed AWE as convenient "
    "and timely (Zhang & Hyland, 2021).",
    "Luo et al. (2025) found that ChatGPT's recall improved from 10% to 55% as "
    "prompts became more sophisticated, while precision stayed high (94–98%), "
    "exceeding Grammarly (85%).",
    "The study's placement testing placed participants at a lower-intermediate mean "
    "proficiency level (Bonilla López et al., 2018).",
    "Koltovskaia (2022) found that teachers using Grammarly did not substantially "
    "change their feedback practices, continuing to address both higher- and "
    "lower-order concerns rather than dividing labour with the tool.",
    "Teacher feedback tends to address more error types than AWE: Zhang and Hyland "
    "(2018) reported that teacher feedback covered 16 error types versus 8 for AWE.",
    "Yallop et al. (2021) found a higher relative distribution of useful justified "
    "comments (56%) than in participants' actual feedback comments (36%).",
    "This review synthesises studies drawn from Web of Science, Scopus, and "
    "Linguistics and Language Behavior Abstracts, covering literature published "
    "between 2002 and 2022, supplemented by more recent work through 2025.",
    "Written corrective feedback (WCF) remains a central concern in second-language "
    "(L2) writing research, and a persistent question is whether supplying the "
    "correct form (direct feedback) or merely locating or coding the error "
    "(indirect feedback) better serves accuracy, revision, and engagement.",
    "Mao et al. (2024) note that most naturalistic classroom studies are not "
    "theoretically motivated.",
    "A synthesis of 50 naturalistic classroom studies published between 2002 and "
    "2022 found that only 11 studies (22.0%) made explicit reference to theoretical "
    "tenets, and most adopted qualitative designs (66.0%), with mixed-methods "
    "approaches accounting for 22.6% (Mao et al., 2024).",
    "On revision behaviour and engagement, Zhang and Hyland (2018) show that "
    "engagement is a crucial mediating variable: one student resubmitted her essay "
    "13 times, with her score rising from 79 to 90, whereas another resubmitted "
    "only once and spent less than 14 minutes on the task.",
    "Liu and Wu (2019) found that lower-proficiency students lacked confidence in "
    "providing peer feedback or self-correcting.",
    "Yallop et al. (2021) found that among four Estonian doctoral writers, "
    "participants rated 93 of 99 segmented comments as useful or very useful, and "
    "that justified comments were more prevalent in perceived-useful feedback (56%) "
    "than in actual feedback (36%).",
    "Automated systems, however, remain limited: ChatGPT showed high precision but "
    "variable recall depending on prompt sophistication (Luo et al., 2025), and "
    "teacher feedback addressed more error types than automated feedback (Zhang & "
    "Hyland, 2018).",
]


@pytest.mark.parametrize(
    "sentence", ORDINARY_COMPARATIVE_SENTENCES_FROM_DELIVERED_DRAFTS
)
def test_check_comparative_meta_evaluation_does_not_flag_ordinary_comparatives(sentence):
    draft = _draft(_paragraph(sentence))
    report = _claim_report([])
    assert check_comparative_meta_evaluation(draft, report, "s") == []


# ---------------------------------------------------------------------------
# Rule 12: a delivered sentence opening with a citation subject immediately
# followed by a participle/gerund reporting verb with no finite verb before it.
# ---------------------------------------------------------------------------


def test_check_participle_after_attribution_flags_the_real_run_sentence():
    """A subject with no finite verb, only the present participle "cautioning"."""
    sentence = (
        "Teng and Ma (2024) cautioning that their scale may not fully reflect "
        "learners' feedback literacy in academic writing."
    )
    draft = _draft(_paragraph(sentence))
    violations = check_participle_after_attribution(draft, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_PARTICIPLE_AFTER_ATTRIBUTION
    assert violations[0].sentence == sentence


def test_check_participle_after_attribution_passes_the_fixed_sentence():
    sentence = (
        "Teng and Ma (2024) report that their scale may not fully reflect "
        "learners' feedback literacy in academic writing."
    )
    draft = _draft(_paragraph(sentence))
    assert check_participle_after_attribution(draft, "s") == []


def test_check_participle_after_attribution_passes_a_finite_verb_governing_the_participle():
    """A finite verb ("reported") governs the participle as its own object -- the
    participle no longer directly abuts the citation, so this is not the broken
    shape rule 12 exists to catch."""
    sentence = (
        "Teng and Ma (2024) reported cautioning that their scale may not fully "
        "reflect learners' feedback literacy in academic writing."
    )
    draft = _draft(_paragraph(sentence))
    assert check_participle_after_attribution(draft, "s") == []


def test_check_participle_after_attribution_passes_on_demo_expected():
    """Zero violations on every section of `demo/expected`'s own backing run
    directory -- `demo/expected` itself carries no per-section
    ``draft_content*.json`` at all, only the merged summary artefacts."""
    run_dir = skip_unless_run_dir(DEMO / "output" / "20260915-195906")
    protocol_files = load_section_files(run_dir, index=None)
    assert protocol_files.draft_content is not None
    assert check_participle_after_attribution(
        protocol_files.draft_content, protocol_files.name
    ) == []
    for index in discover_extra_section_indices(run_dir):
        files = load_section_files(run_dir, index=index)
        assert check_participle_after_attribution(files.draft_content, files.name) == []


# ---------------------------------------------------------------------------
# Rule 13 (`verb_initial_sentence`): a delivered sentence whose own first word is
# a reporting verb, with no subject before it at all.
# ---------------------------------------------------------------------------


def test_check_verb_initial_sentence_flags_the_defect_sentence():
    sentence = (
        "Showed that one student resubmitted her essay 13 times while another "
        "made one resubmission (Zhang & Hyland, 2018)."
    )
    draft = _draft(_paragraph(sentence))
    violations = check_verb_initial_sentence(draft, "s")
    assert len(violations) == 1
    assert violations[0].rule == RULE_VERB_INITIAL_SENTENCE
    assert violations[0].sentence == sentence


def test_check_verb_initial_sentence_is_case_insensitive():
    draft = _draft(_paragraph("showed that X happened (Author, 2020)."))
    assert len(check_verb_initial_sentence(draft, "s")) == 1


def test_check_verb_initial_sentence_passes_an_ordinary_sentence():
    draft = _draft(_paragraph("Zhang and Hyland (2018) showed that X happened."))
    assert check_verb_initial_sentence(draft, "s") == []


def test_check_verb_initial_sentence_passes_a_finite_verb_governed_by_a_citation():
    """This participle sentence opens with a citation subject, not a bare
    verb, which is rule 12's own business, not rule 13's."""
    sentence = (
        "Teng and Ma (2024) cautioning that their scale may not fully reflect "
        "learners' feedback literacy in academic writing."
    )
    draft = _draft(_paragraph(sentence))
    assert check_verb_initial_sentence(draft, "s") == []


def test_check_verb_initial_sentence_passes_on_demo_expected():
    """Zero violations on every section of `demo/expected`'s own backing run
    directory (tracked in git)."""
    run_dir = skip_unless_run_dir(DEMO / "output" / "20260915-195906")
    protocol_files = load_section_files(run_dir, index=None)
    assert protocol_files.draft_content is not None
    assert check_verb_initial_sentence(
        protocol_files.draft_content, protocol_files.name
    ) == []
    for index in discover_extra_section_indices(run_dir):
        files = load_section_files(run_dir, index=index)
        assert check_verb_initial_sentence(files.draft_content, files.name) == []


#: The narrowed rule 13 must never flag an ordinary sentence that merely happens
#: to open with a reporting verb in some shape other than "past/third-person verb
#: immediately followed by 'that'" -- identical to
#: `backend/tests/test_sentence_coverage.py`'s own ``COORDINATOR_NEGATIVE_
#: EXAMPLES`` list, so both copies of the guard are checked against the same
#: real-text grid.
COORDINATOR_NEGATIVE_EXAMPLES = [
    "Studies show that direct correction improved accuracy.",
    "The study shows that direct correction improved accuracy.",
    "Reports from teachers suggest direct correction improves accuracy.",
    "Findings indicate that direct correction improved accuracy.",
    "Research on feedback has shown that direct correction improved accuracy.",
]

#: Twelve ordinary sentence openings drawn verbatim from three delivered drafts
#: (`demo/output/20260915-195906`, `20260915-211700`,
#: `20260915-234219`) -- identical to `backend/tests/test_sentence_coverage.py`'s
#: own ``ORDINARY_OPENINGS_FROM_DELIVERED_DRAFTS`` list.
ORDINARY_OPENINGS_FROM_DELIVERED_DRAFTS = [
    "The question guiding this review is what the project library establishes "
    "about the effects of written corrective feedback (WCF) on second-language "
    "writing accuracy, revision behaviour, and engagement, and how those "
    "effects differ across teacher-provided, peer-provided, and automated "
    "sources.",
    "Comparative evidence on feedback forms is dominated by experimental work.",
    "Zhang and Hyland (2018) found that teacher feedback addressed 16 error "
    "types versus 8 for automated writing evaluation (AWE), though assignment "
    "differences may have confounded the comparison.",
    "Automated feedback is constrained by detection accuracy.",
    "A systematic search for the review identified 50 empirical studies "
    "meeting its inclusion criteria, revealing four major themes: teacher "
    "feedback practices in L2 writing classrooms, L2 learner responses to "
    "feedback, stakeholders' beliefs and perspectives about feedback, and "
    "feedback-related motivation and emotions (Mao et al., 2024).",
    "Research on written corrective feedback (WCF) in L2 writing has examined "
    "how learners revise after feedback, with attention to the form feedback "
    "takes and to the tools that deliver it.",
    "Evidence on automated feedback points to uneven error coverage and to "
    "prompt design as a moderator.",
    "However, Hyland (2025) reports that an AWE programme failed to spot many "
    "important errors, potentially misleading students about their "
    "grammatical correctness.",
    "Case-study evidence traces how learners actually act on feedback.",
    "Engagement functions as the mediating construct linking feedback source "
    "to outcomes, and the studies reviewed here address uptake, attention, "
    "and affective or motivational responses across teacher, peer, and "
    "automated feedback conditions.",
    "Hyland (2025) argues that such tools cannot judge rhetorical or "
    "pragmatic aspects of argument, and that learners' digital literacy "
    "shapes engagement regardless of proficiency.",
    "A mental-effort measure of cognitive load revealed that learners' "
    "cognitive load estimates were significantly lower when processing direct "
    "corrections targeting grammatical issues (Bonilla López et al., 2018).",
]


@pytest.mark.parametrize("sentence", COORDINATOR_NEGATIVE_EXAMPLES)
def test_check_verb_initial_sentence_false_for_coordinator_negative_examples(sentence):
    draft = _draft(_paragraph(sentence))
    assert check_verb_initial_sentence(draft, "s") == []


@pytest.mark.parametrize("sentence", ORDINARY_OPENINGS_FROM_DELIVERED_DRAFTS)
def test_check_verb_initial_sentence_false_for_ordinary_openings_from_runs_3_to_5(
    sentence,
):
    draft = _draft(_paragraph(sentence))
    assert check_verb_initial_sentence(draft, "s") == []


# ---------------------------------------------------------------------------
# Regression: rules 10/11 over the promoted run (design section 4) and over a
# hand-repaired fixture of its own excised output (design section 6.2, item 4).
# ---------------------------------------------------------------------------

PROMOTED_RUN = "20260912-210249"


def test_promoted_run_reproduces_exactly_15_coverage_flags_and_2_meta_eval_flags():
    """Design section 4: the rule run over the promoted run's 44 delivered rows flags
    exactly 15 distinct sentences (rule 10) and exactly rows 23 and 27 (rule 11)."""
    run_dir = skip_unless_run_dir(DEMO / "output" / PROMOTED_RUN)
    protocol = json.loads((DEMO / "protocol.json").read_text(encoding="utf-8"))
    result = check_run_directory(run_dir, protocol=protocol)

    coverage_sentences = {
        v.sentence for v in result.violations if v.rule == RULE_SENTENCE_EXCEEDS_VERIFIED_CLAIM
    }
    meta_sentences = {
        v.sentence for v in result.violations if v.rule == RULE_COMPARATIVE_META_EVALUATION
    }
    assert len(coverage_sentences) == 15
    assert len(meta_sentences) == 2
    assert any("underscoring the persistent value" in (s or "") for s in coverage_sentences)
    assert any("strongest direct evidence" in (s or "") for s in meta_sentences)
    assert any("is treated most directly" in (s or "") for s in meta_sentences)


def _repaired_section(sentence, claim_text, citation_text, keys=("k_2020",)):
    draft = _draft(_paragraph(sentence))
    writing_result = {"citation_links": [_clink(sentence, claim_text, citation_text, keys)]}
    report = _claim_report([_row(claim_text, citation_text, claim_sentence=sentence)])
    return draft, writing_result, report


REPAIRED_ROWS = [
    (
        "Liu and Wu (2019) add that 60% of students preferred teacher feedback.",
        "60% of students preferred teacher feedback",
        "(2019)",
    ),
    (
        "Li and Hebert (2023) found that peer feedback produced meaningful improvements "
        "in paper ratings between drafts (Cohen's d = 1.62).",
        "peer feedback produced meaningful improvements in paper ratings between drafts "
        "(Cohen's d = 1.62)",
        "(2023)",
    ),
    (
        "Bonilla Lopez et al. (2018) reported that learners' cognitive-load estimates "
        "were significantly lower when processing direct corrections targeting "
        "grammatical issues.",
        "learners' cognitive-load estimates were significantly lower when processing "
        "direct corrections targeting grammatical issues",
        "(2018)",
    ),
    (
        "Mao et al. (2024) reviewed 50 naturalistic classroom studies and identified "
        "learner responses to WCF as a major research strand.",
        "reviewed 50 naturalistic classroom studies and identified learner responses to "
        "WCF as a major research strand",
        "(2024)",
    ),
    (
        "Hyland (2025) notes that much feedback research has been conducted outside "
        "naturalistic learning contexts, raising questions about its usefulness to "
        "classroom teachers.",
        "much feedback research has been conducted outside naturalistic learning "
        "contexts, raising questions about its usefulness to classroom teachers",
        "(2025)",
    ),
    (
        "Affective engagement is likewise consequential: affect pervaded peer written "
        "exchanges and strongly shaped their effect (Yallop et al., 2021).",
        "affect pervaded peer written exchanges and strongly shaped their effect",
        "(Yallop et al., 2021)",
    ),
    (
        "This finding is correlational and based on self-report (Teng & Ma, 2024).",
        "This finding is correlational and based on self-report",
        "(Teng & Ma, 2024)",
    ),
]


@pytest.mark.parametrize("sentence,claim_text,citation_text", REPAIRED_ROWS)
def test_hand_repaired_excision_output_of_rows_1_11_28_29_30_40_42_has_no_violations(
    sentence, claim_text, citation_text
):
    """Rules 10 and 11 over a hand-written fixture of the excision output of rows 1,
    11, 28, 29, 30, 40, 42 give zero violations."""
    draft, writing_result, report = _repaired_section(sentence, claim_text, citation_text)
    coverage_violations, _notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    meta_violations = check_comparative_meta_evaluation(draft, report, "s")
    assert coverage_violations == []
    assert meta_violations == []


# ---------------------------------------------------------------------------
# "caution" and its inflections in `_RESIDUE_ATTRIB`.
# ---------------------------------------------------------------------------


def test_hyland_cautioned_that_sentence_has_no_violations():
    """Run `demo/output/20260915-195906/`, section 6: `loop_stats.
    coverage_incomplete_reasons` recorded this exact, fully verified sentence
    dropped whole for an unresolved residue `'cautioned that'` (start=14,
    before_first_claim=True, trailing_gap=' ') -- "cautioned" was absent from
    `_RESIDUE_ATTRIB`, even though "reports that"/"noted that" (already in it) would
    have covered the identical shape. Checked here exactly as `check_run_directory`
    checks a delivered draft: the whole, untrimmed sentence must produce zero rule
    10/11 violations now that "caution" is in the lexicon."""
    sentence = (
        "Hyland (2025) cautioned that automated programmes failed to identify many "
        "important errors, potentially misleading students about their grammatical "
        "accuracy."
    )
    proposition = (
        "automated programmes failed to identify many important errors, potentially "
        "misleading students about their grammatical accuracy"
    )
    draft, writing_result, report = _repaired_section(sentence, proposition, "(2025)")
    coverage_violations, _notices = check_sentence_covered_by_verified_claims(
        draft, writing_result, report, "s"
    )
    meta_violations = check_comparative_meta_evaluation(draft, report, "s")
    assert coverage_violations == []
    assert meta_violations == []


# ---------------------------------------------------------------------------
# The shared frame-classification fixture (design section 3.4): both this module's
# own residue/frame classifier and `app.services.sentence_coverage`'s copy must
# classify every span here the same way. This test checks this module's own copy;
# `backend/tests/test_sentence_coverage_frame_fixture.py` checks the other.
# ---------------------------------------------------------------------------


def test_frame_spans_fixture_matches_this_modules_own_classifier():
    """The fixture stores each span's own text and class, not the surrounding
    sentence, so only the two POSITION-INDEPENDENT classes are checked here: a
    non-frame residue (``class`` null -- content-bearing, or a never-frame/numeric/
    quantifier/over-budget term) stays non-frame regardless of position, and an F1
    (pure attribution, zero content tokens) reproduces with no position at all. F2/F3
    need the residue's own position in its sentence (a colon or comma-plus-opener
    right after it) -- covered instead by this module's own per-sentence tests above
    and by `backend/tests/test_sentence_coverage.py`'s equivalent cases."""
    from check_delivered import _residue_token_list

    fixture_path = DEMO / "fixtures" / "frame_spans.json"
    spans = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert spans, "expected the shared frame_spans.json fixture to be non-empty"
    checked = 0
    for entry in spans:
        if entry["class"] not in (None, "f1"):
            continue
        # An entry carrying its own position
        # (``start``/``before_first_claim``/``trailing_gap``) is what lets this
        # fixture reach the bare-leading-subject branch at all; most entries
        # carry none, which is exactly the input that branch never fires on.
        residue = _Residue(
            text=entry["text"],
            tokens=_residue_token_list(entry["text"]),
            before_first_claim=entry.get("before_first_claim", False),
            trailing_gap=entry.get("trailing_gap", ""),
            leading_gap="",
            start=entry.get("start", -1),
        )
        claim_tokens: set[str] = set()
        assert _is_frame(residue, claim_tokens) == entry["class"], entry
        checked += 1
    assert checked > 0


def _classify_checker_residues(sentence, claims, citations=None):
    """The checker's own copy of `test_sentence_coverage.py::_classify`: every
    residue of *sentence*, once covered by *claims*/*citations*, classified by this
    module's own `_is_frame`."""
    from check_delivered import _residue_token_list, _residues

    claim_tokens: set[str] = set()
    for c in claims:
        claim_tokens.update(_residue_token_list(c))
    return [
        (r.text, _is_frame(r, claim_tokens))
        for r in _residues(sentence, claims, citations)
    ]


def test_where_topic_shift_opener_is_frame_f3_in_the_checkers_own_copy():
    """The checker's own mirror of `test_sentence_coverage.py::
    test_where_topic_shift_opener_is_frame_f3`, over the same real removal
    (`writing_result_7.json -> loop_stats.coverage_incomplete_reasons`)."""
    sentence = (
        "Where peer comments focus is concerned, feedback in one peer-mediated "
        "study covered global aspects such as content, organization, and logic "
        "(Li & Hebert, 2023)."
    )
    claim = (
        "feedback in one peer-mediated study covered global aspects such as "
        "content, organization, and logic"
    )
    classes = _classify_checker_residues(sentence, [claim])
    frames = {text: cls for text, cls in classes}
    assert frames["Where peer comments focus is concerned"] == "f3"


def test_where_relative_clause_mid_sentence_is_not_an_opener_in_the_checkers_own_copy():
    """The checker's own mirror of `test_sentence_coverage.py::
    test_where_relative_clause_mid_sentence_is_not_an_opener`: the opener lists only
    ever grant F3 to a residue that is still ``before_first_claim``, never to one
    trailing an ordinary finding."""
    sentence = (
        "Zhang and Hyland (2018) found that engagement varied widely, where peer "
        "feedback occurred with no structure at all."
    )
    claim = "engagement varied widely"
    classes = _classify_checker_residues(sentence, [claim])
    frames = {text: cls for text, cls in classes}
    trailing = next(t for t in frames if t.strip().startswith("where"))
    assert frames[trailing] is None



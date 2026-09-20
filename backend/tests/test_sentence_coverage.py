"""Tests for ``app.services.sentence_coverage``.

Fixtures below are taken directly from a promoted run's own delivered rows,
so a change to the frame classifier's own budget or lexicon is
caught against real, already-adjudicated examples rather than invented ones.
"""
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

import pytest  # noqa: E402

from app.services.sentence_coverage import (  # noqa: E402
    apply_trailing_excision,
    coverage_incomplete,
    has_meta_evaluation,
    is_frame,
    is_heading_shaped_paragraph,
    opens_with_reporting_verb,
    residues,
    sentence_residue_violations,
    token_list,
    trailing_excision_cut,
)


def _classify(sentence: str, claims: list[str], citations: list[str] | None = None):
    claim_tokens: set[str] = set()
    for c in claims:
        claim_tokens.update(token_list(c))
    return [(r.text, is_frame(r, claim_tokens)) for r in residues(sentence, claims, citations)]


# --- residue / frame classification (design row 1, 2, 9, 13, 37) -----------------


def test_row1_shape_flags_the_evaluative_tail_as_non_frame():
    sentence = (
        "Liu and Wu (2019) add that 60% of students preferred teacher feedback, "
        "underscoring the persistent value of human judgement."
    )
    claim = "60% of students preferred teacher feedback"
    classes = _classify(sentence, [claim], ["(Liu and Wu, 2019)"])
    non_frame = [text for text, cls in classes if cls is None]
    assert any("underscoring" in t for t in non_frame)


def test_row2_shape_colon_lead_in_is_frame_f2():
    sentence = "This similarly documented selective uptake: teachers preferred global comments."
    claim = "teachers preferred global comments"
    classes = _classify(sentence, [claim])
    frames = {text: cls for text, cls in classes}
    lead_in = next(t for t in frames if "similarly documented" in t)
    assert frames[lead_in] == "f2"


def test_row13_shape_connector_opener_is_frame_f3():
    sentence = "On the automated side, Grammarly's precision exceeded human raters."
    claim = "Grammarly's precision exceeded human raters"
    classes = _classify(sentence, [claim])
    frames = {text: cls for text, cls in classes}
    assert frames["On the automated side"] == "f3"


def test_where_topic_shift_opener_is_frame_f3():
    """The real removal a later run recorded (`writing_result_7.json ->
    loop_stats.coverage_incomplete_reasons`): a topic-shift opener naming its own
    topic rather than a preposition or connective adverb, blocked only because
    "where" was missing from `DISCOURSE_OPENERS` -- its own leftover content
    ("comments", "focus", "concerned") sat within budget, and the sentence's own
    finding was otherwise fully verified."""
    sentence = (
        "Where peer comments focus is concerned, feedback in one peer-mediated "
        "study covered global aspects such as content, organization, and logic "
        "(Li & Hebert, 2023)."
    )
    claim = (
        "feedback in one peer-mediated study covered global aspects such as "
        "content, organization, and logic"
    )
    classes = _classify(sentence, [claim])
    frames = {text: cls for text, cls in classes}
    assert frames["Where peer comments focus is concerned"] == "f3"


def test_where_relative_clause_mid_sentence_is_not_an_opener():
    """The opener lists (`PREP_OPENERS`/`DISCOURSE_OPENERS`) only ever grant F3 to
    a residue that is still `before_first_claim` -- a LEADING residue, in practice
    the sentence's own opener -- never to one that follows the claim's own covered
    span. A "where" clause trailing an ordinary finding is not magically exempted
    just because "where" is now a recognised opener word: it is not before the
    claim at all, so the opener check never even runs on it."""
    sentence = (
        "Zhang and Hyland (2018) found that engagement varied widely, where peer "
        "feedback occurred with no structure at all."
    )
    claim = "engagement varied widely"
    classes = _classify(sentence, [claim])
    frames = {text: cls for text, cls in classes}
    trailing = next(t for t in frames if t.strip().startswith("where"))
    assert frames[trailing] is None


def test_notably_emphasis_opener_is_frame_f3():
    """A bare one-token emphasis-marker opener, comma-set, asserting nothing of
    its own, blocked only because "notably" was missing from `DISCOURSE_OPENERS`
    -- the same class of gap `test_where_topic_shift_opener_is_frame_f3` documents
    for "where", found again on this word
    (`writing_result_3.json -> loop_stats.coverage_incomplete_reasons`)."""
    sentence = (
        "Notably, students are not required to address all feedback received "
        "(Zhang & Hyland, 2021), and in peer review some learners selectively "
        "ignored unity- and word-use-related comments (Li & Hébert, 2023)."
    )
    claims = [
        "students are not required to address all feedback received",
        "in peer review some learners selectively ignored unity- and "
        "word-use-related comments",
    ]
    classes = _classify(
        sentence, claims, ["(Zhang & Hyland, 2021)", "(Li & Hébert, 2023)"]
    )
    frames = {text: cls for text, cls in classes}
    assert frames["Notably"] == "f3"


def test_notably_mid_sentence_is_not_an_opener():
    """Mirrors `test_where_relative_clause_mid_sentence_is_not_an_opener`: the
    opener lists only ever grant F3 to a residue that is still
    `before_first_claim` -- a leading residue, in practice the sentence's own
    opener -- never to one that follows the claim's own covered span. A trailing
    "notably" clause is not exempted just because "notably" is now a recognised
    opener word: it is not before the claim at all, so the opener check never
    even runs on it."""
    sentence = (
        "Zhang and Hyland (2018) found that engagement varied widely, notably "
        "diverging between the two focal learners."
    )
    claim = "engagement varied widely"
    classes = _classify(sentence, [claim])
    frames = {text: cls for text, cls in classes}
    trailing = next(t for t in frames if t.strip().startswith("notably"))
    assert frames[trailing] is None


def test_numeric_residue_is_never_a_frame_even_when_short():
    sentence = "Feedback improved outcomes for 57.1% of participants, a finding worth noting."
    claim = "Feedback improved outcomes for 57.1% of participants"
    classes = _classify(sentence, [claim])
    non_frame_texts = [t for t, cls in classes if cls is None]
    assert non_frame_texts  # the trailing "a finding worth noting" residue is flagged


def test_row37_shape_leading_study_design_residue_is_not_a_frame():
    sentence = (
        "In an ethnographic case study of four L2 doctoral students over three months, "
        "drawing on 15 drafts, 60 reviews and 12 revision plans, most feedback segments "
        "were visible revision comments (57.1%) (Yallop et al., 2021)."
    )
    claim = "most feedback segments were visible revision comments (57.1%)"
    non_frame, _frame = coverage_incomplete_helper(sentence, [claim], ["(Yallop et al., 2021)"])
    assert non_frame


def coverage_incomplete_helper(sentence, claims, citations):
    incomplete, spans = coverage_incomplete(sentence, claims, citations)
    return incomplete, spans


# --- numeral coverage relaxation: a residue's own numeral no longer disqualifies
# it outright when the same number, in either form, is verbatim in one of the
# sentence's own kept claims' evidence quotes (several delivered runs lost sentences shaped
# exactly like these). Everything else about the residue budget and frame classes
# is unchanged: the residue below still has to clear the three-content-token
# budget and the comma-plus-opener shape on its own merits once the numeral no
# longer disqualifies it by itself. ---------------------------------------------


def test_139_residue_is_kept_when_the_number_is_in_the_evidence_quote():
    sentence = (
        "In a study of 139 EFL writers, most preferred direct correction "
        "(Bonilla López et al., 2018)."
    )
    claim = "most preferred direct correction"
    non_frame, _frame = sentence_residue_violations(
        sentence, [claim], ["(Bonilla López et al., 2018)"],
        evidence_quotes=["A total of 139 low-intermediate EFL writers took part."],
    )
    assert non_frame == []


def test_139_residue_is_removed_when_the_number_is_in_no_evidence_quote():
    sentence = (
        "In a study of 139 EFL writers, most preferred direct correction "
        "(Bonilla López et al., 2018)."
    )
    claim = "most preferred direct correction"
    non_frame, _frame = sentence_residue_violations(
        sentence, [claim], ["(Bonilla López et al., 2018)"],
        evidence_quotes=["Learners reported a strong preference for explicit feedback."],
    )
    assert non_frame


def test_a_number_word_residue_matches_a_digit_form_evidence_quote():
    """"seventy" (the residue's own wording) and "70" (the evidence quote's own
    wording) normalise to the same key (`numeral_keys_in_text`), so a number word
    is covered by a digit-form quote of the same count and vice versa."""
    sentence = (
        "In a survey of seventy students, most preferred immediate feedback (Liu, 2019)."
    )
    claim = "most preferred immediate feedback"
    non_frame, _frame = sentence_residue_violations(
        sentence, [claim], ["(Liu, 2019)"],
        evidence_quotes=["70 students completed the end-of-term survey."],
    )
    assert non_frame == []


def test_a_number_word_residue_not_covered_by_any_evidence_quote_still_disqualifies():
    sentence = (
        "In a survey of seventy students, most preferred immediate feedback (Liu, 2019)."
    )
    claim = "most preferred immediate feedback"
    non_frame, _frame = sentence_residue_violations(
        sentence, [claim], ["(Liu, 2019)"],
        evidence_quotes=["Students reported high satisfaction with feedback timeliness."],
    )
    assert non_frame


def test_a_residue_with_two_numerals_stays_disqualified_when_only_one_is_covered():
    """Every numeral the residue itself carries must be covered, not just one --
    a residue naming two counts, only one of which any evidence quote confirms,
    is not exempted at all."""
    sentence = (
        "In a study of 139 writers across four sites, most preferred direct "
        "correction (X, 2018)."
    )
    claim = "most preferred direct correction"
    non_frame, _frame = sentence_residue_violations(
        sentence, [claim], ["(X, 2018)"],
        evidence_quotes=["139 writers took part in the study."],
    )
    assert non_frame


def test_is_frame_returns_f3_for_a_covered_numeral_lead_in():
    """Direct unit test of `is_frame` itself, not just its `sentence_residue_
    violations` caller: with no covered numerals at all (the default), the
    residue is disqualified outright; passing its own numeral's key lets it clear
    the same comma-plus-opener (F3) shape any other short lead-in does."""
    sentence = (
        "In a study of four doctoral writers, most segments were visible revision "
        "comments (Yallop et al., 2021)."
    )
    claim = "most segments were visible revision comments"
    claim_tokens = set(token_list(claim))
    lead_in = next(
        r for r in residues(sentence, [claim], ["(Yallop et al., 2021)"])
        if "four doctoral writers" in r.text
    )
    assert is_frame(lead_in, claim_tokens) is None
    assert is_frame(lead_in, claim_tokens, frozenset({"4"})) == "f3"


def test_supported_1_shape_bare_leading_subject_abutting_the_verb_is_frame_f1():
    """The demo protocol fixture ``supported-1`` (`demo/protocol.json`) verified, but the
    citation-link step's own proposition boundary began at the sentence's own finite
    verb rather than at its start, leaving "A systematic search for the review" as a
    leading residue with no comma or colon before the verb it belongs to. No
    deterministic cut can separate a bare subject from its own verb, so this must be
    treated as part of the covered clause rather than removed -- otherwise the
    sentence's claim verifies and the whole sentence is wrongly deleted
    (``sentences_removed_coverage_incomplete``) because the residue rule fails to
    recognise this shape."""
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
    incomplete, spans = coverage_incomplete(sentence, [claim], ["(Mao et al., 2024)"])
    assert incomplete is False
    assert spans == []
    classes = _classify(sentence, [claim], ["(Mao et al., 2024)"])
    leading = next(t for t, _cls in classes if "systematic search" in t)
    assert dict(classes)[leading] == "f1"


def test_a_leading_residue_ending_in_and_is_never_a_frame():
    """A bare-leading-subject branch with no upper bound on the
    residue's own length would wrongly accept a whole independent clause asserting
    unverified, evaluative material -- as long as it opens the sentence and abuts the
    covered proposition with no boundary punctuation. A residue that
    ends in a coordinating conjunction is never a bare subject: it is a full clause
    the writer joined to the next one, which is exactly what rule 10 exists to catch."""
    sentence = (
        "Teacher feedback remains the gold standard for second language writing "
        "instruction and ChatGPT's error-flagging improved considerably with prompt "
        "sophistication (Luo et al., 2025)."
    )
    claim = "ChatGPT's error-flagging improved considerably with prompt sophistication"
    classes = _classify(sentence, [claim], ["(Luo et al., 2025)"])
    leading = next(t for t, _cls in classes if "gold standard" in t)
    assert dict(classes)[leading] is None


def test_a_leading_residue_ending_in_but_is_never_a_frame():
    sentence = (
        "Automated tools are fundamentally unable to teach rhetorical awareness but "
        "Grammarly caught most surface errors (Koltovskaia, 2022)."
    )
    claim = "Grammarly caught most surface errors"
    classes = _classify(sentence, [claim], ["(Koltovskaia, 2022)"])
    leading = next(t for t, _cls in classes if "rhetorical awareness" in t)
    assert dict(classes)[leading] is None


def test_a_leading_residue_ending_in_because_is_never_a_frame():
    """A leading residue ending in a subordinator is a dependent
    clause the writer is attaching to what follows, never a bare subject, exactly like
    the coordinating-conjunction shape above."""
    sentence = (
        "Automated feedback cannot replace a teacher because Grammarly caught most "
        "surface errors (Koltovskaia, 2022)."
    )
    claim = "Grammarly caught most surface errors"
    classes = _classify(sentence, [claim], ["(Koltovskaia, 2022)"])
    leading = next(t for t, _cls in classes if "cannot replace a teacher" in t)
    assert dict(classes)[leading] is None


def test_a_short_subordinator_ending_residue_is_never_a_frame_within_budget():
    """The subordinator bound must fire on its own, not merely as a side effect of the
    content-token budget: this residue ("teachers", "resist", "because") holds exactly
    three content tokens, within the F2/F3 budget, so only the subordinator check
    itself keeps it from being accepted as a bare subject."""
    sentence = "Teachers resist because Grammarly caught most surface errors (Koltovskaia, 2022)."
    claim = "Grammarly caught most surface errors"
    classes = _classify(sentence, [claim], ["(Koltovskaia, 2022)"])
    leading = next(t for t, _cls in classes if "Teachers resist" in t)
    assert dict(classes)[leading] is None


def test_an_over_budget_leading_residue_is_never_a_frame_with_no_boundary_punctuation():
    """A branch unconditional on length for
    any residue not ending in a coordinating conjunction would still accept a whole
    independent clause that does not end in a subordinator either -- "While teachers
    remain indispensable to writing instruction" -- merely for opening the
    sentence and carrying no internal punctuation. The same three-content-token budget
    every other frame shape answers to applies here too."""
    sentence = (
        "While teachers remain indispensable to writing instruction Grammarly caught "
        "most surface errors (Koltovskaia, 2022)."
    )
    claim = "Grammarly caught most surface errors"
    classes = _classify(sentence, [claim], ["(Koltovskaia, 2022)"])
    leading = next(t for t, _cls in classes if "indispensable" in t)
    assert dict(classes)[leading] is None


def test_a_mid_span_subordinator_residue_is_capped_by_the_content_token_budget():
    """The second probe finding M2 names: a subordinator ("that") in the middle of the
    residue, not at its end, so only the content-token budget -- not the subordinator
    bound -- can catch it."""
    sentence = (
        "Research that has repeatedly demonstrated the limits of automated feedback "
        "over the past decade shows Grammarly caught most surface errors "
        "(Koltovskaia, 2022)."
    )
    claim = "Grammarly caught most surface errors"
    classes = _classify(sentence, [claim], ["(Koltovskaia, 2022)"])
    leading = next(t for t, _cls in classes if "repeatedly demonstrated" in t)
    assert dict(classes)[leading] is None


def test_a_leading_residue_with_content_words_and_a_comma_boundary_is_still_flagged():
    """The new bare-subject allowance is gated on there being NO clause-boundary
    punctuation at all between the residue and the verb it precedes; a leading residue
    that a writer set off with its own comma (rows 3, 17, 37, 39's own shape) keeps
    failing on the pre-existing content-token budget exactly as before."""
    sentence = (
        "In a small ethnographic case study of four L2 PhD students, engagement "
        "with feedback varied widely (Lopez, 2019)."
    )
    claim = "engagement with feedback varied widely"
    non_frame, _frame = coverage_incomplete_helper(sentence, [claim], ["(Lopez, 2019)"])
    assert non_frame


def test_clean_sentence_is_fully_covered():
    sentence = "Smith (2020) found that recall improved by 12 points."
    claim = "recall improved by 12 points"
    incomplete, spans = coverage_incomplete(sentence, [claim], ["(2020)"])
    assert incomplete is False
    assert spans == []


# --- comparative meta-evaluation guard (rows 23, 27) ------------------------------


def test_row23_shape_is_treated_most_directly_is_flagged():
    sentence = "Revision behaviour is treated most directly by Yallop et al. (2021)."
    assert has_meta_evaluation(sentence) is True


def test_row27_shape_strongest_direct_evidence_is_flagged():
    sentence = "The strongest direct evidence comes from Bonilla Lopez et al. (2018)."
    assert has_meta_evaluation(sentence) is True


def test_accepted_sentence_is_not_flagged_by_meta_evaluation_guard():
    sentence = "Recall improved dramatically compared with the baseline condition (Smith, 2020)."
    assert has_meta_evaluation(sentence) is False


def test_plain_superlative_two_words_from_a_literature_noun_is_flagged():
    assert has_meta_evaluation("This is the clearest supporting study available.") is True


# --- Comparative evaluations of the literature -- the guard's own lexicon was
# superlative-only ("strongest evidence") and could not see a comparative ranking of
# one study above the rest ("Li and Hébert (2023) provide stronger evidence: ...",
# cited; "Experimental work offers the most controlled comparison.", uncited). ------


def test_bare_comparative_stronger_evidence_is_flagged():
    """An uncaught cited sentence, a bare comparative applied to a literature noun."""
    sentence = (
        "Li and Hébert (2023) provide stronger evidence: a paired t-test on 12 "
        "L2 writers showed paper ratings improved by a mean of 1.1 points."
    )
    assert has_meta_evaluation(sentence) is True


def test_most_plus_adjective_controlled_comparison_is_flagged():
    """An uncaught uncited sentence, "most" plus an adjective applied to "comparison"."""
    sentence = "Experimental work offers the most controlled comparison."
    assert has_meta_evaluation(sentence) is True


def test_least_plus_adjective_evidence_is_flagged():
    """The symmetric "least" case alongside "most"."""
    assert has_meta_evaluation("This is the least convincing evidence available.") is True


def test_more_plus_adjective_support_is_flagged():
    assert has_meta_evaluation("This body of work offers more robust support.") is True


#: At least fifteen ordinary comparative sentences about the STUDIED PHENOMENON, not
#: the literature itself, drawn verbatim from real delivered draft sentences -- none
#: of these carries a comparative evaluation of the literature, and the guard must
#: never fire on any of them.
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
def test_ordinary_comparative_about_the_studied_phenomenon_is_not_flagged(sentence):
    assert has_meta_evaluation(sentence) is False


# --- deterministic trailing excision (design edit 6, outcome 2) ------------------


def test_row1_shape_trailing_excision_cuts_at_the_comma():
    sentence = (
        "Liu and Wu (2019) add that 60% of students preferred teacher feedback, "
        "underscoring the persistent value of human judgement."
    )
    claim = "60% of students preferred teacher feedback"
    cut = trailing_excision_cut(sentence, [claim], ["(Liu and Wu, 2019)"])
    assert cut is not None
    excised = apply_trailing_excision(sentence, cut)
    assert excised == "Liu and Wu (2019) add that 60% of students preferred teacher feedback."


def test_row40_shape_trailing_excision_keeps_the_first_clause_and_its_citation():
    sentence = (
        "Affective engagement is likewise consequential: affect pervaded peer written "
        "exchanges and strongly shaped their effect (Yallop et al., 2021), though explicit "
        "praise appeared in only about 8% of comments (Yallop et al., 2021)."
    )
    claim = "affect pervaded peer written exchanges and strongly shaped their effect"
    # Only the verified proposition's own citation is passed -- the gate never treats
    # the unverified second proposition's own (identically rendered) citation as
    # covering text, even though the sentence renders the same citation string twice.
    cut = trailing_excision_cut(sentence, [claim], ["(Yallop et al., 2021)"])
    assert cut is not None
    excised = apply_trailing_excision(sentence, cut)
    assert excised == (
        "Affective engagement is likewise consequential: affect pervaded peer written "
        "exchanges and strongly shaped their effect (Yallop et al., 2021)."
    )


def test_row38_shape_leading_residue_has_no_safe_trailing_cut():
    """Row 38's own residue is the sentence's main clause -- a leading span, not a
    trailing one -- so no deterministic excision exists; the sentence must be removed
    whole instead (design edit 6, outcome 3)."""
    sentence = (
        "This suggests that the uptake of peer feedback depends less on its volume than "
        "on its discursive type, an insight consistent with evidence that students value "
        "non-corrective over corrective feedback (Yallop et al., 2021)."
    )
    claim = (
        "an insight consistent with evidence that students value non-corrective over "
        "corrective feedback"
    )
    cut = trailing_excision_cut(sentence, [claim], ["(Yallop et al., 2021)"])
    assert cut is None


def test_row17_shape_multi_residue_sentence_has_no_safe_trailing_cut():
    """Two non-frame residues (a leading study-design clause and, if present, another
    span) never license a trailing cut -- only exactly one, trailing, residue does."""
    sentence = (
        "In an ethnographic case study of four L2 PhD students, participants seldom "
        "critically engaged with local comments, an unrelated closing remark about "
        "methodology follows here."
    )
    claim = "participants seldom critically engaged with local comments"
    cut = trailing_excision_cut(sentence, [claim])
    assert cut is None


def test_trailing_excision_refuses_a_cut_with_no_verified_prefix():
    sentence = "This is entirely unverified content, with a trailing clause too."
    cut = trailing_excision_cut(sentence, [])
    assert cut is None


# --- "caution" and its inflections were missing from ATTRIB/REPORTING_VERBS ----


def test_hyland_cautioned_that_lead_in_is_covered():
    """Run `demo/output/20260915-195906/`, section 6: `loop_stats.
    coverage_incomplete_reasons` recorded this exact sentence dropped whole for
    ``unresolved residue Residue(text='cautioned that', tokens=['cautioned', 'that'],
    start=14, end=28, before_first_claim=True, is_trailing=False, trailing_gap=' ',
    leading_gap=') ')`` -- "cautioned" was absent from `ATTRIB`, so its own residue
    "cautioned that" was a content-bearing, non-frame residue, even though "reports
    that" and "noted that" (already in `ATTRIB`) would have covered the identical
    shape. The whole proposition after "cautioned that" had already verified in
    full; re-running this module's own coverage check on the real sentence and its
    real verified proposition must find no residue at all now that "caution" is in
    the lexicon."""
    sentence = (
        "Hyland (2025) cautioned that automated programmes failed to identify many "
        "important errors, potentially misleading students about their grammatical "
        "accuracy."
    )
    proposition = (
        "automated programmes failed to identify many important errors, potentially "
        "misleading students about their grammatical accuracy"
    )
    incomplete, spans = coverage_incomplete(sentence, [proposition])
    assert incomplete is False
    assert spans == []


# --- "concede" was missing from ATTRIB/REPORTING_VERBS ---------------------------


def test_liu_wu_conceded_that_lead_in_is_covered():
    """A fully verified sentence dropped whole for an unresolved residue "Yet the
    authors conceded that" (start=0, before_first_claim=True, trailing_gap=' ') --
    "conceded" was absent from `ATTRIB`, the same lexicon-gap shape "caution" closed
    for earlier, on a verb that fix did not add. Re-running this module's own
    coverage check on the real sentence and its real verified proposition must find
    no residue at all now that "concede" is in the lexicon."""
    sentence = (
        "Yet the authors conceded that their non-randomised, descriptive design is "
        "necessarily descriptive (Liu and Wu, 2019)."
    )
    proposition = (
        "their non-randomised, descriptive design is necessarily descriptive"
    )
    incomplete, spans = coverage_incomplete(sentence, [proposition], ["(Liu and Wu, 2019)"])
    assert incomplete is False
    assert spans == []


# --- `opens_with_reporting_verb` (rule 13's own shared predicate) ----------------
#
# A lead-in blocked from a permitted frame by a numeral or an out-of-lexicon noun
# is never rewritten (the pipeline delivers only what is verified, or omits the
# sentence); this predicate instead catches a different, independent risk: a
# residue whose own content is entirely zero (a bare "showed that"/"reports
# that") already passes the coverage decision as a permitted frame with no regard
# for whether a grammatical subject actually precedes the verb. A sentence
# delivered in exactly that shape, with its own citation trailing at the end
# rather than sitting in front of the verb, has no subject at all: "Showed that
# one student resubmitted her essay 13 times while another made one resubmission
# (Zhang & Hyland, 2018)."

DEFECT_SENTENCE = (
    "Showed that one student resubmitted her essay 13 times while another made "
    "one resubmission (Zhang & Hyland, 2018)."
)


def test_opens_with_reporting_verb_flags_the_defect_sentence():
    assert opens_with_reporting_verb(DEFECT_SENTENCE) is True


def test_opens_with_reporting_verb_is_case_insensitive():
    assert opens_with_reporting_verb("showed that X happened.") is True
    assert opens_with_reporting_verb("SHOWED that X happened.") is True


def test_opens_with_reporting_verb_flags_other_past_and_third_person_inflections():
    assert opens_with_reporting_verb("Found that X happened (Author, 2020).") is True
    assert opens_with_reporting_verb("Reports that X happened (Author, 2020).") is True
    assert opens_with_reporting_verb("Noted that X happened (Author, 2020).") is True


def test_opens_with_reporting_verb_false_for_the_base_form():
    """The base form ("show", not "showed"/"shows") is deliberately excluded: it
    is not a shape any part of the pipeline produces."""
    assert opens_with_reporting_verb("Show that X happened (Author, 2020).") is False


def test_opens_with_reporting_verb_false_for_the_gerund_form():
    assert (
        opens_with_reporting_verb("Cautioning that X happened (Author, 2020).")
        is False
    )


def test_opens_with_reporting_verb_false_for_a_sentence_with_a_citation_subject():
    assert (
        opens_with_reporting_verb("Zhang and Hyland (2018) showed that X happened.")
        is False
    )


def test_opens_with_reporting_verb_false_for_an_empty_sentence():
    assert opens_with_reporting_verb("") is False


#: The narrowed rule 13 must never flag an ordinary sentence that merely happens
#: to open with a reporting verb in some shape other than "past/third-person verb
#: immediately followed by 'that'".
COORDINATOR_NEGATIVE_EXAMPLES = [
    "Studies show that direct correction improved accuracy.",
    "The study shows that direct correction improved accuracy.",
    "Reports from teachers suggest direct correction improves accuracy.",
    "Findings indicate that direct correction improved accuracy.",
    "Research on feedback has shown that direct correction improved accuracy.",
]

#: Twelve ordinary sentence openings drawn verbatim from three delivered drafts
#: (`demo/output/20260915-195906`, `20260915-211700`,
#: `20260915-234219`) -- none of them is this shape, including the ones that open
#: with a citation, a reporting-verb-adjacent noun ("Research"), or a discourse
#: connective.
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
def test_opens_with_reporting_verb_false_for_coordinator_negative_examples(sentence):
    assert opens_with_reporting_verb(sentence) is False


@pytest.mark.parametrize("sentence", ORDINARY_OPENINGS_FROM_DELIVERED_DRAFTS)
def test_opens_with_reporting_verb_false_for_ordinary_openings_from_runs_3_to_5(
    sentence,
):
    assert opens_with_reporting_verb(sentence) is False


# --- is_heading_shaped_paragraph ---------------------------------------------------


def test_is_heading_shaped_paragraph_flags_the_bare_sub_heading_fragment():
    """The real shape an independent audit found: the writer's own sub-heading
    ("## Comparative Scope and Accuracy") survives finalize with its markdown marker
    already gone, once its intro paragraph is removed, and is delivered as a bare
    noun phrase -- no verb, no subject, no citation."""
    assert is_heading_shaped_paragraph("Comparative Scope and Accuracy") is True


def test_is_heading_shaped_paragraph_false_for_ordinary_short_sentences():
    for sentence in (
        "Uptake was uneven.",
        "Findings diverge.",
        "In contrast, results differed (Mao et al., 2024).",
    ):
        assert is_heading_shaped_paragraph(sentence) is False


def test_is_heading_shaped_paragraph_false_for_the_protocol_fixture_sentences():
    supported = (
        "A systematic search for the review identified 50 empirical studies "
        "meeting its inclusion criteria, revealing four major themes: teacher "
        "feedback practices in L2 writing classrooms, L2 learner responses to "
        "feedback, stakeholders' beliefs and perspectives about feedback, and "
        "feedback-related motivation and emotions (Mao et al., 2024)."
    )
    unsupported = (
        "Screening yielded the same 50 eligible studies for the review, sorted "
        "into two major themes: writing accuracy outcomes and learner motivation "
        "only (Mao et al., 2024)."
    )
    assert is_heading_shaped_paragraph(supported) is False
    assert is_heading_shaped_paragraph(unsupported) is False


def test_is_heading_shaped_paragraph_false_for_a_long_title_case_phrase():
    """The ten-word budget excludes a longer title-case run, even one with no verb
    and no citation, from this narrow guard -- it is not a general title-case
    detector, only a bound on how much unverified prose the guard may remove."""
    long_phrase = (
        "Automated Written Corrective Feedback And Second Language Writing "
        "Development And Overview"
    )
    assert len(long_phrase.split()) > 10
    assert is_heading_shaped_paragraph(long_phrase) is False


def test_is_heading_shaped_paragraph_false_for_a_verb_bearing_lowercase_fragment():
    """A short, lower-case, unpunctuated fragment that DOES carry a finite verb --
    neither title case nor free of a lexicon verb -- is not heading-shaped: it reads
    as a truncated sentence, a different, already-handled defect class."""
    assert is_heading_shaped_paragraph("results were mixed across studies") is False


def test_is_heading_shaped_paragraph_true_for_a_title_case_phrase_with_a_lexicon_word():
    """Title case overrides a stray lexicon match: "Studies Show Promise" contains
    "show" (a reporting verb), but every content word is capitalised, so this reads
    as a heading, not a sentence."""
    assert is_heading_shaped_paragraph("Studies Show Promise") is True


def test_is_heading_shaped_paragraph_false_for_empty_text():
    assert is_heading_shaped_paragraph("") is False
    assert is_heading_shaped_paragraph("   ") is False

"""Code guards: normalisation
helpers, and guards 0-3 in ``app.services.fulltext``. Every guard is unit tested in
isolation (independent of the DB-backed ``_verify_claims`` pipeline, which is covered by
``test_claim_verification_service.py``).
"""

from __future__ import annotations

import pytest

# --------------------------------------------------------------------------------------
# _normalise_for_match (M1)
# --------------------------------------------------------------------------------------


def test_normalise_for_match_folds_curly_quotes_dashes_and_nbsp():
    """The fold keeps only letters
    and digits, case-folded: every quote mark, dash, nbsp and other punctuation character
    vanishes identically, rather than being translated to an ASCII look-alike."""
    from app.services.fulltext import _normalise_for_match

    text = "“The effect’s size” was 12–" + " 15 (large)."
    normalised = _normalise_for_match(text)
    assert normalised == "theeffectssizewas1215large"


def test_normalise_for_match_collapses_whitespace_and_strips():
    """Whitespace is removed
    entirely, not collapsed to a single space (one more consequence of "letters and digits
    only")."""
    from app.services.fulltext import _normalise_for_match

    assert _normalise_for_match("  a\n  b\t c  ") == "abc"
    assert _normalise_for_match("") == ""


# --------------------------------------------------------------------------------------
# A pymupdf line-break
# hyphenation ("post-\ntest") survives as "post- test" (hyphen, one space) through every
# fold above, which is a substring of neither a model's joined quote ("posttest") nor its
# contiguously hyphenated one ("post-test"). Without this fold, `_guard_quote_fidelity`
# would misread this as `quote_not_verbatim` on an otherwise correct `verified` answer.
# --------------------------------------------------------------------------------------


def test_normalise_for_match_folds_a_letter_hyphen_linebreak_to_the_joined_form():
    from app.services.fulltext import _normalise_for_match

    # "word- word" (line break), "word-word" (contiguous) and "wordword" (joined) all
    # normalise to the identical form.
    assert (
        _normalise_for_match("post- test")
        == _normalise_for_match("post-test")
        == _normalise_for_match("posttest")
        == "posttest"
    )


def test_normalise_for_match_no_longer_spares_a_digit_adjacent_hyphen():
    """The "letters and digits only" rule vanishes every hyphen, digit-adjacent or not (a
    numeral range such as "12-15", or a hyphen-fused identifier such as "CTLA-4"). This is
    a deliberate trade documented on `_keep_letters_and_digits`: `_quote_segments`/
    `_guard_attribution` only ever compare a real (30+ raw character) span of prose, never
    a bare numeral or identifier standing alone, so the risk of a coincidental unrelated
    match is negligible, while "CTLA-4" and "CTLA4" (two real spellings of the same
    identifier) compare equal."""
    from app.services.fulltext import _normalise_for_match

    assert _normalise_for_match("12-15") == "1215"
    assert _normalise_for_match("12- 15") == "1215"
    assert _normalise_for_match("CTLA-4") == "ctla4"
    assert _normalise_for_match("CTLA-4") == _normalise_for_match("CTLA4")


def test_guard_quote_fidelity_does_not_fire_on_hss_verbatim_01s_line_break_hyphenation():
    """Reproduces the ``hss-verbatim-01`` miss: the
    cached text hyphenates "post-test" at a line break, the model quoted the contiguous
    spelling, and without this fold the guard would demote a correct ``verified`` answer."""
    from app.services.fulltext import _guard_quote_fidelity

    chunk_text = (
        "Given the control group, there is a statistically significant difference "
        "between post-\ntest and pre-test overall scores (z=3.297, p=000<.0167). In "
        "other words, the traditional instruction made a significant difference by "
        "leading to an increase in the overall scores."
    )
    quote = (
        "there is a statistically significant difference between post-test and "
        "pre-test overall scores (z=3.297, p=000<.0167). In other words, the "
        "traditional instruction made a significant difference by leading to an "
        "increase in the overall scores."
    )
    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=quote, chunk_texts=[chunk_text]
    )
    assert status == "verified"
    assert reasons == []


def test_guard_quote_fidelity_does_not_fire_on_hss_paraphrase_04s_line_break_hyphenation():
    """Reproduces the ``hss-paraphrase-04`` run-B miss: the cached text hyphenates
    "integration" at a line break ("inte-\\ngration"), the model quoted the fully joined
    spelling with no hyphen at all."""
    from app.services.fulltext import _guard_quote_fidelity

    chunk_text = (
        "(H2) University teachers' information and data literacy can positively "
        "predict their subjective norms toward ICT inte-\ngration. (H3) University "
        "teachers' information and data literacy can positively predict their "
        "perceived behavioral control toward ICT integration."
    )
    quote = (
        "(H2) University teachers' information and data literacy can positively "
        "predict their subjective norms toward ICT integration."
    )
    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=quote, chunk_texts=[chunk_text]
    )
    assert status == "verified"
    assert reasons == []


def test_guard_quote_fidelity_does_not_fire_on_real_test_12s_line_break_hyphenation():
    """Reproduces the ``real-test-12`` miss: the cached text hyphenates "problem-solving"
    at a line break, the model quoted the contiguous spelling."""
    from app.services.fulltext import _guard_quote_fidelity

    chunk_text = (
        "Participants viewed a set of photographs showing problem-\nsolving "
        "situations in eighth grade mathematics education. The photographs were "
        "video stills."
    )
    quote = (
        "Participants viewed a set of photographs showing problem-solving "
        "situations in eighth grade mathematics education."
    )
    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=quote, chunk_texts=[chunk_text]
    )
    assert status == "verified"
    assert reasons == []


def test_guard_quote_fidelity_does_not_fire_on_real_test_20s_line_break_hyphenations():
    """Reproduces the ``real-test-20`` miss: two independent line-break hyphenations in
    the same cached paper ("perspec-\\ntive" and "Indian-\\nIranian"), across two of the
    model's four evidence spans -- every span of every quote must still be verbatim."""
    from app.services.fulltext import _guard_quote_fidelity

    chunk_texts = [
        "Examining a nine-year-long FLP in retrospect (diachronic perspective), and "
        "its implications for the child's linguistic development at present "
        "(synchronic perspec-\ntive), we seek to show that translingual practices "
        "in multilingual families...",
        "the life trajectory of a family from the Global South, an Indian-\n"
        "Iranian transnational family with a nine-year-old daughter, we seek to "
        "illustrate the complexities in FLPs in this transnational family.",
    ]
    quotes = [
        "Examining a nine-year-long FLP in retrospect (diachronic perspective), and "
        "its implications for the child's linguistic development at present "
        "(synchronic perspective)",
        "an Indian-Iranian transnational family with a nine-year-old daughter",
    ]
    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=None, evidence_quotes=quotes, chunk_texts=chunk_texts
    )
    assert status == "verified"
    assert reasons == []


# --------------------------------------------------------------------------------------
# The general "letters and
# digits only" canonical form recovers two acceptance misses a narrower
# letter-hyphen-linebreak fold does not reach. Both fixtures below use the real chunk text
# (`evaluation/claims/data/real_claims_fulltext/`, `data/hss_dev_fulltext/`, gitignored,
# confirmed present locally) and the real committed evidence quote
# (`evaluation/claims/results/v3/real_run{A,B}.jsonl`,
# `results/dev/v3-iter4/hss_runA.jsonl`).
# --------------------------------------------------------------------------------------


def test_guard_quote_fidelity_does_not_fire_on_real_test_13s_stray_equals_byte():
    """Reproduces the ``real-test-13`` miss: the cached PDF text substitutes a stray
    control byte (0x01) for what should be an "=" sign in "(r=0.28; p=0.00)" -- pymupdf's
    own extraction artefact, confirmed against the real cached bytes
    (`data/real_claims_fulltext/10-3402_rlt-v20i0-14430.json`), not merely a stylistic
    choice by the source PDF. The model quoted the ordinary, correctly-spelled "=" form.
    Neither an "=" sign nor a stray control byte is a letter or a digit, so both vanish
    identically under the new canonical form."""
    from app.services.fulltext import _guard_quote_fidelity

    chunk_text = (
        "grade and his or her downloading lecture slides (r\x010.28; p\x010.00) or "
        "accessing\ntutorial resources (r\x010.31; p\x010.00) and/or posting to the "
        "content forums."
    )
    quote = (
        "his or her downloading lecture slides (r=0.28; p=0.00) or accessing "
        "tutorial resources (r=0.31; p=0.00) and/or posting to the content forums."
    )
    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=quote, chunk_texts=[chunk_text]
    )
    assert status == "verified"
    assert reasons == []


def test_guard_quote_fidelity_does_not_see_a_flipped_sign_or_inequality_after_the_fold():
    """Documents a known, deliberately unfixed limitation of the same fold that fixes
    `real-test-13` above: removing every
    non-letter/non-digit character also removes a minus sign and a comparison operator
    ("=", "<", ">") standing between two numerals, not only the "=" sign or stray control
    byte `real-test-13` needed folded away. "r = -0.28; p < 0.05" and "r = 0.28; p = 0.05"
    both fold to "r028p005", so a quote that flips a reported correlation's sign, or quotes
    a "<" as "=", is indistinguishable from the source under this comparison and the guard
    verifies it. This is a missed demotion, not a false one (the guard only ever makes a
    status stricter), and is of the same kind as the injected running-header/page-number
    limitation `test_guard_quote_fidelity_still_fires_on_an_injected_page_number_inside_a_
    quote` pins below -- recorded here so a later change cannot silently "fix" or worsen it
    without this test forcing a decision."""
    from app.services.fulltext import _guard_quote_fidelity

    chunk_text = (
        "Test anxiety was negatively associated with grade (r = -0.28; p < 0.05) in the "
        "cohort of high schoolers surveyed."
    )
    quote = (
        "Test anxiety was negatively associated with grade (r = 0.28; p = 0.05) in the "
        "cohort of high schoolers surveyed."
    )
    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=quote, chunk_texts=[chunk_text]
    )
    assert status == "verified"
    assert reasons == []


def test_guard_quote_fidelity_does_not_fire_on_hss_verbatim_03s_ligature_split():
    """Reproduces the ``hss-verbatim-03`` (dev) miss: pymupdf extracts the "fi" ligature in
    "findings" as the two-character sequence U+FB01 ("ﬁ") followed by an injected
    space before the rest of the word ("The ﬁ ndings"), confirmed against the real
    cached text (`data/hss_dev_fulltext/10-17323_jle-2021-10737.json`). NFKC already folds
    the ligature to the letters "f" and "i" (unchanged by this fix), but only removing
    *all* whitespace, not merely collapsing it to one space, removes the space the
    ligature-splitting artefact leaves between "fi" and "ndings". The model quoted the
    word correctly spelled and joined ("findings"), with a straight apostrophe where the
    source carries a curly one (already folded away, independently of this fix)."""
    from app.services.fulltext import _guard_quote_fidelity

    chunk_text = (
        "cluded from consideration for this study. The ﬁ ndings showed that \n"
        "the environmental factor was the primary factor for learners’ poor OECS "
        "performance in EFL \ncontexts."
    )
    quote = (
        "The findings showed that the environmental factor was the primary factor "
        "for learners' poor OECS performance in EFL contexts."
    )
    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=quote, chunk_texts=[chunk_text]
    )
    assert status == "verified"
    assert reasons == []


def test_guard_quote_fidelity_still_fires_on_an_injected_page_number_inside_a_quote():
    """Documents a known, deliberately unfixed limitation (`real-test-18`): a page break
    pymupdf injects mid-sentence ("...considerable benefit to
    the \\n\\n3 \\n \\n \\ndevelopment of child language...", confirmed against the real
    cached text, `data/real_claims_fulltext/10-1111_jcpp-12352.json`) carries a real digit
    ("3", a page number) and real letters (from a running header, in the sibling
    `hss-verbatim-02` case) -- neither is punctuation or whitespace, so "letters and digits
    only" cannot remove either one. This is cache noise the guard cannot and should not try
    to bridge: a page number or running header genuinely is not part of the quoted
    sentence, and folding digits/letters away generally to accommodate it would erase real
    evidentiary content elsewhere."""
    from app.services.fulltext import _guard_quote_fidelity

    chunk_text = (
        "South African sample was shown to be of considerable benefit to the \n\n3 \n \n \n"
        "development of child language and focussed attention."
    )
    quote = (
        "South African sample was shown to be of considerable benefit to the "
        "development of child language and focussed attention."
    )
    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=quote, chunk_texts=[chunk_text]
    )
    assert status == "needs_nuance"
    assert reasons == ["quote_not_verbatim"]


# --------------------------------------------------------------------------------------
# _quote_segments (M1)
# --------------------------------------------------------------------------------------


def test_quote_segments_splits_double_quoted_fragments_of_10_or_more_chars():
    from app.services.fulltext import _quote_segments

    quote = (
        '"Rather than seeking to identify the most effective type" ... '
        '"was found to bring benefits to both teachers and students"'
    )
    segments = _quote_segments(quote)
    assert len(segments) == 2
    assert segments[0].startswith("Rather than")
    assert segments[1].startswith("was found")


def test_quote_segments_strips_leading_and_trailing_ellipsis():
    from app.services.fulltext import _quote_segments

    segments = _quote_segments("...the effect was statistically significant overall...")
    assert segments == ["the effect was statistically significant overall"]


def test_quote_segments_drops_short_quoted_fragments_under_30_normalised_chars():
    from app.services.fulltext import _quote_segments

    quote = '"a claim the paper never actually makes" ... "nor this one"'
    segments = _quote_segments(quote)
    assert segments == ["a claim the paper never actually makes"]


def test_quote_segments_falls_back_to_the_raw_quote_when_unquoted():
    from app.services.fulltext import _quote_segments

    assert _quote_segments("the effect was statistically significant") == [
        "the effect was statistically significant"
    ]


def test_quote_segments_splits_typographic_double_quoted_fragments():
    """The fragment extractor ran on the RAW quote
    with an ASCII-only regex while normalisation (which folds curly to straight quotes)
    happened afterwards, so a model quote wrapped in typographic double quotes (U+201C/
    U+201D) found no fragment, fell back to the whole raw string, and that fallback string
    still carried the curly-quote characters, which are absent from the chunk. This is the
    "curly quotes" edge case the guard contract names, only previously covered for a curly
    apostrophe inside a straight-quoted wrapper."""
    from app.services.fulltext import _quote_segments

    quote = "“Participants reported a mean gain of 1.62 points on the post-test”"
    segments = _quote_segments(quote)
    assert segments == ["Participants reported a mean gain of 1.62 points on the post-test"]


def test_guard_quote_fidelity_does_not_fire_on_a_typographic_double_quoted_verbatim_quote():
    """A verified answer whose evidence_quote is
    wrapped in curly double quotes, and whose inner text is verbatim in the chunk, must not
    be demoted to needs_nuance."""
    from app.services.fulltext import _guard_quote_fidelity

    quote = "“Participants reported a mean gain of 1.62 points on the post-test”"
    chunk = (
        "Participants reported a mean gain of 1.62 points on the post-test, which was "
        "statistically significant."
    )
    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=quote, chunk_texts=[chunk]
    )
    assert status == "verified"
    assert reasons == []


def test_quote_segments_empty_for_none_or_blank():
    from app.services.fulltext import _quote_segments

    assert _quote_segments(None) == []
    assert _quote_segments("") == []


# --------------------------------------------------------------------------------------
# STATUS_RANK / _cap
# --------------------------------------------------------------------------------------


def test_cap_returns_the_stricter_status():
    from app.services.fulltext import _cap

    assert _cap("verified", "unsupported") == "unsupported"
    assert _cap("unsupported", "verified") == "unsupported"
    assert _cap("needs_nuance", "needs_nuance") == "needs_nuance"
    assert _cap("verified", "needs_nuance") == "needs_nuance"


# --------------------------------------------------------------------------------------
# Guard 0: no_full_text returned despite chunks
# --------------------------------------------------------------------------------------


def test_guard_no_full_text_with_chunks_rewrites_to_unsupported():
    from app.services.fulltext import _guard_no_full_text_with_chunks

    status, reasons = _guard_no_full_text_with_chunks("no_full_text", ["a chunk of text"])
    assert status == "unsupported"
    assert reasons == ["no_full_text_with_chunks"]


def test_guard_no_full_text_with_chunks_leaves_the_deterministic_path_alone():
    from app.services.fulltext import _guard_no_full_text_with_chunks

    status, reasons = _guard_no_full_text_with_chunks("no_full_text", [])
    assert status == "no_full_text"
    assert reasons == []


def test_guard_no_full_text_with_chunks_does_not_touch_other_statuses():
    from app.services.fulltext import _guard_no_full_text_with_chunks

    status, reasons = _guard_no_full_text_with_chunks("verified", ["a chunk"])
    assert status == "verified"
    assert reasons == []


# --------------------------------------------------------------------------------------
# Guard 1: attribution
# --------------------------------------------------------------------------------------


def test_guard_attribution_caps_a_wrong_paper_fixture_to_unsupported():
    from app.services.fulltext import _guard_attribution

    status, reasons = _guard_attribution(
        "unsupported",
        claim_text="Reminders increased hand hygiene compliance among nurses by 20 percent.",
        chunk_texts=["This paper is about coral reef bleaching in the Pacific Ocean."],
        evidence_quote=None,
    )
    assert status == "unsupported"
    assert reasons == ["attribution_mismatch"]


def test_guard_attribution_does_not_fire_when_the_quote_is_located():
    from app.services.fulltext import _guard_attribution

    status, reasons = _guard_attribution(
        "unsupported",
        claim_text="Reminders increased hand hygiene compliance among nurses by 20 percent.",
        chunk_texts=[
            "This paper is about coral reef bleaching. "
            "Reminders increased hand hygiene compliance among nurses by 20 percent."
        ],
        evidence_quote='"Reminders increased hand hygiene compliance among nurses by 20 percent."',
    )
    assert status == "unsupported"
    assert reasons == []


def test_guard_attribution_does_not_fire_when_claim_terms_are_covered():
    from app.services.fulltext import _guard_attribution

    status, reasons = _guard_attribution(
        "verified",
        claim_text="Scores improved.",
        chunk_texts=["Test scores improved by 12 percent among the treatment group."],
        evidence_quote=None,
    )
    assert status == "verified"
    assert reasons == []


def test_guard_attribution_does_not_fire_when_claim_terms_is_empty():
    from app.services.fulltext import _guard_attribution

    status, reasons = _guard_attribution(
        "unsupported",
        claim_text="123 456 (2020).",  # no content terms at all after stopword/length filter
        chunk_texts=["Completely unrelated text."],
        evidence_quote=None,
    )
    assert status == "unsupported"
    assert reasons == []


def test_guard_attribution_reads_the_threshold_from_the_named_constant(monkeypatch):
    from app.services import fulltext

    monkeypatch.setattr(fulltext, "ATTRIBUTION_COVERAGE_MIN", 0.0)
    status, reasons = fulltext._guard_attribution(
        "unsupported",
        claim_text="Reminders increased hand hygiene compliance among nurses.",
        chunk_texts=["This paper is about coral reef bleaching in the Pacific Ocean."],
        evidence_quote=None,
    )
    # With the threshold monkeypatched to 0.0, any coverage >= 0.0 clears it: never fires.
    assert reasons == []
    assert status == "unsupported"

    monkeypatch.setattr(fulltext, "ATTRIBUTION_COVERAGE_MIN", 1.0)
    status, reasons = fulltext._guard_attribution(
        "unsupported",
        claim_text="Reminders increased hand hygiene compliance among nurses.",
        chunk_texts=["This paper is about coral reef bleaching in the Pacific Ocean."],
        evidence_quote=None,
    )
    assert reasons == ["attribution_mismatch"]


def test_claim_terms_strips_a_parenthetical_citation_before_tokenising():
    """Production claim text always carries its in-text
    citation, and the cited author's own surname is essentially never repeated in that
    paper's own body text, so leaving the citation in the coverage calculation would fire
    guard 1 on citation metadata rather than on-topic content for every real claim, not just
    a genuine wrong-paper mismatch."""
    from app.services.fulltext import _claim_terms

    assert _claim_terms("Scores improved (Smith, 2020).") == {"scores", "improved"}
    assert _claim_terms("Scores improved (Smith & Jones, 2019a).") == {"scores", "improved"}
    assert _claim_terms("Scores improved (Smith et al., 2021).") == {"scores", "improved"}
    # a bare year with no parentheses is not a citation and is left alone (it is simply
    # never a "term" either way, since _content_tokens only keeps alphabetic runs)
    assert _claim_terms("The effect held in 2020 without parentheses.") == {
        "effect", "held", "without", "parentheses",
    }


# --------------------------------------------------------------------------------------
# The letter-hyphen-linebreak fold reaches the chunk haystack; without folding the
# claim-term side too, a hyphenated claim term ("self-evaluation") could never be a
# substring of the folded haystack ("selfevaluation") for ANY source spelling, including
# the claim's own contiguous one. `_claim_terms` folds each term the same way.
# --------------------------------------------------------------------------------------


def test_claim_terms_folds_a_letter_hyphen_linebreak_the_same_way_as_the_chunk_side():
    from app.services.fulltext import _claim_terms

    assert _claim_terms("Self-evaluation improved retention.") == {
        "selfevaluation", "improved", "retention",
    }


@pytest.mark.parametrize(
    "chunk_text",
    [
        "Improved retention followed self-evaluation exercises.",  # contiguous
        "Improved retention followed self- evaluation exercises.",  # line-break
        "Improved retention followed selfevaluation exercises.",  # joined
    ],
)
def test_guard_attribution_matches_a_hyphenated_claim_term_against_any_source_spelling(
    chunk_text,
):
    """Before the fix, the un-folded claim term "self-evaluation" (with its hyphen) is
    never a substring of the folded haystack for any of the three source spellings, so
    coverage drops to 2/3 (0.667), under ATTRIBUTION_COVERAGE_MIN (0.71), and the guard
    wrongly caps a correct "verified" status to "unsupported"."""
    from app.services.fulltext import _guard_attribution

    status, reasons = _guard_attribution(
        "verified",
        claim_text="Self-evaluation improved retention.",
        chunk_texts=[chunk_text],
        evidence_quote=None,
    )
    assert status == "verified"
    assert reasons == []


def test_guard_attribution_is_wired_status_changing_after_dev_set_calibration():
    """The classes separated on hss-dev-30, so the guard must be
    live (ATTRIBUTION_GUARD_REPORT_ONLY = False), not merely reporting the slug."""
    from app.services.fulltext import ATTRIBUTION_GUARD_REPORT_ONLY, apply_verification_guards

    assert ATTRIBUTION_GUARD_REPORT_ONLY is False
    status, reasons, _diagnostics = apply_verification_guards(
        "verified",
        claim_text="Reminders increased hand hygiene compliance among nurses by 20 percent.",
        evidence_quote=None,
        chunk_texts=["This paper is about coral reef bleaching in the Pacific Ocean."],
        chunks=[{"text": "This paper is about coral reef bleaching in the Pacific Ocean."}],
    )
    assert status == "unsupported"
    assert "attribution_mismatch" in reasons


# --------------------------------------------------------------------------------------
# New guard: assertion-status consistency
# --------------------------------------------------------------------------------------


def test_guard_assertion_status_consistency_never_fires_on_empty_assertions():
    from app.services.fulltext import _guard_assertion_status_consistency

    status, reasons, diagnostics = _guard_assertion_status_consistency(
        "verified", assertions=[]
    )
    assert status == "verified" and reasons == [] and diagnostics == []
    status, reasons, diagnostics = _guard_assertion_status_consistency(
        "verified", assertions=None
    )
    assert status == "verified" and reasons == [] and diagnostics == []


def test_guard_assertion_status_consistency_floors_unsupported_on_any_contradicted():
    from app.services.fulltext import _guard_assertion_status_consistency

    status, reasons, diagnostics = _guard_assertion_status_consistency(
        "verified",
        assertions=[
            {"text": "central one", "verdict": "supported", "central": True},
            {"text": "peripheral one", "verdict": "contradicted", "central": False},
        ],
    )
    assert status == "unsupported"
    assert reasons == ["assertion_status_inconsistent"]
    assert diagnostics == []


def test_guard_assertion_status_consistency_floors_unsupported_when_central_is_absent():
    from app.services.fulltext import _guard_assertion_status_consistency

    status, reasons, diagnostics = _guard_assertion_status_consistency(
        "verified",
        assertions=[
            {"text": "central one", "verdict": "absent", "central": True},
            {"text": "peripheral one", "verdict": "supported", "central": False},
        ],
    )
    assert status == "unsupported"
    assert reasons == ["assertion_status_inconsistent"]
    assert diagnostics == []


def test_guard_assertion_status_consistency_floors_needs_nuance_when_a_peripheral_is_absent():
    from app.services.fulltext import _guard_assertion_status_consistency

    status, reasons, diagnostics = _guard_assertion_status_consistency(
        "verified",
        assertions=[
            {"text": "central one", "verdict": "supported", "central": True},
            {"text": "peripheral one", "verdict": "absent", "central": False},
        ],
    )
    assert status == "needs_nuance"
    assert reasons == ["assertion_status_inconsistent"]
    assert diagnostics == []


def test_guard_assertion_status_consistency_does_not_fire_when_everything_is_supported():
    from app.services.fulltext import _guard_assertion_status_consistency

    status, reasons, diagnostics = _guard_assertion_status_consistency(
        "verified",
        assertions=[
            {"text": "central one", "verdict": "supported", "central": True},
            {"text": "peripheral one", "verdict": "supported", "central": False},
        ],
    )
    assert status == "verified" and reasons == [] and diagnostics == []


def test_guard_assertion_status_consistency_skips_centrality_clause_with_zero_central():
    """Zero `central=True` assertions: the centrality clause is skipped (there is no "the"
    central assertion to test) and `centrality_unmarked` is recorded as a diagnostic, never
    as a machine reason. The peripheral-absent check still runs over every assertion."""
    from app.services.fulltext import _guard_assertion_status_consistency

    status, reasons, diagnostics = _guard_assertion_status_consistency(
        "verified",
        assertions=[
            {"text": "one", "verdict": "supported", "central": False},
            {"text": "two", "verdict": "supported", "central": False},
        ],
    )
    assert status == "verified"
    assert reasons == []
    assert diagnostics == ["centrality_unmarked"]


def test_guard_assertion_status_consistency_skips_centrality_clause_with_several_central():
    from app.services.fulltext import _guard_assertion_status_consistency

    status, reasons, diagnostics = _guard_assertion_status_consistency(
        "verified",
        assertions=[
            {"text": "one", "verdict": "absent", "central": True},
            {"text": "two", "verdict": "supported", "central": True},
        ],
    )
    # Two central=True: the centrality clause (which would otherwise floor `unsupported`
    # on the absent one) is skipped; the absent assertion is instead caught by the
    # peripheral-absent check (every assertion is peripheral when centrality is unmarked),
    # flooring `needs_nuance`, and `centrality_unmarked` is recorded.
    assert status == "needs_nuance"
    assert reasons == ["assertion_status_inconsistent"]
    assert diagnostics == ["centrality_unmarked"]


def test_guard_assertion_status_consistency_contradicted_short_circuits_before_centrality_check():
    """Precedence (brief 1.3): contradicted anywhere floors unsupported first, in a strict
    ``else`` chain -- the centrality clause (and its `centrality_unmarked` diagnostic) is
    never reached once a contradicted assertion is found, whatever the central marking."""
    from app.services.fulltext import _guard_assertion_status_consistency

    status, reasons, diagnostics = _guard_assertion_status_consistency(
        "verified",
        assertions=[
            {"text": "one", "verdict": "contradicted", "central": False},
            {"text": "two", "verdict": "supported", "central": False},
        ],
    )
    assert status == "unsupported"
    assert reasons == ["assertion_status_inconsistent"]
    assert diagnostics == []


# --------------------------------------------------------------------------------------
# Guard 2: numeric consistency. Tier A is a diagnostic,
# `_diagnose_numeric_not_in_source`: no status in, no status out, never calls `_cap`. Tier
# B stays a status-changing guard, floor `needs_nuance`.
# --------------------------------------------------------------------------------------


def test_diagnose_numeric_not_in_source_fires_when_a_claim_numeral_is_absent_from_every_chunk():
    from app.services.fulltext import _diagnose_numeric_not_in_source

    reasons = _diagnose_numeric_not_in_source(
        claim_text="The intervention produced a large effect (d = 1.62).",
        chunk_texts=["The intervention produced a large effect (d = 0.81)."],
    )
    assert reasons == ["numeric_not_in_source"]


def test_diagnose_numeric_not_in_source_does_not_fire_when_the_numeral_matches_elsewhere():
    from app.services.fulltext import _diagnose_numeric_not_in_source

    reasons = _diagnose_numeric_not_in_source(
        claim_text="Seventeen students (36 percent) raised the concern.",
        chunk_texts=[
            "A separate part of the assignment was rated at 36.1 percent by students.",
            "Seventeen students (18 percent) raised the concern in interviews.",
        ],
    )
    assert reasons == []


def test_diagnose_numeric_not_in_source_takes_no_status_and_never_touches_one():
    """The renamed function's signature has no ``status``
    parameter at all and its return is a bare ``list[str]``, not a tuple -- there is
    nothing left for a caller to misuse as a status."""
    import inspect

    from app.services.fulltext import _diagnose_numeric_not_in_source

    params = inspect.signature(_diagnose_numeric_not_in_source).parameters
    assert "status" not in params
    result = _diagnose_numeric_not_in_source(
        claim_text="The intervention produced a large effect (d = 1.62).",
        chunk_texts=["The intervention produced a large effect (d = 0.81)."],
    )
    assert isinstance(result, list)


def test_guard_numeric_tier_b_fires_when_the_match_is_outside_the_quoted_chunk():
    from app.services.fulltext import _guard_numeric_tier_b

    chunk_texts = [
        "A separate part of the assignment was rated at 36.1 percent by students.",
        "Seventeen students (18 percent) raised the concern in interviews.",
    ]
    status, reasons = _guard_numeric_tier_b(
        "needs_nuance",
        claim_text="Seventeen students (36 percent) raised the concern.",
        chunk_texts=chunk_texts,
        evidence_index=1,
    )
    assert status == "needs_nuance"
    assert reasons == ["numeric_outside_cited_passage"]


def test_guard_numeric_tier_b_never_reaches_unsupported():
    from app.services.fulltext import _guard_numeric_tier_b

    status, reasons = _guard_numeric_tier_b(
        "verified",
        claim_text="Seventeen students (36 percent) raised the concern.",
        chunk_texts=[
            "A separate part of the assignment was rated at 36.1 percent by students.",
            "Seventeen students (18 percent) raised the concern in interviews.",
        ],
        evidence_index=1,
    )
    assert status == "needs_nuance"  # capped, never unsupported
    assert reasons == ["numeric_outside_cited_passage"]


def test_guard_numeric_tier_b_skipped_when_evidence_location_is_null():
    from app.services.fulltext import _guard_numeric_tier_b

    status, reasons = _guard_numeric_tier_b(
        "verified",
        claim_text="Scores rose by 1.62 points.",
        chunk_texts=["Scores rose by 0.81 points."],
        evidence_index=None,
    )
    assert status == "verified"
    assert reasons == []


def test_guard_numeric_tier_b_considers_every_located_chunk():
    """Tier B considers every located chunk,
    not just one. Here the claim's 36 percent IS present in chunk 0; passing both located
    indices must not fire, even though index 1 alone (its own numeral, 18 percent) would."""
    from app.services.fulltext import _guard_numeric_tier_b

    chunk_texts = [
        "Seventeen students (36 percent) raised the concern in the pilot.",
        "A separate part of the assignment was rated at 18 percent by students.",
    ]
    status, reasons = _guard_numeric_tier_b(
        "verified",
        claim_text="Seventeen students (36 percent) raised the concern.",
        chunk_texts=chunk_texts,
        evidence_indices=[0, 1],
    )
    assert status == "verified" and reasons == []


def test_guard_numeric_tier_b_fires_when_the_match_is_outside_every_located_chunk():
    from app.services.fulltext import _guard_numeric_tier_b

    chunk_texts = [
        "A separate part of the assignment was rated at 36.1 percent by students.",
        "Seventeen students (18 percent) raised the concern in interviews.",
        "The pilot cohort saw no comparable figure at all.",
    ]
    status, reasons = _guard_numeric_tier_b(
        "verified",
        claim_text="Seventeen students (36 percent) raised the concern.",
        chunk_texts=chunk_texts,
        evidence_indices=[1, 2],
    )
    assert status == "needs_nuance"
    assert reasons == ["numeric_outside_cited_passage"]


def test_guard_numeric_skips_years_and_citation_years():
    from app.services.fulltext import _diagnose_numeric_not_in_source

    reasons = _diagnose_numeric_not_in_source(
        claim_text="Engagement rose in 2019 (Smith, 2021).",
        chunk_texts=["Engagement rose in 2019, according to the annual report."],
    )
    assert reasons == []


def test_guard_numeric_skips_figure_table_and_chunk_indices():
    from app.services.fulltext import _diagnose_numeric_not_in_source

    reasons = _diagnose_numeric_not_in_source(
        claim_text="As shown in Table 3 and Figure 2, scores improved.",
        chunk_texts=["See Chunk 4 for the full breakdown; scores improved overall."],
    )
    assert reasons == []


def test_guard_numeric_verbatim_claim_fires_neither_check():
    from app.services.fulltext import _diagnose_numeric_not_in_source, _guard_numeric_tier_b

    chunk = "The intervention produced a large effect (d = 0.81)."
    assert _diagnose_numeric_not_in_source(claim_text=chunk, chunk_texts=[chunk]) == []
    status, reasons = _guard_numeric_tier_b(
        "verified", claim_text=chunk, chunk_texts=[chunk], evidence_index=0
    )
    assert status == "verified" and reasons == []


def test_numerals_equivalent_matches_same_value_across_percent_flags():
    """Two numerals with the identical value but
    only one carrying a literal '%' must match (a purely typographic difference), in both
    directions, and a bare integer must match the same integer written with a percent sign.
    Without this, the mixed-flag branch would compare ONLY the cross-scaled forms, so
    ``_numerals_equivalent((36.0, False), (36.0, True))`` would be ``False``."""
    from app.services.fulltext import _numerals_equivalent

    assert _numerals_equivalent((36.0, False), (36.0, True)) is True
    assert _numerals_equivalent((36.0, True), (36.0, False)) is True
    assert _numerals_equivalent((36.0, False), (36.0, False)) is True


def test_diagnose_numeric_not_in_source_does_not_fire_on_percent_vs_spelled_out_same_number():
    """'About 36 percent of learners improved.' against a
    chunk that writes the same number as '36%' must not fire, in either direction, and a
    bare count against the same count written with a percent sign must not fire."""
    from app.services.fulltext import _diagnose_numeric_not_in_source

    assert _diagnose_numeric_not_in_source(
        claim_text="About 36 percent of learners improved.",
        chunk_texts=["In total, 36% of the learners improved on the post-test."],
    ) == []

    assert _diagnose_numeric_not_in_source(
        claim_text="36% of the learners improved.",
        chunk_texts=["In total, 36 percent of the learners improved on the post-test."],
    ) == []

    assert _diagnose_numeric_not_in_source(
        claim_text="The study enrolled 36 participants.",
        chunk_texts=["Of the 36% who volunteered, all were enrolled."],
    ) == []


def test_diagnose_numeric_not_in_source_still_fires_on_the_162_vs_161_163_collision():
    """The same-scale comparison for mixed-flag pairs must not conflate
    1.62 (not a percent) and 161/163 (not percents), which share the
    same ``is_percent`` flag, so they are still compared only at their own scale."""
    from app.services.fulltext import _diagnose_numeric_not_in_source

    reasons = _diagnose_numeric_not_in_source(
        claim_text="The intervention produced a large effect (d = 1.62).",
        chunk_texts=["Sample sizes ranged from 161 to 163 across the three cohorts."],
    )
    assert reasons == ["numeric_not_in_source"]


# --------------------------------------------------------------------------------------
# New guard: promotes part of
# the numeric-not-in-source diagnostic back to status-changing, evidence-gated. Adopted
# after tabulating the diagnostic's firing against the label over every committed v3 row
# (results/v3, the superseded run, dev iterations 1-4 and their band2/): it fired on 16 of
# 472 STATUS_RANK-eligible rows, every one labelled `unsupported`, never `verified` or
# `needs_nuance` -- see the guard report for the full table.
# --------------------------------------------------------------------------------------


def test_guard_numeric_value_absent_does_not_fire_when_every_claim_numeral_matches():
    from app.services.fulltext import _guard_numeric_value_absent

    status, reasons = _guard_numeric_value_absent(
        "verified",
        claim_text="The intervention produced a large effect (d = 0.81).",
        chunk_texts=["The intervention produced a large effect (d = 0.81)."],
    )
    assert status == "verified"
    assert reasons == []


def test_guard_numeric_value_absent_caps_needs_nuance_rule_a():
    """Rule A: a claim numeral absent from every shown chunk caps a `verified` status to
    `needs_nuance`, unconditionally -- no `assertions` at all here (defaults to `None`)."""
    from app.services.fulltext import _guard_numeric_value_absent

    status, reasons = _guard_numeric_value_absent(
        "verified",
        claim_text="The intervention produced a large effect (d = 1.62).",
        chunk_texts=["The intervention produced a large effect (d = 0.81)."],
    )
    assert status == "needs_nuance"
    assert reasons == ["numeric_value_absent_from_source"]


def test_guard_numeric_value_absent_never_lowers_an_already_unsupported_status():
    """Guard monotonicity: capping `unsupported` to `needs_nuance` is a no-op (`_cap` only
    ever raises the rank); the reason is still reported, matching every other guard's
    convention of reporting whenever its own condition holds, not only when the cap
    actually changes anything (see `_guard_attribution`, `_guard_assertion_status_
    consistency`)."""
    from app.services.fulltext import _guard_numeric_value_absent

    status, reasons = _guard_numeric_value_absent(
        "unsupported",
        claim_text="The intervention produced a large effect (d = 1.62).",
        chunk_texts=["The intervention produced a large effect (d = 0.81)."],
    )
    assert status == "unsupported"
    assert reasons == ["numeric_value_absent_from_source"]


def test_guard_numeric_value_absent_floors_unsupported_rule_b_hss_altered_15_shape():
    """Rule B, reproducing the `hss-altered-15` run A shape (the lenient miss this guard
    exists to fix): the model marks the assertion carrying the
    altered number ``absent`` rather than ``contradicted``, so
    `_guard_assertion_status_consistency`'s contradicted clause never fires and its
    peripheral-absent clause only floors `needs_nuance` -- this guard's Rule B floors the
    remaining step, to `unsupported`, because the model's own assertion names the very
    number the diagnostic already found unsupported."""
    from app.services.fulltext import _guard_numeric_value_absent

    claim = (
        "The total number of interpersonal metadiscourse features (normalized per 1370 "
        "words) is higher in BR sub-corpus (76.26) than in RA sub-corpus (44.09)."
    )
    chunk = (
        "The total number of instances for each category found in each sub-corpus has "
        "been normalized per 1000 words. Total 275 76.26 478 44.09"
    )
    assertions = [
        {
            "text": "The total number of interpersonal metadiscourse features is higher "
            "in the BR sub-corpus than in the RA sub-corpus.",
            "verdict": "supported",
            "central": True,
        },
        {"text": "The BR sub-corpus value is 76.26.", "verdict": "supported", "central": False},
        {"text": "The RA sub-corpus value is 44.09.", "verdict": "supported", "central": False},
        {
            "text": "The values are normalized per 1370 words.",
            "verdict": "absent",
            "central": False,
        },
    ]
    status, reasons = _guard_numeric_value_absent(
        "needs_nuance", claim_text=claim, chunk_texts=[chunk], assertions=assertions
    )
    assert status == "unsupported"
    assert reasons == ["numeric_value_absent_from_source"]


def test_guard_numeric_value_absent_stays_at_needs_nuance_when_no_assertion_carries_the_number():
    """Rule B does not fire just because *some* assertion is `absent`: the absent assertion
    must itself carry a numeral equivalent to the one the diagnostic found missing. Here the
    absent assertion is about something else entirely."""
    from app.services.fulltext import _guard_numeric_value_absent

    claim = "The intervention produced a large effect (d = 1.62), replicating prior work."
    chunk = "The intervention produced a large effect (d = 0.81)."
    assertions = [
        {"text": "The effect size was 0.81.", "verdict": "supported", "central": True},
        {"text": "This replicates prior work.", "verdict": "absent", "central": False},
    ]
    status, reasons = _guard_numeric_value_absent(
        "verified", claim_text=claim, chunk_texts=[chunk], assertions=assertions
    )
    assert status == "needs_nuance"
    assert reasons == ["numeric_value_absent_from_source"]


def test_guard_numeric_value_absent_rule_b_reads_claim_value_when_text_has_no_numeral():
    """`_assertion_carries_an_absent_missing_numeral` also reads `claim_value`, not only
    `text`, since a model does not always restate a number inside the sentence-shaped
    ``text`` field."""
    from app.services.fulltext import _guard_numeric_value_absent

    claim = "The effect was large (d = 1.62), consistent with prior work."
    chunk = "The effect was large, consistent with prior work."
    assertions = [
        {
            "text": "The effect size was reported.",
            "claim_value": "1.62",
            "verdict": "absent",
            "central": True,
        },
    ]
    status, reasons = _guard_numeric_value_absent(
        "verified", claim_text=claim, chunk_texts=[chunk], assertions=assertions
    )
    assert status == "unsupported"
    assert reasons == ["numeric_value_absent_from_source"]


def test_apply_verification_guards_wires_the_numeric_value_absent_guard_rule_b():
    """End-to-end through `apply_verification_guards`: the `hss-altered-15`-shaped claim
    above reaches `unsupported`, and the reason lands alongside the assertion-consistency
    guard's own reason, in guard order."""
    from app.services.fulltext import apply_verification_guards

    claim = (
        "The total number of interpersonal metadiscourse features (normalized per 1370 "
        "words) is higher in BR sub-corpus (76.26) than in RA sub-corpus (44.09)."
    )
    chunk = (
        "The total number of instances for each category found in each sub-corpus has "
        "been normalized per 1000 words. Total 275 76.26 478 44.09"
    )
    quote = (
        "The total number of instances for each category found in each sub-corpus has "
        "been normalized per 1000 words."
    )
    assertions = [
        {
            "text": "The total number of interpersonal metadiscourse features is higher "
            "in the BR sub-corpus than in the RA sub-corpus.",
            "verdict": "supported",
            "central": True,
            "quote": quote,
        },
        {"text": "The BR sub-corpus value is 76.26.", "verdict": "supported", "central": False},
        {"text": "The RA sub-corpus value is 44.09.", "verdict": "supported", "central": False},
        {
            "text": "The values are normalized per 1370 words.",
            "verdict": "absent",
            "central": False,
        },
    ]
    status, reasons, diagnostics = apply_verification_guards(
        "needs_nuance",
        claim_text=claim,
        evidence_quote=quote,
        chunk_texts=[chunk],
        chunks=[{"text": chunk}],
        assertions=assertions,
    )
    assert status == "unsupported"
    assert "assertion_status_inconsistent" in reasons
    assert "numeric_value_absent_from_source" in reasons
    assert diagnostics == ["numeric_not_in_source"]


# --------------------------------------------------------------------------------------
# _numeral_is_identifier_fragment / _extract_numerals
# --------------------------------------------------------------------------------------


def test_extract_numerals_skips_a_digit_fused_to_a_following_letter():
    """A cell-line/clone name such as "NIH 3T3" must not extract "3" as a quantity: the
    digit is immediately followed by a letter with no separating space."""
    from app.services.fulltext import _extract_numerals

    assert _extract_numerals("Cellular clocks in NIH 3T3 cells.") == []


def test_extract_numerals_skips_a_hyphenated_compound_code():
    """A gene/protein/compound code such as "A-769662", "CTLA-4" or "interleukin-2" must not
    extract its trailing digits as a quantity: the digits are immediately preceded by a
    hyphen whose other side is a run of letters."""
    from app.services.fulltext import _extract_numerals

    assert _extract_numerals("A-769662 activates dephosphorylated AMPK.") == []
    assert _extract_numerals("anti-CTLA-4 treatment reinvigorates exhausted cells.") == []
    assert _extract_numerals("Reduced responsiveness to interleukin-2 in T cells.") == []


def test_extract_numerals_keeps_a_hyphenated_range_and_a_negative_number():
    """A plain numeric range ("12-15") or a leading-minus-sign number ("-0.42") is not a
    letter-hyphen-digit compound code (no letters touch the hyphen), so both sides still
    extract as ordinary numerals."""
    from app.services.fulltext import _extract_numerals

    assert _extract_numerals("scores of 12-15 were reported") == [(12.0, False), (15.0, False)]
    assert _extract_numerals("-0.42 correlation") == [(0.42, False)]


def test_extract_numerals_skip_identifiers_false_disables_the_filter():
    """``skip_identifiers=False`` is the escape hatch the guards use for the source-side
    candidate pool: the same text that extracts to
    ``[]`` by default must extract its digit run when the filter is turned off."""
    from app.services.fulltext import _extract_numerals

    assert _extract_numerals("Cellular clocks in NIH 3T3 cells.") == []
    assert _extract_numerals("Cellular clocks in NIH 3T3 cells.", skip_identifiers=False) == [
        (3.0, False)
    ]


def test_diagnose_numeric_not_in_source_ignores_source_only_unit_fused_form():
    """``_numeral_is_identifier_fragment`` is applied
    symmetrically to the source-side candidate pool. A claim spells a dose with a space
    ("50 mg") while the cited abstract fuses the unit to the number ("50mg"); without
    this, the source's "50" would be stripped as an apparent identifier fragment, the
    claim's "50" would then match nothing, and tier A would wrongly floor a correct
    "verified" answer to "unsupported". The source pool must stay maximal."""
    from app.services.fulltext import _diagnose_numeric_not_in_source

    reasons = _diagnose_numeric_not_in_source(
        claim_text="Patients received 50 mg daily.",
        chunk_texts=["A 50mg daily dose was administered to all patients."],
    )
    assert reasons == []


def test_diagnose_numeric_not_in_source_ignores_source_only_hyphen_fused_code():
    """Same asymmetry as above, for a hyphenated compound identifier: the claim writes the
    code with spaces around the hyphen ("CTLA - 4"), so its "4" is an ordinary extracted
    numeral, while the source's hyphen-fused spelling ("CTLA-4") must still contribute a
    matching numeral from the (unfiltered) source pool, not be stripped as an identifier."""
    from app.services.fulltext import _diagnose_numeric_not_in_source

    reasons = _diagnose_numeric_not_in_source(
        claim_text="LRBA prevents CTLA - 4 recycling.",
        chunk_texts=["This study shows that LRBA prevents CTLA-4 recycling to the cell surface."],
    )
    assert reasons == []


def test_diagnose_numeric_not_in_source_does_not_fire_on_an_identifier_only_claim():
    from app.services.fulltext import _diagnose_numeric_not_in_source

    reasons = _diagnose_numeric_not_in_source(
        claim_text="C2 works synergistically with A-769662 to activate AMPK.",
        chunk_texts=["A-769662 is a small-molecule activator of AMPK studied here."],
    )
    assert reasons == []


def test_guard_numeric_percentage_matches_its_proportion_form():
    from app.services.fulltext import _diagnose_numeric_not_in_source, _guard_numeric_tier_b

    reasons = _diagnose_numeric_not_in_source(
        claim_text="The proportion was 0.36 of the sample.",
        chunk_texts=["36% of the sample reported this outcome."],
    )
    assert reasons == []
    status, reasons = _guard_numeric_tier_b(
        "verified",
        claim_text="The proportion was 0.36 of the sample.",
        chunk_texts=["36% of the sample reported this outcome."],
        evidence_index=0,
    )
    assert status == "verified" and reasons == []


def test_diagnose_numeric_not_in_source_fires_on_the_162_vs_161_163_collision():
    """A claim numeral of 1.62 must fire when the
    only nearby source numerals are 161 and 163, neither a percentage. An unconditional
    times-100 expansion would scale 1.62 to 162.0, which would then "match" 161.0/163.0
    well within tolerance -- exactly the real-data collision this guard must avoid."""
    from app.services.fulltext import _diagnose_numeric_not_in_source

    reasons = _diagnose_numeric_not_in_source(
        claim_text="The intervention produced a large effect (d = 1.62).",
        chunk_texts=["Sample sizes ranged from 161 to 163 across the three cohorts."],
    )
    assert reasons == ["numeric_not_in_source"]


# --------------------------------------------------------------------------------------
# _extract_numerals: documented edge cases. Each pins the *intended* behaviour, not an
# incidental one, so a future change is deliberate. See the docstring of each test for
# the decision it records.
# --------------------------------------------------------------------------------------


def test_extract_numerals_loses_a_negative_sign():
    """Documented limitation: the extraction regex has no sign awareness, so a negative
    number's magnitude is extracted but the sign is dropped. A sign flip (-0.42 -> 0.42)
    therefore never fires tier A on its own. This is intentional, not an oversight: sign
    and direction are exactly what the v2 prompt's "direction or sign" atomic assertion
    (brief section 3.1, step 1) asks the model to decide, with a "contradicted" verdict
    when it flips; teaching the numeric guard to parse a leading `-` correctly for every
    surrounding punctuation context (a hyphenated range, a parenthesised negative, a
    minus used as a dash) is a materially larger change than this guard's scope, so the
    magnitude-only check is deliberately left as the (weaker) backstop and the sign is the
    model's job."""
    from app.services.fulltext import _extract_numerals

    assert _extract_numerals("-0.42 correlation") == [(0.42, False)]


def test_extract_numerals_has_no_unit_awareness():
    """Documented limitation: units are not part of the extraction regex (brief section
    4.3 defines it as digits plus an optional trailing `%`, nothing else), so "3.5 mg/kg"
    extracts the bare magnitude with no unit attached. A source using the same magnitude in
    a different unit is therefore treated as a match; the v2 prompt's per-assertion
    "quantity" step, which requires unit conversion before comparison, is what actually
    catches a unit mismatch."""
    from app.services.fulltext import _extract_numerals

    assert _extract_numerals("3.5 mg/kg") == [(3.5, False)]


def test_extract_numerals_treats_a_range_as_two_independent_numerals():
    """Documented limitation: "12-15" is not recognised as a range; each side is extracted
    as its own numeral ((12.0, False), (15.0, False)). A source that states only one end of
    the range therefore still lets tier A find a match for the other end. Range-aware
    extraction is out of the brief's scope (section 4.3 does not mention ranges)."""
    from app.services.fulltext import _extract_numerals

    assert _extract_numerals("scores of 12-15 were reported") == [(12.0, False), (15.0, False)]


def test_extract_numerals_handles_the_literal_digit_pair_form():
    """The "17 (18%)" pair form, exercised in other tests only through the spelled-out
    "Seventeen students (36 percent)" variant: the literal digit form must extract both
    numerals with the percent flag set on the parenthesised one only."""
    from app.services.fulltext import _extract_numerals

    assert _extract_numerals("17 (18%) students raised the concern.") == [
        (17.0, False),
        (18.0, True),
    ]


def test_extract_numerals_does_not_set_is_percent_for_the_spelled_out_word():
    """A spelled-out "percent" does NOT
    set `is_percent`. Brief section 4.3 defines the extraction regex literally as digits
    plus an optional trailing `%` sign; it does not mention the word "percent" at all, so
    "36 percent" extracts as (36.0, False) -- exactly as if the word were not there.
    Consequently `_diagnose_numeric_not_in_source` does not grant this numeral the
    percent/proportion equivalence against a source written as a bare proportion ("0.36"):
    both numerals carry `is_percent=False`, so they are compared directly, not at percent
    scale, and 36 is far outside tolerance of 0.36. This is an accepted scope
    limitation, not a defect: reproducing the expected bucket counts (0/5, 0/5, 1/10,
    0/5) required exactly this literal, regex-only reading of
    `is_percent` (a claim written with the word "percent" is unusual in the frozen
    HSS/SciFact data; broadening extraction to sniff neighbouring words would go beyond the
    regex the brief specifies and risks false negatives elsewhere, e.g. "36 percentile")."""
    from app.services.fulltext import _diagnose_numeric_not_in_source, _extract_numerals

    assert _extract_numerals("36 percent of the sample") == [(36.0, False)]
    reasons = _diagnose_numeric_not_in_source(
        claim_text="36 percent of the sample",
        chunk_texts=["a proportion of 0.36 of the sample"],
    )
    assert reasons == ["numeric_not_in_source"]


# --------------------------------------------------------------------------------------
# Guard 3: quote fidelity
# --------------------------------------------------------------------------------------


def test_guard_quote_fidelity_fires_when_verified_and_quote_is_null():
    from app.services.fulltext import _guard_quote_fidelity

    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=None, chunk_texts=["Scores improved by 12 percent."]
    )
    assert status == "needs_nuance"
    assert reasons == ["quote_not_verbatim"]


def test_guard_quote_fidelity_fires_when_a_segment_is_not_in_any_chunk():
    from app.services.fulltext import _guard_quote_fidelity

    status, reasons = _guard_quote_fidelity(
        "verified",
        evidence_quote='"Scores improved by 40 percent among the treatment group."',
        chunk_texts=["Scores improved by 12 percent among the treatment group."],
    )
    assert status == "needs_nuance"
    assert reasons == ["quote_not_verbatim"]


def test_guard_quote_fidelity_does_not_fire_on_a_verbatim_quote():
    from app.services.fulltext import _guard_quote_fidelity

    status, reasons = _guard_quote_fidelity(
        "verified",
        evidence_quote='"Scores improved by 12 percent among the treatment group."',
        chunk_texts=["Scores improved by 12 percent among the treatment group."],
    )
    assert status == "verified"
    assert reasons == []


def test_guard_quote_fidelity_tolerates_curly_quotes_and_line_wraps():
    from app.services.fulltext import _guard_quote_fidelity

    status, reasons = _guard_quote_fidelity(
        "verified",
        evidence_quote='"Students’ scores rose\n by 12 percent."',
        chunk_texts=["Students' scores rose by 12 percent."],
    )
    assert status == "verified"
    assert reasons == []


def test_guard_quote_fidelity_only_applies_to_verified_status():
    from app.services.fulltext import _guard_quote_fidelity

    status, reasons = _guard_quote_fidelity(
        "needs_nuance", evidence_quote=None, chunk_texts=["anything"]
    )
    assert status == "needs_nuance"
    assert reasons == []


def test_guard_quote_fidelity_falls_back_to_the_raw_quote_when_every_segment_is_too_short():
    """A deliberate "no evidence either way"
    pass previously applied when a quote's only double-quoted fragment was shorter than
    QUOTE_SEGMENT_MIN_CHARS: `_quote_segments` dropped it and returned `[]`, so the guard's
    `for segment in segments` loop never ran and the status passed through unchanged.

    That decision was overturned: a *long* quote whose only
    effect of containing a short inner quoted term is that the whole surrounding text is
    discarded as a candidate is not "no evidence either way" -- it is a hole a wholly
    fabricated quote can walk straight through. `_quote_segments` falls back to the raw
    quote itself in exactly this case (every double-quoted fragment found but none long
    enough), so the guard checks the full span instead of passing it through unchecked.
    """
    from app.services.fulltext import _guard_quote_fidelity

    status, reasons = _guard_quote_fidelity(
        "verified",
        evidence_quote='The authors describe a "communicative approach" that improved '
        "outcomes substantially.",
        chunk_texts=["This paper is about coral reef bleaching in the Pacific Ocean."],
    )
    assert status == "needs_nuance"
    assert reasons == ["quote_not_verbatim"]


def test_guard_quote_fidelity_catches_a_fabricated_quote_hidden_behind_a_short_inner_quote():
    """A fabricated, wholly
    unverifiable `evidence_quote` must not pass just because its only double-quoted
    fragment ("significant", well under the 30-char floor) makes `_quote_segments` discard
    every candidate."""
    from app.services.fulltext import _guard_quote_fidelity

    status, reasons = _guard_quote_fidelity(
        "verified",
        evidence_quote='The authors found "significant" gains in every measured cohort '
        "across three years.",
        chunk_texts=["Nothing in this chunk resembles the quoted sentence at all."],
    )
    assert status == "needs_nuance"
    assert reasons == ["quote_not_verbatim"]


def test_guard_quote_fidelity_still_passes_a_genuinely_short_unquoted_claim():
    """A quote with no double-quoted fragment at all (``found`` empty) is unaffected by the
    fallback: the raw string is already the sole candidate, so a short unquoted
    quote below the floor still yields no segments and the guard stays quiet. This is the
    only case the "no evidence either way" reasoning still applies to."""
    from app.services.fulltext import _guard_quote_fidelity

    status, reasons = _guard_quote_fidelity(
        "verified",
        evidence_quote="a short answer",
        chunk_texts=["This paper is about something else entirely."],
    )
    assert status == "verified"
    assert reasons == []


# --------------------------------------------------------------------------------------
# Multi-span quotes: `_guard_quote_fidelity`,
# `_locate_evidence_chunk_index` and `_find_evidence_location` take `evidence_quotes` and
# apply `_quote_segments` to each, keeping single-span behaviour on one span.
# --------------------------------------------------------------------------------------


def test_guard_quote_fidelity_checks_every_span_in_evidence_quotes():
    from app.services.fulltext import _guard_quote_fidelity

    chunks = [
        "Scores rose by 12 percent among the treatment group.",
        "The effect held across three follow-up sessions.",
    ]
    status, reasons = _guard_quote_fidelity(
        "verified",
        evidence_quote=None,
        evidence_quotes=[
            "Scores rose by 12 percent among the treatment group.",
            "The effect held across three follow-up sessions.",
        ],
        chunk_texts=chunks,
    )
    assert status == "verified" and reasons == []


def test_guard_quote_fidelity_fires_when_any_span_in_evidence_quotes_is_not_verbatim():
    from app.services.fulltext import _guard_quote_fidelity

    chunks = [
        "Scores rose by 12 percent among the treatment group.",
        "The effect held across three follow-up sessions.",
    ]
    status, reasons = _guard_quote_fidelity(
        "verified",
        evidence_quote=None,
        evidence_quotes=[
            "Scores rose by 12 percent among the treatment group.",
            "The effect held across five follow-up sessions.",  # not verbatim
        ],
        chunk_texts=chunks,
    )
    assert status == "needs_nuance" and reasons == ["quote_not_verbatim"]


def test_guard_quote_fidelity_falls_back_to_evidence_quote_when_evidence_quotes_is_empty():
    """``evidence_quotes`` defaults to ``None``, so a caller offering only the single-span
    ``evidence_quote`` (or an explicit empty list) keeps the old behaviour unchanged."""
    from app.services.fulltext import _guard_quote_fidelity

    chunk = "Scores rose by 12 percent among the treatment group."
    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=chunk, evidence_quotes=[], chunk_texts=[chunk]
    )
    assert status == "verified" and reasons == []


def test_locate_evidence_chunk_index_searches_every_span_in_evidence_quotes():
    from app.services.fulltext import _locate_evidence_chunk_index

    chunks = [
        {"section": "intro", "text": "Nothing relevant here at all in this section."},
        {"section": "results", "text": "Scores rose by 12 percent among the treatment group."},
    ]
    idx = _locate_evidence_chunk_index(
        None,
        chunks,
        evidence_quotes=[
            "a span not found anywhere in the paper at all, quite long indeed",
            "Scores rose by 12 percent among the treatment group.",
        ],
    )
    assert idx == 1


def test_find_evidence_location_uses_evidence_quotes_when_the_single_quote_is_absent():
    from app.services.fulltext import _find_evidence_location

    chunks = [
        {"section": "intro", "text": "Nothing relevant here at all in this section."},
        {"section": "results", "text": "Scores rose by 12 percent among the treatment group."},
    ]
    location = _find_evidence_location(
        None,
        chunks,
        evidence_quotes=["Scores rose by 12 percent among the treatment group."],
    )
    assert location == "chunk 2 (results)"


def test_locate_evidence_chunk_indices_returns_every_distinct_located_chunk():
    from app.services.fulltext import _locate_evidence_chunk_indices

    chunks = [
        {"text": "Scores rose by 12 percent among the treatment group."},
        {"text": "The effect held across three follow-up sessions."},
        {"text": "Nothing relevant here at all."},
    ]
    indices = _locate_evidence_chunk_indices(
        None,
        chunks,
        evidence_quotes=[
            "Scores rose by 12 percent among the treatment group.",
            "The effect held across three follow-up sessions.",
        ],
    )
    assert indices == [0, 1]


def test_locate_evidence_chunk_indices_falls_back_to_the_single_evidence_quote():
    from app.services.fulltext import _locate_evidence_chunk_indices

    chunks = [{"text": "Scores rose by 12 percent among the treatment group."}]
    indices = _locate_evidence_chunk_indices(
        "Scores rose by 12 percent among the treatment group.", chunks
    )
    assert indices == [0]
    assert _locate_evidence_chunk_indices(None, chunks) == []


# --------------------------------------------------------------------------------------
# Text normalisation: collapsing a model's doubly-escaped quotes and
# newlines in `explanation` / `evidence_quote` / `suggested_revision`, and proving the
# ordering requirement -- this must run before `_guard_quote_fidelity` reads the text.
# --------------------------------------------------------------------------------------

#: Excerpt copied verbatim (including the literal backslash-quote defect) from the
#: `supported-1` explanation in `demo/expected/claim_report.json` before this fix.
DEFECTIVE_DEMO_EXPLANATION = (
    "Every atomic assertion in the claim is directly supported by the source text. The "
    "paper is a synthesis of naturalistic classroom studies where the type and amount of "
    "feedback were not manipulated or controlled (\\\"there is not yet a synthesis of "
    "naturalistic classroom studies where the type and amount of feedback provided on "
    "students' writing performance is not manipulated or controlled\\\")."
)


def test_collapse_double_escapes_replaces_backslash_quote_apostrophe_and_n():
    from app.services.fulltext import _collapse_double_escapes

    assert _collapse_double_escapes('He said \\"hi\\" and left.') == 'He said "hi" and left.'
    assert _collapse_double_escapes("It\\'s fine.") == "It's fine."
    assert _collapse_double_escapes("line one\\nline two") == "line one\nline two"
    assert _collapse_double_escapes(None) is None
    # A real, single-backslash escape sequence (as opposed to a doubled one) is untouched:
    # this helper only targets the two-character sequences a JSON parse leaves behind.
    assert _collapse_double_escapes("C:\\path") == "C:\\path"


def test_collapse_double_escapes_fixes_the_defective_demo_explanation_fixture():
    from app.services.fulltext import _collapse_double_escapes

    cleaned = _collapse_double_escapes(DEFECTIVE_DEMO_EXPLANATION)
    assert "\\" not in cleaned
    assert (
        '("there is not yet a synthesis of naturalistic classroom studies where the type '
        "and amount of feedback provided on students' writing performance is not "
        'manipulated or controlled").' in cleaned
    )


def test_strip_wrapping_quotes_strips_one_matched_pair():
    from app.services.fulltext import _strip_wrapping_quotes

    assert _strip_wrapping_quotes('"The paper supports this."') == "The paper supports this."
    assert _strip_wrapping_quotes("'The paper supports this.'") == "The paper supports this."
    assert _strip_wrapping_quotes("“The paper supports this.”") == "The paper supports this."


def test_strip_wrapping_quotes_leaves_unwrapped_or_mismatched_text_alone():
    from app.services.fulltext import _strip_wrapping_quotes

    assert _strip_wrapping_quotes("The paper supports this.") == "The paper supports this."
    # Only a matching open/close pair counts as "wrapped": a lone leading quote (e.g. the
    # model quoting a source fragment mid-sentence) must not be stripped.
    assert _strip_wrapping_quotes('"quoted mid-way through, not wrapped') == (
        '"quoted mid-way through, not wrapped'
    )
    assert _strip_wrapping_quotes("") == ""
    assert _strip_wrapping_quotes('"') == '"'


def test_strip_wrapping_quotes_does_not_strip_across_two_separate_quoted_spans():
    """Stripping must fire only when the
    whole string is ONE quoted span. A string that starts and ends with a matching mark but
    contains ANOTHER instance of that mark in between -- two separate quoted spans joined by
    plain text, not one wrapped answer -- must be left alone: stripping the two boundary
    marks here would leave a stray, unpaired quote mark on each inner span."""
    from app.services.fulltext import _strip_wrapping_quotes

    assert _strip_wrapping_quotes('"a" and "b"') == '"a" and "b"'
    assert _strip_wrapping_quotes("'a' and 'b'") == "'a' and 'b'"
    assert _strip_wrapping_quotes("“a” and “b”") == "“a” and “b”"


def test_normalise_verification_text_collapses_escapes_and_optionally_strips_quotes():
    from app.services.fulltext import normalise_verification_text

    assert normalise_verification_text(None) is None
    assert normalise_verification_text('\\"wrapped\\"', strip_wrapping_quotes=True) == "wrapped"
    # Same input, without the flag (the `evidence_quote` call site): escapes are still
    # collapsed, but the wrapping marks -- which `_quote_segments` relies on to find the
    # quoted fragment -- are left in place.
    assert normalise_verification_text('\\"wrapped\\"') == '"wrapped"'


def test_guard_quote_fidelity_no_longer_cares_about_a_doubly_escaped_quote_mark():
    """A model that pre-escaped its own closing quote mark leaves a literal
    trailing backslash inside the fragment `_QUOTED_FRAGMENT_RE` captures.
    `_normalise_for_match` keeps only letters and digits, so a bare backslash vanishes on
    its own, whether or not `normalise_verification_text` ran first -- both calls below
    agree. See the next test for a doubly-escaped case (a literal ``\\n``, two characters)
    that still requires the ordering.
    """
    from app.services.fulltext import _guard_quote_fidelity, normalise_verification_text

    chunk = "Test scores improved by 12 percent, which is significant for this cohort."
    raw_from_model = '\\"' + chunk + '\\"'

    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=raw_from_model, chunk_texts=[chunk]
    )
    assert status == "verified"
    assert reasons == []

    status, reasons = _guard_quote_fidelity(
        "verified",
        evidence_quote=normalise_verification_text(raw_from_model),
        chunk_texts=[chunk],
    )
    assert status == "verified"
    assert reasons == []


def test_guard_quote_fidelity_would_wrongly_demote_a_doubly_escaped_newline_unnormalised():
    """The ordering requirement the previous test used to prove, restated with a case P11c's
    broader fold does not close on its own: a doubly-escaped newline is the two literal
    characters backslash-then-``n``, and ``n`` is a letter, so it survives
    `_normalise_for_match`'s "letters and digits only" fold as a real, spurious letter
    inserted between the two words the line break split -- unlike a bare backslash (which has
    no letter/digit component and simply vanishes), this is not self-healing.
    `normalise_verification_text` must still run first to collapse the two-character
    sequence to one real newline character, which then vanishes with nothing left behind.
    """
    from app.services.fulltext import _guard_quote_fidelity, normalise_verification_text

    chunk = "Test scores improved by 12 percent, which is significant for this cohort."
    raw_from_model = (
        "Test scores improved" + "\\" + "n" + "by 12 percent, which is significant "
        "for this cohort."
    )

    status, reasons = _guard_quote_fidelity(
        "verified", evidence_quote=raw_from_model, chunk_texts=[chunk]
    )
    assert status == "needs_nuance"
    assert reasons == ["quote_not_verbatim"]

    status, reasons = _guard_quote_fidelity(
        "verified",
        evidence_quote=normalise_verification_text(raw_from_model),
        chunk_texts=[chunk],
    )
    assert status == "verified"
    assert reasons == []


# --------------------------------------------------------------------------------------
# apply_verification_guards: order, the monotonicity invariant, and the v3 additions
# (evidence_quotes, assertions, diagnostics)
# --------------------------------------------------------------------------------------


def test_apply_verification_guards_runs_guard_0_first_and_skips_the_rest():
    from app.services.fulltext import STATUS_RANK, apply_verification_guards

    status, reasons, diagnostics = apply_verification_guards(
        "no_full_text",
        claim_text="A chunk.",  # its only content term ("chunk") is covered: guard 1 is quiet
        evidence_quote=None,
        chunk_texts=["a chunk"],
        chunks=[{"section": "results", "text": "a chunk"}],
    )
    assert status == "unsupported"
    assert reasons == ["no_full_text_with_chunks"]
    assert diagnostics == []
    assert status in STATUS_RANK


def test_apply_verification_guards_keeps_the_numeric_diagnostic_slug_out_of_machine_reasons():
    """`_diagnose_numeric_not_in_source`'s own slug, ``numeric_not_in_source``, never
    appears in `machine_reasons` and the diagnostic itself never caps `status` (see that
    guard's own docstring). Here the underlying condition fires (the
    claim's 1.62 is absent from the chunk), alongside guard 3 (quote fidelity, no
    evidence_quote) and the `_guard_numeric_value_absent` guard, which acts
    on the very same condition through its own, different slug
    (``numeric_value_absent_from_source``)."""
    from app.services.fulltext import apply_verification_guards

    status, reasons, diagnostics = apply_verification_guards(
        "verified",
        claim_text="The intervention produced a large effect (d = 1.62).",
        evidence_quote=None,
        chunk_texts=["The intervention produced a large effect (d = 0.81)."],
        chunks=[
            {"section": "results", "text": "The intervention produced a large effect (d = 0.81)."}
        ],
    )
    assert status == "needs_nuance"
    assert reasons == ["quote_not_verbatim", "numeric_value_absent_from_source"]
    assert "numeric_not_in_source" not in reasons
    assert diagnostics == ["numeric_not_in_source"]


def test_apply_verification_guards_reports_quote_fidelity_when_numerals_are_fine():
    from app.services.fulltext import apply_verification_guards

    status, reasons, diagnostics = apply_verification_guards(
        "verified",
        claim_text="The intervention produced a large effect.",
        evidence_quote=None,
        chunk_texts=["The intervention produced a large effect."],
        chunks=[{"section": "results", "text": "The intervention produced a large effect."}],
    )
    assert status == "needs_nuance"
    assert reasons == ["quote_not_verbatim"]
    assert diagnostics == []


def test_apply_verification_guards_is_backward_compatible_with_the_old_five_argument_call():
    """New signature (brief 1.3): `evidence_quotes` and `assertions` default to `None`, so
    an un-updated caller that only ever knew the five original keywords still runs, and now
    gets a three-tuple back."""
    from app.services.fulltext import apply_verification_guards

    result = apply_verification_guards(
        "verified",
        claim_text="Scores improved.",
        evidence_quote="Scores improved.",
        chunk_texts=["Scores improved."],
        chunks=[{"text": "Scores improved."}],
    )
    assert len(result) == 3
    status, reasons, diagnostics = result
    assert status == "verified" and reasons == [] and diagnostics == []


def test_apply_verification_guards_floors_unsupported_on_a_contradicted_assertion():
    """The new consistency guard reaches through `apply_verification_guards`: a contradicted
    assertion floors `unsupported` even though the model itself answered `verified`, and the
    slug lands in `machine_reasons`, never in `diagnostics`."""
    from app.services.fulltext import apply_verification_guards

    chunk = "Test scores declined by 12 percent among the treatment group."
    status, reasons, diagnostics = apply_verification_guards(
        "verified",
        claim_text="Test scores improved.",
        evidence_quote='"Test scores declined by 12 percent among the treatment group."',
        chunk_texts=[chunk],
        chunks=[{"text": chunk}],
        assertions=[
            {
                "text": "Test scores improved.",
                "verdict": "contradicted",
                "central": True,
                "claim_value": "improved",
                "source_value": "declined",
            }
        ],
    )
    assert status == "unsupported"
    assert reasons == ["assertion_status_inconsistent"]
    assert diagnostics == []


def test_apply_verification_guards_records_centrality_unmarked_without_capping_status():
    """Zero `central=True` assertions never fires the centrality clause (brief 1.3): the
    claim stays `verified` and `centrality_unmarked` is a diagnostic only."""
    from app.services.fulltext import apply_verification_guards

    chunk = "Scores rose by 12 percent among the treatment group."
    quote = '"Scores rose by 12 percent among the treatment group."'
    status, reasons, diagnostics = apply_verification_guards(
        "verified",
        claim_text="Scores rose by 12 percent among the treatment group.",
        evidence_quote=quote,
        chunk_texts=[chunk],
        chunks=[{"text": chunk}],
        assertions=[
            {"text": "Scores rose.", "verdict": "supported", "central": False},
            {"text": "By 12 percent.", "verdict": "supported", "central": False},
        ],
    )
    assert status == "verified"
    assert reasons == []
    assert diagnostics == ["centrality_unmarked"]


def test_diagnostics_forced_on_and_off_never_change_the_final_status():
    """Extended monotonicity (brief 1.3): a diagnostic firing (or not) must never move the
    final status. The claim, quote and chunk are identical in both dispatches (so every
    status-changing guard behaves identically); only the assertions' `central` marking
    differs, so only the `centrality_unmarked` diagnostic differs between the two.

    The claim carries no numeral, so neither the numeric diagnostic
    (`numeric_not_in_source`) nor the status-changing `_guard_numeric_value_absent` guard
    fires in either dispatch, keeping the test isolated to exactly the one variable its
    name describes (`centrality_unmarked`).
    """
    from app.services.fulltext import apply_verification_guards

    claim = "The effect was large, consistent with prior work."
    chunk = "The effect was large, consistent with prior work."
    quote = '"The effect was large, consistent with prior work."'
    kwargs = dict(
        claim_text=claim,
        evidence_quote=quote,
        chunk_texts=[chunk],
        chunks=[{"text": chunk}],
    )

    status_off, reasons_off, diagnostics_off = apply_verification_guards(
        "verified",
        assertions=[
            {"text": "t1", "verdict": "supported", "central": True},
            {"text": "t2", "verdict": "supported", "central": False},
        ],
        **kwargs,
    )
    status_on, reasons_on, diagnostics_on = apply_verification_guards(
        "verified",
        assertions=[
            {"text": "t1", "verdict": "supported", "central": False},
            {"text": "t2", "verdict": "supported", "central": False},
        ],
        **kwargs,
    )

    assert reasons_off == reasons_on == []
    assert "centrality_unmarked" not in diagnostics_off
    assert "centrality_unmarked" in diagnostics_on
    assert "numeric_not_in_source" not in diagnostics_off
    assert "numeric_not_in_source" not in diagnostics_on
    assert status_off == status_on == "verified"


def test_apply_verification_guards_normalises_multi_span_evidence_quotes_and_assertion_quotes():
    """Proves `apply_verification_guards` itself
    normalises the multi-span fields (`fulltext.py:963-972`), not only the single-span
    `evidence_quote` the older tests above exercise. Two doubly-escaped spans, one verbatim
    in each of two chunks, are passed straight into `evidence_quotes` (no pre-normalisation
    by this test, unlike `test_guard_quote_fidelity_would_wrongly_demote_a_doubly_escaped_
    verbatim_quote_unnormalised` above, which normalises at the call site before calling the
    lower-level guard directly), plus an assertion carrying doubly-escaped `quote`/`quotes`.
    Without the function's own internal normalisation, the literal trailing backslash before
    each closing quote mark would not be found in either source chunk and `quote_not_verbatim`
    would fire, capping `verified` to `needs_nuance`.
    """
    from app.services.fulltext import apply_verification_guards

    chunk1 = "Test scores improved by 12 percent, which is significant for this cohort."
    chunk2 = "Sleep quality also improved significantly for every participant in the trial."
    raw_quote1 = '\\"' + chunk1 + '\\"'
    raw_quote2 = '\\"' + chunk2 + '\\"'

    status, reasons, diagnostics = apply_verification_guards(
        "verified",
        claim_text="Test scores and sleep quality both improved for the treatment cohort.",
        evidence_quote=raw_quote1,
        evidence_quotes=[raw_quote1, raw_quote2],
        assertions=[
            {
                "text": "Both measures improved.",
                "verdict": "supported",
                "quote": raw_quote1,
                "quotes": [raw_quote2],
            }
        ],
        chunk_texts=[chunk1, chunk2],
        chunks=[{"text": chunk1}, {"text": chunk2}],
    )
    assert status == "verified"
    assert "quote_not_verbatim" not in reasons
    assert reasons == []
    # Zero `central=True` assertions still fires the (status-neutral) centrality diagnostic;
    # unrelated to the multi-span normalisation this test targets, but worth pinning down so
    # a future change to the assertion's `verdict` here does not silently start hiding it.
    assert diagnostics == ["centrality_unmarked"]


@pytest.mark.parametrize(
    "model_status,evidence_quote,chunk_texts",
    [
        ("verified", '"Scores improved by 12 percent."', ["Scores improved by 12 percent."]),
        ("verified", None, ["Scores improved by 12 percent."]),
        ("needs_nuance", None, ["Scores improved by 12 percent."]),
        ("unsupported", None, ["Scores improved by 12 percent."]),
        ("verified", '"1.62 effect size"', ["0.81 effect size reported here."]),
        ("no_full_text", None, ["a chunk"]),
        ("no_full_text", None, []),
    ],
)
def test_apply_verification_guards_never_makes_status_less_strict(
    model_status, evidence_quote, chunk_texts
):
    """Property test (section 4.0): STATUS_RANK[final] >= STATUS_RANK[model_status] for
    every (model status, chunk set, claim) fixture, except no_full_text-with-no-chunks
    which guards never touch at all."""
    from app.services.fulltext import STATUS_RANK, apply_verification_guards

    final_status, _reasons, _diagnostics = apply_verification_guards(
        model_status,
        claim_text="Scores improved by 12 percent among the treatment group.",
        evidence_quote=evidence_quote,
        chunk_texts=chunk_texts,
        chunks=[{"section": "results", "text": t} for t in chunk_texts],
    )
    if model_status not in STATUS_RANK:
        # no_full_text with no chunks is untouched; with chunks, guard 0 rewrites it to
        # "unsupported", the strictest rank, which trivially satisfies the invariant.
        assert final_status in ("no_full_text", "unsupported")
    else:
        assert STATUS_RANK[final_status] >= STATUS_RANK[model_status]


# --------------------------------------------------------------------------------------
# `_extract_claims` narrative-citation coverage: only 4 of the 12 code-audit-matched
# citations in a demo draft reached the verifier, because `_extract_claims` recognised
# only the parenthetical citation shape ("(Author, Year)"); the rest were narrative
# ("Author (Year)"), which the citation audit (`app.services.citation_audit`) already
# counts as matched but `_extract_claims` never saw. These tests pin the five narrative
# forms the writing agent's own citation style produces, that the parenthetical path and
# sentence count are unaffected on text with no narrative citation, and the "et al."
# sentence-split fix the narrative "Author et al. (Year)" form depends on.
# --------------------------------------------------------------------------------------


def test_extract_claims_still_matches_the_parenthetical_forms_unchanged():
    """Regression pin: bracketed, plain, `et al.` and `&`
    parenthetical citations all still resolve to the same (sentence, key, text) triples."""
    from app.services.fulltext import _extract_claims

    text = (
        "Scores improved [Doe, 2019]. "
        "Scores improved (Smith, 2020). "
        "Scores improved (Jones et al., 2021). "
        "Scores improved (Lee & Park, 2022)."
    )
    assert _extract_claims(text) == [
        ("Scores improved [Doe, 2019].", "doe_2019", "[Doe, 2019]"),
        ("Scores improved (Smith, 2020).", "smith_2020", "(Smith, 2020)"),
        ("Scores improved (Jones et al., 2021).", "jones_2021", "(Jones et al., 2021)"),
        ("Scores improved (Lee & Park, 2022).", "lee_2022", "(Lee & Park, 2022)"),
    ]


@pytest.mark.parametrize(
    "sentence,expected_key,expected_citation_text",
    [
        ("Zhang (2018) found strong effects.", "zhang_2018", "Zhang (2018)"),
        ("Zhang and Li (2019) confirmed the effect.", "zhang_2019", "Zhang and Li (2019)"),
        ("Zhang et al. (2020) extended the design.", "zhang_2020", "Zhang et al. (2020)"),
        (
            "Zhang, Li and Wang (2021) replicated the study.",
            "zhang_2021",
            "Zhang, Li and Wang (2021)",
        ),
        ("Zhang's (2018) study was widely cited.", "zhang_2018", "Zhang's (2018)"),
    ],
)
def test_extract_claims_matches_each_narrative_citation_form(
    sentence, expected_key, expected_citation_text
):
    """The five narrative forms the writing agent's citation style produces:
    author-prominent, "and", "et al.", a three-author list, and
    the possessive. Each yields exactly one claim, keyed the same way `_build_paper_lookup`
    indexes the cited paper."""
    from app.services.fulltext import _extract_claims

    assert _extract_claims(sentence) == [(sentence, expected_key, expected_citation_text)]


def test_extract_claims_matches_a_narrative_citation_with_a_diacritic_surname():
    """A demo draft's actual failure case:
    "Bonilla López et al. (2018)" -- a diacritic in the surname, and (see the next test)
    the "et al." abbreviation immediately followed by the parenthesised year."""
    from app.services.fulltext import _extract_claims

    sentence = "Bonilla López et al. (2018) randomly assigned participants to conditions."
    assert _extract_claims(sentence) == [(sentence, "lópez_2018", "López et al. (2018)")]


def test_extract_claims_narrative_et_al_does_not_split_the_sentence_at_the_abbreviation():
    """A plain `re.split(r"(?<=[.!?])\\s+", text)` would split
    right after "et al." (the period there is followed by whitespace, unlike the comma
    that follows it in the parenthetical form), so "Yallop et al." and "(2021), employing
    ..." would land in two different sentence fragments and `_NARRATIVE_CITE` -- searched
    per fragment -- could never see the surname and the year together. This test pins one
    sentence in, one claim out, not two fragments and zero claims.
    """
    from app.services.fulltext import _extract_claims

    sentence = (
        "Yallop et al. (2021), employing an ethnographic approach, found that hedging "
        "devices influenced uptake."
    )
    assert _extract_claims(sentence) == [(sentence, "yallop_2021", "Yallop et al. (2021)")]


def test_extract_claims_et_al_sentence_split_fix_does_not_merge_unrelated_sentences():
    """The negative lookbehind (`_CLAIM_SENTENCE_SPLIT_RE`) is anchored on a *standalone*
    "al." token (a word boundary before the "a"), so ordinary sentence endings that merely
    end in the letters "al." ("deal.", "medal.") still split normally -- proving the fix is
    narrowly scoped to the "et al." abbreviation, not a general suppression of splitting
    before any word ending "al."."""
    from app.services.fulltext import _extract_claims

    text = "They struck a deal. Zhang (2018) found strong effects."
    assert _extract_claims(text) == [
        ("Zhang (2018) found strong effects.", "zhang_2018", "Zhang (2018)"),
    ]


def test_extract_claims_does_not_split_on_a_comparative_vs_abbreviation():
    """Without this handling, the sentence "Luo et al. (2025) found
    ChatGPT's precision exceeded Grammarly's (94-98% vs. 85%), but recall rose only from
    10% to 55% ..., and Grammarly outperformed ChatGPT in five error categories." would be
    cut in two right after "vs.", because that period is followed by whitespace and then a
    plain number, not a parenthesised year -- the one shape `_CLAIM_SENTENCE_SPLIT_RE`'s
    existing "et al." lookahead already excuses. The claim built from the first half would
    be a truncated, unbalanced fragment, and the second half (carrying two further
    findings) would never be linked to a citation or verified at all. One sentence in, one
    claim out."""
    from app.services.fulltext import _extract_claims

    sentence = (
        "Luo et al. (2025) found ChatGPT's precision exceeded Grammarly's (94-98% vs. "
        "85%), but recall rose only from 10% to 55% as prompts became more sophisticated, "
        "and Grammarly outperformed ChatGPT in five error categories."
    )
    assert _extract_claims(sentence) == [(sentence, "luo_2025", "Luo et al. (2025)")]


def test_sentence_fragments_keeps_a_vs_comparison_in_one_fragment():
    """The same fix, checked at `_sentence_fragments`' own level (the splitter
    `extract_claims_from_document` and `finalize_generated_section` both use): the
    fragment holding the citation spans the whole sentence, not just its first half."""
    from app.services.fulltext import _sentence_fragments

    text = (
        "Luo et al. (2025) found ChatGPT's precision exceeded Grammarly's (94-98% vs. "
        "85%), but recall rose only from 10% to 55% as prompts became more sophisticated, "
        "and Grammarly outperformed ChatGPT in five error categories."
    )
    fragments = _sentence_fragments(text)
    assert fragments == [(0, len(text))]


def test_extract_claims_vs_fix_does_not_block_a_genuine_split_after_a_different_word():
    """The fix keys narrowly on the word "vs" immediately before the period, so an
    ordinary sentence ending in an unrelated word is unaffected: "vs." only suppresses
    the split when it is itself the word right before the split point."""
    from app.services.fulltext import _extract_claims

    text = "Zhang (2018) found strong effects vs prior work. Lee (2019) replicated it."
    assert _extract_claims(text) == [
        (
            "Zhang (2018) found strong effects vs prior work.",
            "zhang_2018",
            "Zhang (2018)",
        ),
        ("Lee (2019) replicated it.", "lee_2019", "Lee (2019)"),
    ]


def test_extract_claims_narrative_possessive_and_tail_combine():
    """`Author and Other's (Year)` (seen in the demo draft as "Liu and Wu's (2019)"): the
    "and"-joined tail and the possessive apply together, and the key still resolves to the
    first author alone."""
    from app.services.fulltext import _extract_claims

    sentence = "This aligns with Liu and Wu's (2019) observation about proficiency."
    assert _extract_claims(sentence) == [(sentence, "liu_2019", "Liu and Wu's (2019)")]


def test_extract_claims_narrative_key_uses_the_last_surname_token_like_paper_lookup():
    """`_build_paper_lookup` indexes a paper under the *last* space-separated token of its
    first author's name (`first_author.split(",")[0].split()[-1]`), so "Van Dijk, Teun" is
    indexed as "dijk_2020". A narrative citation carrying the same particle-plus-surname
    ("Van Dijk (2020)") must resolve to the identical key for the two to ever match."""
    from app.services.fulltext import _extract_claims

    sentence = "Van Dijk (2020) proposed a discourse framework."
    assert _extract_claims(sentence) == [(sentence, "dijk_2020", "Van Dijk (2020)")]


def test_extract_claims_ignores_a_narrative_citation_with_a_placeholder_year():
    """`_build_paper_lookup` only ever indexes a paper under a real four-digit year, so a
    placeholder ("n.d.", "in press") can never resolve to a matched paper; `_extract_claims`
    skips it rather than emitting a claim that is guaranteed to come back `no_full_text`."""
    from app.services.fulltext import _extract_claims

    assert _extract_claims("Zhang (n.d.) is an unpublished manuscript.") == []
    assert _extract_claims("Zhang (in press) reports a related finding.") == []


def test_extract_claims_multi_citation_sentence_keeps_parenthetical_and_narrative_in_order():
    """"multi-citation sentences ... unchanged": a sentence citing two
    papers, one parenthetical and one narrative, still yields two claims, in the order the
    citations appear in the sentence text."""
    from app.services.fulltext import _extract_claims

    sentence = "Zhang (2018) found this, consistent with (Smith, 2020)."
    assert _extract_claims(sentence) == [
        (sentence, "zhang_2018", "Zhang (2018)"),
        (sentence, "smith_2020", "(Smith, 2020)"),
    ]


def test_extract_claims_on_the_demo_draft_covers_every_citation_the_audit_matched():
    """End-to-end coverage check against the real 2026-09-08 demo draft
    (`demo/output/20260908-102739/writing_result.json`): every one of the 12 citations its
    own `citation_audit` field reports as `matched` against the library now yields at least
    one claim sentence from `_extract_claims`."""
    import json
    from pathlib import Path

    from app.services.fulltext import _extract_claims
    from tests.conftest import skip_unless_run_dir

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = skip_unless_run_dir(repo_root / "demo" / "output" / "20260908-102739")
    demo_path = run_dir / "writing_result.json"
    data = json.loads(demo_path.read_text(encoding="utf-8"))
    audit = data["citation_audit"]
    assert audit["total"] == 12

    matched_keys = set()
    for label in audit["matched"]:
        surname, year = label.rsplit(", ", 1)
        matched_keys.add(f"{surname.split()[-1].lower()}_{year}")

    claim_keys = {key for _sentence, key, _citation_text in _extract_claims(data["content"])}
    missing = matched_keys - claim_keys
    assert missing == set(), f"citations matched by the audit but never extracted: {missing}"


# --------------------------------------------------------------------------------------
# `_extract_claims` narrative-citation edge cases: `citation_audit._AUTHOR_TAIL` lets
# the co-author group start one token to the left of the true first author whenever a
# capitalised lead-in word ("However,", "In China,") is directly followed by a comma and
# a two- or three-author narrative citation, so `_NARRATIVE_CITE`'s own ``surname``
# group resolves to the lead-in word, not the first author. Also covered: the
# sentence-split fix over-blocking and merging a genuine sentence break after "et al."
# into the next sentence; a source cited both narratively and parenthetically in the
# same sentence reaching the verifier twice; and a capitalised prose word directly
# followed by a parenthesised year, with no real citation there at all, which is a
# recorded design decision, not a code change: see the docstring of `_extract_claims`.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence,expected_key,expected_citation_text",
    [
        (
            "However, Zhang and Li (2019) found strong effects.",
            "zhang_2019",
            "However, Zhang and Li (2019)",
        ),
        (
            "Similarly, Liu and Wu (2019) reported gains.",
            "liu_2019",
            "Similarly, Liu and Wu (2019)",
        ),
        (
            "Nevertheless, Zhang, Li and Wang (2021) replicated it.",
            "zhang_2021",
            "Nevertheless, Zhang, Li and Wang (2021)",
        ),
        (
            "In China, Zhang and Li (2019) found strong effects.",
            "zhang_2019",
            "China, Zhang and Li (2019)",
        ),
        (
            "The effect held in Japan, and Zhang (2018) confirmed it.",
            "zhang_2018",
            "Japan, and Zhang (2018)",
        ),
    ],
)
def test_extract_claims_narrative_leading_word_does_not_steal_the_citation_key(
    sentence, expected_key, expected_citation_text
):
    """These five sentences would each be wrong-keyed ("however_2019", "similarly_2019",
    "nevertheless_2021", "china_2019", "japan_2018") without this -- the true first
    author's key, "zhang_2019" etc., would never be emitted at all. `_extract_claims`
    tries every capitalised name-shaped token in the whole narrative match, left to right, against a
    supplied `paper_lookup`, and keys the claim to the first one present -- the true first
    author, when the library has them, regardless of what capitalised word precedes them.
    """
    from app.services.fulltext import _extract_claims

    paper_lookup = {expected_key: object()}
    assert _extract_claims(sentence, paper_lookup) == [
        (sentence, expected_key, expected_citation_text)
    ]


def test_extract_claims_narrative_leading_word_falls_back_without_a_paper_lookup():
    """The disambiguation in the previous test only fires when a library is available to
    check candidates against. `_extract_claims(text)` (no second argument, still the
    direct-unit-test default) keeps resolving to the *last* token of `_NARRATIVE_CITE`'s
    own ``surname`` group -- the documented fallback -- since there is nothing else to
    disambiguate with."""
    from app.services.fulltext import _extract_claims

    sentence = "However, Zhang and Li (2019) found strong effects."
    assert _extract_claims(sentence) == [
        (sentence, "however_2019", "However, Zhang and Li (2019)")
    ]


def test_extract_claims_narrative_three_author_list_is_unaffected_by_the_fix():
    """`_resolve_narrative_citation_key` must not regress the ordinary case: when the
    first capitalised token in the match *is* already the first author (no lead-in word),
    the fix picks it immediately, same as before. "Zhang, Li and Wang (2021)" stays on
    `zhang_2021` whether or not a `paper_lookup` is supplied."""
    from app.services.fulltext import _extract_claims

    sentence = "Zhang, Li and Wang (2021) replicated the study."
    expected = [(sentence, "zhang_2021", "Zhang, Li and Wang (2021)")]
    assert _extract_claims(sentence) == expected
    assert _extract_claims(sentence, {"zhang_2021": object()}) == expected


def test_extract_claims_et_al_followed_by_a_new_sentence_still_splits():
    """Blocking the sentence
    split after *any* standalone "al." token, including one that genuinely ends a
    sentence, would merge "Smith et al. The replication failed (Jones, 2021)." into one
    claim whose claim_text held a sentence the verifier was never asked about.
    `_CLAIM_SENTENCE_SPLIT_RE` blocks the split only when a parenthesised year (or
    year placeholder) immediately follows, so a genuine new sentence after "et al." still
    splits -- this text yields one claim, from the second sentence only, not a merged one
    with both."""
    from app.services.fulltext import _extract_claims

    text = "This was demonstrated by Smith et al. The replication failed (Jones, 2021)."
    assert _extract_claims(text) == [
        ("The replication failed (Jones, 2021).", "jones_2021", "(Jones, 2021)"),
    ]


def test_extract_claims_deduplicates_a_citation_appearing_twice_in_one_sentence():
    """A source cited both narratively and
    parenthetically in the same sentence would otherwise yield two identical (sentence,
    citation_key) claims, sending the same paper to the verifier twice for the price
    of two model calls and two report/workbook rows. `_extract_claims` de-duplicates
    on (sentence, citation_key), keeping the first citation_text encountered (here, the
    narrative form, since it appears first in the sentence)."""
    from app.services.fulltext import _extract_claims

    sentence = "Zhang (2018) found X (Zhang, 2018)."
    assert _extract_claims(sentence) == [
        (sentence, "zhang_2018", "Zhang (2018)"),
    ]


def test_extract_claims_still_emits_a_narrative_claim_with_no_real_author():
    """A recorded design decision (see
    `_extract_claims`'s own docstring): a capitalised prose word directly followed by a
    parenthesised year, with no real citation there at all, is still emitted as a claim
    rather than silently dropped, for parity with the parenthetical path, which already
    emits a claim for any bracketed citation absent from the library. This is a decision,
    not a code change, on text no library could ever resolve any candidate against."""
    from app.services.fulltext import _extract_claims

    assert _extract_claims("The (2018) revision changed the framework.") == [
        ("The (2018) revision changed the framework.", "the_2018", "The (2018)")
    ]
    assert _extract_claims(
        "Data were collected in Beijing (2019) and Shanghai (2020)."
    ) == [
        (
            "Data were collected in Beijing (2019) and Shanghai (2020).",
            "beijing_2019",
            "Beijing (2019)",
        ),
        (
            "Data were collected in Beijing (2019) and Shanghai (2020).",
            "shanghai_2020",
            "Shanghai (2020)",
        ),
    ]


# --------------------------------------------------------------------------------------
# `extract_claims_from_document` -- mapping first, regex fallback for an unlinked or
# edited-away sentence, reference-list nodes excluded, dedupe preserved. The numbered
# resolver has its own dedicated file, `test_numbered_citations.py`.
# --------------------------------------------------------------------------------------


def _doc_paragraph(text, citation_links=None):
    attrs = {"citationLinks": citation_links} if citation_links else {}
    return {"type": "paragraph", "attrs": attrs, "content": [{"type": "text", "text": text}]}


def _public_coverage(coverage):
    """`coverage` without its ``diagnostics`` key: the six fields
    `app.schemas.fulltext.CitationCoverage` carries into the API. ``diagnostics`` is the
    unit/link counter set, kept out of these whole-dict
    comparisons so a new counter never rewrites a behavioural test."""
    return {key: value for key, value in coverage.items() if key != "diagnostics"}


def test_extract_claims_from_document_uses_the_mapping_first_not_the_regex_too():
    """A sentence with a surviving link is claimed exactly once, from the mapping --
    the author-year regex is never also run against it ("mapping first")."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"smith_2020": object()}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                "Tutoring improves outcomes (Smith, 2020).",
                citation_links=[
                    {
                        "sentence": "Tutoring improves outcomes (Smith, 2020).",
                        "keys": ["smith_2020"],
                        "citation_text": "(Smith, 2020)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (
            "Tutoring improves outcomes (Smith, 2020).",
            "smith_2020",
            "(Smith, 2020)",
            "Tutoring improves outcomes (Smith, 2020).",
        )
    ]
    assert coverage["by_source"] == {"mapping": 1, "author-year": 0, "numbered": 0}


def test_extract_claims_from_document_falls_back_to_the_regex_for_an_unlinked_sentence():
    """The mapping covers one sentence in the paragraph; a second sentence the writer's
    map never mentioned still reaches the verifier through the existing regexes."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"smith_2020": object(), "brown_2018": object()}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                "Tutoring improves outcomes (Smith, 2020). Brown et al. (2018) concur.",
                citation_links=[
                    {
                        "sentence": "Tutoring improves outcomes (Smith, 2020).",
                        "keys": ["smith_2020"],
                        "citation_text": "(Smith, 2020)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (
            "Tutoring improves outcomes (Smith, 2020).",
            "smith_2020",
            "(Smith, 2020)",
            "Tutoring improves outcomes (Smith, 2020).",
        ),
        (
            "Brown et al. (2018) concur.",
            "brown_2018",
            "Brown et al. (2018)",
            "Brown et al. (2018) concur.",
        ),
    ]
    assert coverage["by_source"] == {"mapping": 1, "author-year": 1, "numbered": 0}


def test_extract_claims_from_document_falls_back_when_the_linked_sentence_was_edited_away():
    """A link counts only while its sentence is still present in
    the paragraph's current text. Here the paragraph text no longer contains the
    sentence the link names at all (the user rewrote it) -- the stored link is ignored
    and the paragraph's actual, current sentence is what reaches the regex fallback."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"jones_2021": object()}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                "The user rewrote this sentence entirely (Jones, 2021).",
                citation_links=[
                    {
                        "sentence": "Tutoring improves outcomes (Smith, 2020).",
                        "keys": ["smith_2020"],
                        "citation_text": "(Smith, 2020)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (
            "The user rewrote this sentence entirely (Jones, 2021).",
            "jones_2021",
            "(Jones, 2021)",
            "The user rewrote this sentence entirely (Jones, 2021).",
        )
    ]
    assert coverage["by_source"] == {"mapping": 0, "author-year": 1, "numbered": 0}


def test_extract_claims_from_document_excludes_reference_list_nodes():
    """A reference-list entry that itself reads like a citable claim never becomes one:
    nodes at and after the references heading are used only by the numbered resolver."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"smith_2020": object()}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph("Tutoring improves outcomes (Smith, 2020)."),
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "References"}],
            },
            _doc_paragraph(
                "Jones, K. (2021). A different study that mentions Smith (2020) too."
            ),
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (
            "Tutoring improves outcomes (Smith, 2020).",
            "smith_2020",
            "(Smith, 2020)",
            "Tutoring improves outcomes (Smith, 2020).",
        )
    ]
    assert coverage["found"] == 1


def test_extract_claims_from_document_sends_one_claim_for_a_same_sentence_double_citation():
    """One claim per (sentence, citation_key): a source cited both narratively and
    parenthetically in one sentence asks the verifier the identical question twice, so it
    still costs one claim, one model call and one workbook row.

    `citation_audit.citation_spans` reports two citations in this
    sentence, and the unit of accounting is the citation, so ``found`` is 2 -- the
    coverage line counts citation renderings, the claim list counts questions."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"zhang_2018": object()}
    doc = {
        "type": "doc",
        "content": [_doc_paragraph("Zhang (2018) found X (Zhang, 2018).")],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (
            "Zhang (2018) found X (Zhang, 2018).",
            "zhang_2018",
            "Zhang (2018)",
            "Zhang (2018) found X (Zhang, 2018).",
        )
    ]
    assert coverage["found"] == 2
    assert coverage["linked"] == 2
    assert coverage["sent_to_verifier"] == 1


# --------------------------------------------------------------------------------------
# A sentence with two separately rendered citations must
# not lose the one the linker folded into the surviving link's own citation_text. The
# demo sentence is a reproduction
# (`demo/output/20260908-102739/draft_content.json`), tested in both directions.
# --------------------------------------------------------------------------------------

_DEMO_MULTI_CITATION_SENTENCE = (
    "Qualitative case studies such as Yallop et al. (2021) and Li and Hebert (2023) "
    "offer rich insights into feedback practices."
)


def test_extract_claims_from_document_with_a_map_still_finds_the_second_citation_in_a_sentence():
    """The model's link named only the first citation in `citation_text` (exactly what
    `CITATION_LINK_PROMPT` used to invite): after V5, keys=["yallop_2021"] survives on
    the link, with "li_2023" dropped. The span-aware fallback must still find and claim
    the second, uncovered citation."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"yallop_2021": object(), "li_2023": object()}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                _DEMO_MULTI_CITATION_SENTENCE,
                citation_links=[
                    {
                        "sentence": _DEMO_MULTI_CITATION_SENTENCE,
                        "keys": ["yallop_2021"],
                        "citation_text": "Yallop et al. (2021)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("yallop_2021", "Yallop et al. (2021)"),
        ("li_2023", "Li and Hebert (2023)"),
    }
    assert coverage["found"] == 2
    assert coverage["linked"] == 2
    assert coverage["unresolved"] == 0
    assert coverage["by_source"] == {"mapping": 1, "author-year": 1, "numbered": 0}


def test_extract_claims_from_document_without_a_map_finds_both_citations_in_a_sentence():
    """Control direction: the same sentence with no citationLinks at all must extract
    the identical two citations through the plain regex path, so the mapped and
    unmapped paths agree on coverage."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"yallop_2021": object(), "li_2023": object()}
    doc = {
        "type": "doc",
        "content": [_doc_paragraph(_DEMO_MULTI_CITATION_SENTENCE)],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("yallop_2021", "Yallop et al. (2021)"),
        ("li_2023", "Li and Hebert (2023)"),
    }
    assert coverage["found"] == 2
    assert coverage["linked"] == 2
    assert coverage["unresolved"] == 0
    assert coverage["by_source"] == {"mapping": 0, "author-year": 2, "numbered": 0}


# --------------------------------------------------------------------------------------
# Matching an uncovered citation to a link by recomputing a (surname, year) pair and
# comparing it against the pair `citation_audit.citation_spans` found in the whole
# sentence is unreliable: two independently written regexes are not guaranteed to agree
# on that pair (a capitalised lead-in word steals the citation audit's own surname; a
# square-bracket citation is not a span the citation audit recognises at all), so the
# fallback claim could be silently dropped, with coverage still reporting
# `unresolved: 0`. Matching by the POSITION of each side's own text inside the sentence
# instead avoids this. Each of the three sentences below is tested "with a map" (only
# one of the sentence's citations is linked) and "without a map" (the plain regex path)
# so the two directions can be compared directly.
# --------------------------------------------------------------------------------------

_SQUARE_BRACKET_AND_PAREN_SENTENCE = (
    "The effect held [Jones, 2021] and was replicated (Smith, 2020)."
)


def test_extract_claims_from_document_without_a_map_finds_a_square_bracket_and_a_paren_citation():
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"jones_2021": object(), "smith_2020": object()}
    doc = {"type": "doc", "content": [_doc_paragraph(_SQUARE_BRACKET_AND_PAREN_SENTENCE)]}
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("jones_2021", "[Jones, 2021]"),
        ("smith_2020", "(Smith, 2020)"),
    }
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 2,
        "sent_to_verifier": 2,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 0, "author-year": 2, "numbered": 0},
    }


def test_extract_claims_from_document_with_a_map_still_finds_the_square_bracket_citation():
    """`citation_spans` only looks inside round parentheses, so the square-bracket
    citation contributes no (surname, year) pair at all: the old pair-based comparison
    found `uncovered_pairs` empty after subtracting the link's own pair and skipped the
    fallback entirely, dropping "[Jones, 2021]" with no trace."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"jones_2021": object(), "smith_2020": object()}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                _SQUARE_BRACKET_AND_PAREN_SENTENCE,
                citation_links=[
                    {
                        "sentence": _SQUARE_BRACKET_AND_PAREN_SENTENCE,
                        "keys": ["smith_2020"],
                        "citation_text": "(Smith, 2020)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("jones_2021", "[Jones, 2021]"),
        ("smith_2020", "(Smith, 2020)"),
    }
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 2,
        "sent_to_verifier": 2,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 1, "author-year": 1, "numbered": 0},
    }


_LEAD_IN_WORD_SENTENCE = (
    "However, Zhang and Li (2019) reported the opposite effect, "
    "consistent with (Brown, 2021)."
)


def test_extract_claims_from_document_without_a_map_finds_the_narrative_and_the_paren_citation():
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"zhang_2019": object(), "brown_2021": object()}
    doc = {"type": "doc", "content": [_doc_paragraph(_LEAD_IN_WORD_SENTENCE)]}
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("zhang_2019", "However, Zhang and Li (2019)"),
        ("brown_2021", "(Brown, 2021)"),
    }
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 2,
        "sent_to_verifier": 2,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 0, "author-year": 2, "numbered": 0},
    }


def test_extract_claims_from_document_with_a_map_still_finds_the_citation_the_audit_mispairs():
    """`citation_audit.citation_spans` itself pairs this narrative citation with the
    lead-in word ("however", "2019"), not the true first author, because its own
    `_AUTHOR_TAIL` allows a comma-separated co-author list to absorb "Zhang" as a
    trailing name. `_extract_claims`'s own `_resolve_narrative_citation_key` resolves
    the same span to the true first author ("zhang_2019") instead. The old pair-based
    comparison rejected the fallback claim because `("zhang", "2019")` was never in the
    (surname, year) pairs the sentence-level audit reported uncovered."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"zhang_2019": object(), "brown_2021": object()}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                _LEAD_IN_WORD_SENTENCE,
                citation_links=[
                    {
                        "sentence": _LEAD_IN_WORD_SENTENCE,
                        "keys": ["brown_2021"],
                        "citation_text": "(Brown, 2021)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("zhang_2019", "However, Zhang and Li (2019)"),
        ("brown_2021", "(Brown, 2021)"),
    }
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 2,
        "sent_to_verifier": 2,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 1, "author-year": 1, "numbered": 0},
    }


# --------------------------------------------------------------------------------------
# A linked, verified citation must
# never be reported unresolved, and a citation rendered twice and linked twice must be
# counted twice, not half reported. Covered is decided by overlapping the citation
# audit's own reported span, reconstructed to its full extent, against a covered range --
# not by asking whether the audit's single reported position falls inside one.
# --------------------------------------------------------------------------------------

_LEAD_IN_WORD_CLAUSE_ONLY = "However, Zhang and Li (2019) reported the opposite effect."


def test_extract_claims_from_document_with_a_map_does_not_report_the_linked_citation_unresolved():
    """`citation_spans` reports this narrative citation at "However" (its own co-author
    absorption steals the lead-in word as part of the surname group), one or more tokens
    to the left of where the link's own `citation_text`, "Zhang and Li (2019)", starts.
    Before the fix, the audit's position fell outside the link's covered range, so the
    same citation that was linked and sent to the verifier was also reported unresolved,
    and `found` counted it twice."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"zhang_2019": object()}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                _LEAD_IN_WORD_CLAUSE_ONLY,
                citation_links=[
                    {
                        "sentence": _LEAD_IN_WORD_CLAUSE_ONLY,
                        "keys": ["zhang_2019"],
                        "citation_text": "Zhang and Li (2019)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (
            _LEAD_IN_WORD_CLAUSE_ONLY,
            "zhang_2019",
            "Zhang and Li (2019)",
            _LEAD_IN_WORD_CLAUSE_ONLY,
        )
    ]
    assert _public_coverage(coverage) == {
        "found": 1,
        "linked": 1,
        "sent_to_verifier": 1,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 1, "author-year": 0, "numbered": 0},
    }


def test_extract_claims_from_document_with_lead_in_word_citation_linked_still_finds_second():
    """Same lead-in-word citation, linked, plus the sentence's second, unlinked
    parenthetical citation (`_LEAD_IN_WORD_SENTENCE`): the fallback must still resolve
    "(Brown, 2021)", and the linked "Zhang and Li (2019)" must still not be reported
    unresolved -- `found` is 2, one per citation the audit sees, not 3."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"zhang_2019": object(), "brown_2021": object()}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                _LEAD_IN_WORD_SENTENCE,
                citation_links=[
                    {
                        "sentence": _LEAD_IN_WORD_SENTENCE,
                        "keys": ["zhang_2019"],
                        "citation_text": "Zhang and Li (2019)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("zhang_2019", "Zhang and Li (2019)"),
        ("brown_2021", "(Brown, 2021)"),
    }
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 2,
        "sent_to_verifier": 2,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 1, "author-year": 1, "numbered": 0},
    }


_TWICE_RENDERED_SENTENCE = "The effect (Smith, 2020) was replicated (Smith, 2020)."


def test_extract_claims_from_document_a_citation_rendered_twice_and_linked_twice_counts_twice():
    """Two separate links, each naming the identical `citation_text` because the same
    citation is rendered twice in the sentence: each link's search must land on its own
    occurrence, not the first one both times, so the two are counted once each --
    `found` and `linked` are both 2, matching the two spans `citation_spans` itself
    sees.

    Two renderings of one source in
    one sentence ask the verifier the identical (sentence, key) question, so they
    collapse into one claim, one model call and one workbook row. The coverage line a
    user reads ("2 found, 2 linked") is unchanged.
    """
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"smith_2020": object()}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                _TWICE_RENDERED_SENTENCE,
                citation_links=[
                    {
                        "sentence": _TWICE_RENDERED_SENTENCE,
                        "keys": ["smith_2020"],
                        "citation_text": "(Smith, 2020)",
                    },
                    {
                        "sentence": _TWICE_RENDERED_SENTENCE,
                        "keys": ["smith_2020"],
                        "citation_text": "(Smith, 2020)",
                    },
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (
            _TWICE_RENDERED_SENTENCE,
            "smith_2020",
            "(Smith, 2020)",
            _TWICE_RENDERED_SENTENCE,
        )
    ]
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 2,
        "sent_to_verifier": 1,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 2, "author-year": 0, "numbered": 0},
    }


_TWO_WORD_SURNAME_SENTENCE = (
    "While experimental evidence favours direct correction for long-term accuracy "
    "gains (Bonilla Lopez et al., 2018), naturalistic studies indicate that teachers "
    "benefit from delayed feedback (Mao, 2024)."
)


def test_extract_claims_from_document_without_a_map_reports_the_two_word_surname_unresolved():
    """Control direction: `_extract_claims`'s own `_CITATION_RE` cannot parse a
    two-word surname followed by "et al." inside one parenthetical citation at all
    (unlike `citation_audit`'s richer `_PAREN_CITE`), so this citation is still never
    claimed by either regex without a map.

    Coverage counts citation units everywhere: the citation audit's spans are consulted
    for every sentence, not only one with a surviving link, so an unclaimable citation
    in an unlinked paragraph is not invisible to coverage. The two directions agree: two
    citations found, the one the regexes cannot parse still keyed and claimed from the
    citation audit's own (surname, year), and reported unresolved because the key it
    built ("lopez_2018") is not in this test's library, not because it has no key at
    all."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"mao_2024": object()}
    doc = {"type": "doc", "content": [_doc_paragraph(_TWO_WORD_SURNAME_SENTENCE)]}
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("mao_2024", "(Mao, 2024)"),
        ("lopez_2018", "(Bonilla Lopez et al., 2018)"),
    }
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 1,
        "sent_to_verifier": 2,
        "unresolved": 1,
        "unresolved_citations": ["(Bonilla Lopez et al., 2018)"],
        "by_source": {"mapping": 0, "author-year": 2, "numbered": 0},
    }


def test_extract_claims_from_document_with_a_map_reports_the_two_word_surname_citation_unresolved():
    """If the linker links only "Mao" and not "Bonilla Lopez", the second
    citation must never vanish silently -- `citation_spans` still sees it (paired with
    "lopez", not "bonilla lopez", by the citation audit's own recognition), so it is
    still claimed, under the key the audit's own surname and year build, and reported
    unresolved because that key is absent from this test's library, with its own
    rendered text rather than dropped with `unresolved: 0` asserting a coverage that is
    not real. The rendered text is the whole citation, "(Bonilla Lopez et al., 2018)",
    not the truncated, unbalanced fragment starting at "Lopez": a parenthetical group
    with a single member renders with its own brackets.

    ``by_source`` reports ``{"mapping": 1, "author-year": 1}``: the unresolved
    citation is an author-year citation the regexes could not resolve, not something
    the writer's own map produced. It is given a key from the citation audit itself, so
    it is sent to the verifier (and would surface as `no_full_text`) instead of never
    being asked about at all."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"mao_2024": object()}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                _TWO_WORD_SURNAME_SENTENCE,
                citation_links=[
                    {
                        "sentence": _TWO_WORD_SURNAME_SENTENCE,
                        "keys": ["mao_2024"],
                        "citation_text": "(Mao, 2024)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("mao_2024", "(Mao, 2024)"),
        ("lopez_2018", "(Bonilla Lopez et al., 2018)"),
    }
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 1,
        "sent_to_verifier": 2,
        "unresolved": 1,
        "unresolved_citations": ["(Bonilla Lopez et al., 2018)"],
        "by_source": {"mapping": 1, "author-year": 1, "numbered": 0},
    }


# --------------------------------------------------------------------------------------
# A link whose every key was
# dropped by validation (V5) could still update `linked_pairs` from its own
# `citation_text`, marking the span covered and blocking the regex fallback for it -- a
# citation the fallback would have resolved perfectly would go unverified purely because
# of a model error the validator had already caught.
# --------------------------------------------------------------------------------------


def test_extract_claims_from_document_a_keyless_link_still_lets_the_fallback_resolve_its_span():
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"smith_2020": object()}
    sentence = "The replication failed (Smith, 2020)."
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    {"sentence": sentence, "keys": [], "citation_text": "(Smith, 2020)"}
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [(sentence, "smith_2020", "(Smith, 2020)", sentence)]
    assert _public_coverage(coverage) == {
        "found": 1,
        "linked": 1,
        "sent_to_verifier": 1,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 0, "author-year": 1, "numbered": 0},
    }


# --------------------------------------------------------------------------------------
# A citation inside a list item or a blockquote
# must be extracted and counted exactly like a top-level paragraph.
# --------------------------------------------------------------------------------------


def test_extract_claims_from_document_descends_into_list_items_and_blockquotes():
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {
        "smith_2020": object(),
        "brown_2018": object(),
        "jones_2021": object(),
    }
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph("Tutoring improves outcomes (Smith, 2020)."),
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [_doc_paragraph("Brown et al. (2018) concur.")],
                    }
                ],
            },
            {
                "type": "blockquote",
                "content": [_doc_paragraph("Jones (2021) offers a dissenting view.")],
            },
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (
            "Tutoring improves outcomes (Smith, 2020).",
            "smith_2020",
            "(Smith, 2020)",
            "Tutoring improves outcomes (Smith, 2020).",
        ),
        (
            "Brown et al. (2018) concur.",
            "brown_2018",
            "Brown et al. (2018)",
            "Brown et al. (2018) concur.",
        ),
        (
            "Jones (2021) offers a dissenting view.",
            "jones_2021",
            "Jones (2021)",
            "Jones (2021) offers a dissenting view.",
        ),
    ]
    assert coverage["found"] == 3
    assert coverage["linked"] == 3


# --------------------------------------------------------------------------------------
# The mapping path must dedupe on
# (sentence, citation_key) exactly as the regex path already does.
# --------------------------------------------------------------------------------------


def test_extract_claims_from_document_dedupes_a_link_with_a_duplicated_key():
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"smith_2020": object()}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                "Tutoring improves outcomes (Smith, 2020).",
                citation_links=[
                    {
                        "sentence": "Tutoring improves outcomes (Smith, 2020).",
                        "keys": ["smith_2020", "smith_2020"],
                        "citation_text": "(Smith, 2020)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (
            "Tutoring improves outcomes (Smith, 2020).",
            "smith_2020",
            "(Smith, 2020)",
            "Tutoring improves outcomes (Smith, 2020).",
        )
    ]
    assert coverage["found"] == 1
    assert coverage["linked"] == 1
    assert coverage["by_source"]["mapping"] == 1


def test_extract_claims_from_document_dedupes_the_same_link_returned_twice():
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"smith_2020": object()}
    link = {
        "sentence": "Tutoring improves outcomes (Smith, 2020).",
        "keys": ["smith_2020"],
        "citation_text": "(Smith, 2020)",
    }
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                "Tutoring improves outcomes (Smith, 2020).",
                citation_links=[link, dict(link)],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (
            "Tutoring improves outcomes (Smith, 2020).",
            "smith_2020",
            "(Smith, 2020)",
            "Tutoring improves outcomes (Smith, 2020).",
        )
    ]
    assert coverage["found"] == 1
    assert coverage["linked"] == 1
    assert coverage["by_source"]["mapping"] == 1


# --------------------------------------------------------------------------------------
# A link whose sentence the claim
# splitter breaks into two fragments could otherwise be resolved once per fragment,
# counting its citation twice, verifying it twice and writing it to the report twice.
# Links are resolved once per paragraph, against the paragraph's own citation units.
# --------------------------------------------------------------------------------------


def test_extract_claims_from_document_counts_a_link_once_when_et_al_splits_the_sentence():
    """`_CLAIM_SENTENCE_SPLIT_RE` splits at "et al. " when
    no parenthesised year follows, so this one sentence is two fragments and the link
    covers both. Without deduping per paragraph, this would count `found 2, linked 2,
    sent_to_verifier 2` for one citation the audit sees once."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"smith_2020": object()}
    sentence = "Smith et al. reported large gains (Smith, 2020)."
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    {
                        "sentence": sentence,
                        "keys": ["smith_2020"],
                        "citation_text": "(Smith, 2020)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (
            "reported large gains (Smith, 2020).",
            "smith_2020",
            "(Smith, 2020)",
            "reported large gains (Smith, 2020).",
        )
    ]
    assert _public_coverage(coverage) == {
        "found": 1,
        "linked": 1,
        "sent_to_verifier": 1,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 1, "author-year": 0, "numbered": 0},
    }


def test_extract_claims_from_document_counts_a_link_once_when_an_abbreviation_splits_it():
    """The same shape from an ordinary abbreviation ("U.S.") rather than "et al."."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"smith_2020": object()}
    sentence = "Gains were reported in the U.S. by prior work (Smith, 2020)."
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    {
                        "sentence": sentence,
                        "keys": ["smith_2020"],
                        "citation_text": "(Smith, 2020)",
                    }
                ],
            )
        ],
    }
    _claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert coverage["found"] == 1
    assert coverage["linked"] == 1
    assert coverage["sent_to_verifier"] == 1


def test_extract_claims_from_document_counts_a_link_once_despite_a_whitespace_difference():
    """`validate_citation_link` compares whitespace-normalised (V1/V2), so a link whose
    ``citation_text`` writes one space where the draft has two is accepted and keeps its
    key. Resolution normalises whitespace on both sides too, so the link lands on the
    citation instead of finding nothing and letting the fallback claim the same citation
    a second time (`found 2, linked 2, sent 2` before the redesign)."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"jones_2021": object()}
    sentence = "The effect was replicated (Jones,  2021)."
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    {
                        "sentence": sentence,
                        "keys": ["jones_2021"],
                        "citation_text": "(Jones, 2021)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [(sentence, "jones_2021", "(Jones, 2021)", sentence)]
    assert coverage["found"] == 1
    assert coverage["linked"] == 1
    assert coverage["sent_to_verifier"] == 1


def test_extract_claims_from_document_counts_a_link_once_when_its_sentence_spans_two():
    """A link whose ``sentence`` covers two document sentences (the model returned both)
    still resolves to exactly one citation unit."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"smith_2020": object()}
    sentence = "The effect was large (Smith, 2020). It replicated widely."
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    {
                        "sentence": sentence,
                        "keys": ["smith_2020"],
                        "citation_text": "(Smith, 2020)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (
            "The effect was large (Smith, 2020).",
            "smith_2020",
            "(Smith, 2020)",
            "The effect was large (Smith, 2020).",
        )
    ]
    assert coverage["found"] == 1
    assert coverage["linked"] == 1
    assert coverage["sent_to_verifier"] == 1


def test_extract_claims_from_document_counts_a_grouped_citation_once_per_member():
    """Round 4's multiplying case: a grouped citation plus a fragment split, one link per
    citation. Before the redesign this reported `found 4, linked 4, sent 4` for the two
    citations the audit sees."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"lee_2020": object(), "storch_2018": object()}
    sentence = "Uptake varied (Lee, 2020; Storch, 2018). Teachers differed e.g. in feedback."
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    {"sentence": sentence, "keys": ["lee_2020"], "citation_text": "Lee, 2020"},
                    {
                        "sentence": sentence,
                        "keys": ["storch_2018"],
                        "citation_text": "Storch, 2018",
                    },
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("lee_2020", "Lee, 2020"),
        ("storch_2018", "Storch, 2018"),
    }
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 2,
        "sent_to_verifier": 2,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 2, "author-year": 0, "numbered": 0},
    }


# --------------------------------------------------------------------------------------
# Every citation inside a
# semicolon-separated parenthetical group could otherwise reconstruct to the whole
# group, so one link on one member would make every member look covered and the rest
# would be dropped with no claim and no unresolved entry. Each member has its own
# extent.
# --------------------------------------------------------------------------------------


def test_extract_claims_from_document_reports_the_unlinked_member_of_a_group_unresolved():
    """The unlinked
    member ("Lee, 2020") is given a key from the citation audit's own (surname, year)
    instead of leaving it keyless: since the project library holds `lee_2020` too, it
    resolves rather than staying unresolved."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"lee_2020": object(), "storch_2018": object()}
    sentence = "Teacher practice diverges (Lee, 2020; Storch, 2018) in classrooms."
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    {
                        "sentence": sentence,
                        "keys": ["storch_2018"],
                        "citation_text": "Storch, 2018",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (sentence, "lee_2020", "Lee, 2020", sentence),
        (sentence, "storch_2018", "Storch, 2018", sentence),
    ]
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 2,
        "sent_to_verifier": 2,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 1, "author-year": 1, "numbered": 0},
    }


# --------------------------------------------------------------------------------------
# A link's `citation_text` naming
# the whole group, rather than one member of it, could otherwise consume just the first
# unconsumed unit and hand it every key -- a group-wide link consumes every unit its
# occurrence overlaps and distributes the keys by surname and year.
# --------------------------------------------------------------------------------------

_GROUP_WIDE_LINK_SENTENCE = "Evidence is mixed (Lee, 2020; Storch, 2018) in the field."


def test_extract_claims_from_document_a_group_wide_link_with_both_keys_resolves_every_member():
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"lee_2020": object(), "storch_2018": object()}
    sentence = _GROUP_WIDE_LINK_SENTENCE
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    {
                        "sentence": sentence,
                        "keys": ["lee_2020", "storch_2018"],
                        "citation_text": "(Lee, 2020; Storch, 2018)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("lee_2020", "Lee, 2020"),
        ("storch_2018", "Storch, 2018"),
    }
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 2,
        "sent_to_verifier": 2,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 2, "author-year": 0, "numbered": 0},
    }


def test_extract_claims_from_document_a_group_wide_link_with_one_key_credits_the_right_member():
    """Without this, the group-wide `citation_text` would match the first unconsumed unit
    (Lee), which would take the Storch key by accident: "Lee, 2020" would never be
    claimed under its own key at all, and "Storch, 2018" would surface only once,
    mis-sourced to "mapping". The key is matched to the member it actually names, so Lee
    is left for the audit-key fallback, which resolves it separately under its own key."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"lee_2020": object(), "storch_2018": object()}
    sentence = _GROUP_WIDE_LINK_SENTENCE
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    {
                        "sentence": sentence,
                        "keys": ["storch_2018"],
                        "citation_text": "(Lee, 2020; Storch, 2018)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (sentence, "lee_2020", "Lee, 2020", sentence),
        (sentence, "storch_2018", "Storch, 2018", sentence),
    ]
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 2,
        "sent_to_verifier": 2,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 1, "author-year": 1, "numbered": 0},
    }


# --------------------------------------------------------------------------------------
# The single-target branch above
# counts a link that claims nothing in `diagnostics["links_ignored"]`; the multi-target
# branch does too. A group-wide link whose keys name neither member it overlaps leaves
# every target unconsumed, and would otherwise end the loop without incrementing
# anything.
# --------------------------------------------------------------------------------------


def test_extract_claims_from_document_counts_a_group_wide_link_matching_no_member_as_ignored():
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"lee_2020": object(), "storch_2018": object()}
    sentence = "Mixed evidence (Lee, 2020; Storch, 2018) exists."
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    {
                        "sentence": sentence,
                        "keys": ["nobody_1999", "other_1998"],
                        "citation_text": "(Lee, 2020; Storch, 2018)",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    # The user-facing counts stay correct -- the author-year fallback keys both members
    # from the text itself, since the link claimed neither.
    assert claims == [
        (sentence, "lee_2020", "Lee, 2020", sentence),
        (sentence, "storch_2018", "Storch, 2018", sentence),
    ]
    assert coverage["diagnostics"] == {
        "links_seen": 1,
        "links_stale": 0,
        "links_without_keys": 0,
        "links_ignored": 1,
        "units_linked": 0,
        "units_fallback": 2,
        "units_numbered": 0,
        "units_without_keys": 0,
    }


def test_extract_claims_from_document_reports_both_unlinked_members_of_a_three_way_group():
    """From the demo output's own prose: three
    citations in one group, one linked, would otherwise leave two dropped silently and
    reported unresolved with no claim, even though the project library holds both papers.
    Both resolve from the citation audit's own (surname, year)."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {
        "koltovskaia_2022": object(),
        "luo_2025": object(),
        "hyland_2025": object(),
    }
    sentence = (
        "AI feedback uptake varies widely (Koltovskaia, 2022; Luo et al., 2025; Hyland, 2025)."
    )
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    {
                        "sentence": sentence,
                        "keys": ["koltovskaia_2022"],
                        "citation_text": "Koltovskaia, 2022",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (sentence, "koltovskaia_2022", "Koltovskaia, 2022", sentence),
        (sentence, "luo_2025", "Luo et al., 2025", sentence),
        (sentence, "hyland_2025", "Hyland, 2025", sentence),
    ]
    assert coverage["found"] == 3
    assert coverage["linked"] == 3
    assert coverage["unresolved"] == 0
    assert coverage["unresolved_citations"] == []
    assert coverage["by_source"] == {"mapping": 1, "author-year": 2, "numbered": 0}


def test_extract_claims_from_document_counts_a_grouped_citation_in_an_unlinked_paragraph():
    """A grouped citation in a paragraph with no link at all would otherwise give
    `found 0`, because the citation audit's spans would be consulted only for a sentence
    that had a surviving link. Coverage counts citation units everywhere, and each member
    is keyed from the citation audit's own (surname, year), so both resolve here exactly
    as the mapped direction resolves them for the same text."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"lee_2020": object(), "storch_2018": object()}
    sentence = "Teacher practice diverges (Lee, 2020; Storch, 2018) in classrooms."
    doc = {"type": "doc", "content": [_doc_paragraph(sentence)]}
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [
        (sentence, "lee_2020", "Lee, 2020", sentence),
        (sentence, "storch_2018", "Storch, 2018", sentence),
    ]
    assert _public_coverage(coverage) == {
        "found": 2,
        "linked": 2,
        "sent_to_verifier": 2,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 0, "author-year": 2, "numbered": 0},
    }


# --------------------------------------------------------------------------------------
# A citation the citation audit
# itself parsed into a surname and a year -- an "as cited in" attribution, or a
# two-token surname the audit's own backtracking resolves to its last token, "see" or
# similar lowercase lead-in and all -- would otherwise be reported unresolved with no
# key and no claim, even when the project library held the paper, because neither
# `_CITATION_RE` nor `_NARRATIVE_CITE` can parse the shape at all.
# --------------------------------------------------------------------------------------


def test_extract_claims_from_document_resolves_an_as_cited_in_attribution_from_the_audit_key():
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"storch_2018": object()}
    sentence = "The claim (Old, 1990, as cited in Storch, 2018) still stands."
    doc = {"type": "doc", "content": [_doc_paragraph(sentence)]}
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [(sentence, "storch_2018", "(Old, 1990, as cited in Storch, 2018)", sentence)]
    assert _public_coverage(coverage) == {
        "found": 1,
        "linked": 1,
        "sent_to_verifier": 1,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 0, "author-year": 1, "numbered": 0},
    }


def test_extract_claims_from_document_resolves_a_see_lead_in_two_token_surname_from_the_audit_key():
    """The citation audit's own backtracking resolves "(see Bonilla Lopez et al., 2018)"
    to the surname "Lopez" (the lowercase "see" and the unparseable "Bonilla Lopez" tail
    both fail `_PAREN_CITE`'s own attempt, so the engine advances to the token that does
    match), which is exactly the key `_build_paper_lookup` would index this paper under
    if its first author were stored as "Lopez, ..."."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"lopez_2018": object()}
    sentence = "Direct correction helps (see Bonilla Lopez et al., 2018) in most settings."
    doc = {"type": "doc", "content": [_doc_paragraph(sentence)]}
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [(sentence, "lopez_2018", "(see Bonilla Lopez et al., 2018)", sentence)]
    assert _public_coverage(coverage) == {
        "found": 1,
        "linked": 1,
        "sent_to_verifier": 1,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 0, "author-year": 1, "numbered": 0},
    }


def test_extract_claims_from_document_leaves_a_placeholder_year_citation_keyless():
    """The one case the audit-key fallback must never touch: `_year_key` returns `None`
    for a placeholder such as "n.d.", so there is no real year to build a key from, and
    the citation stays exactly as unresolved and keyless as it was before this fix."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"mao_2024": object()}
    sentence = "Mao's (n.d.) review is still cited."
    doc = {"type": "doc", "content": [_doc_paragraph(sentence)]}
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == []
    assert _public_coverage(coverage) == {
        "found": 1,
        "linked": 0,
        "sent_to_verifier": 0,
        "unresolved": 1,
        "unresolved_citations": ["Mao's (n.d.)"],
        "by_source": {"mapping": 0, "author-year": 1, "numbered": 0},
    }


# --------------------------------------------------------------------------------------
# `unresolved_citations` is
# supposed to hold only balanced, non-empty text. That holds for `unit.rendered`, which
# this module builds itself, but not for a `citation_text` a link supplied and left the
# unit unresolved (a key absent from the library): `validate_citation_link` never checks
# bracket balance. A second consumer is also left unbalanced: the claim
# `unit.citation_text` feeds the verifier prompt and the workbook row with.
# `_assign_links_to_units` falls back to `unit.rendered` at the point it sets
# `citation_text` from a link, not only where `unresolved_citations` displays it, so
# both consumers see the same balanced string.
# --------------------------------------------------------------------------------------


def test_extract_claims_from_document_falls_back_to_rendered_when_the_links_text_is_unbalanced():
    from app.services.fulltext import extract_claims_from_document

    paper_lookup: dict[str, object] = {}
    sentence = "Findings (as in Smith (2020)) were mixed."
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    {
                        "sentence": sentence,
                        "keys": ["smith_2020"],
                        "citation_text": "Smith (2020))",
                    }
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [(sentence, "smith_2020", "Smith (2020)", sentence)]
    assert coverage["unresolved"] == 1
    assert coverage["unresolved_citations"] == ["Smith (2020)"]


# --------------------------------------------------------------------------------------
# The accounting invariants, on the documents the
# named reproductions above use. The generated-document version of the same assertions
# lives in `test_citation_coverage_properties.py`.
# --------------------------------------------------------------------------------------


def test_extract_claims_from_document_coverage_invariants_hold_on_the_reproductions():
    from app.services.citation_audit import citation_spans
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {
        "smith_2020": object(),
        "brown_2021": object(),
        "zhang_2019": object(),
        "lee_2020": object(),
        "storch_2018": object(),
        "mao_2024": object(),
    }
    sentences = [
        _LEAD_IN_WORD_SENTENCE,
        _LEAD_IN_WORD_CLAUSE_ONLY,
        _TWICE_RENDERED_SENTENCE,
        _TWO_WORD_SURNAME_SENTENCE,
        "Teacher practice diverges (Lee, 2020; Storch, 2018) in classrooms.",
        "Smith et al. reported large gains (Smith, 2020).",
    ]
    for sentence in sentences:
        for links in (
            None,
            [{"sentence": sentence, "keys": ["smith_2020"], "citation_text": "(Smith, 2020)"}],
        ):
            doc = {"type": "doc", "content": [_doc_paragraph(sentence, citation_links=links)]}
            claims, coverage = extract_claims_from_document(doc, paper_lookup)
            assert coverage["found"] == len({p for p, *_ in citation_spans(sentence)}), sentence
            assert coverage["linked"] + coverage["unresolved"] == coverage["found"]
            assert sum(coverage["by_source"].values()) == coverage["found"]
            assert coverage["sent_to_verifier"] == len(claims)
            assert len({(s, k) for s, k, _c, _cs in claims}) == len(claims)


# --------------------------------------------------------------------------------------
# The verification prompt and the guards are frozen. The
# coverage work changes extraction and counting only, so the prompt sha and the source of
# every guard function must be exactly what they were before it.
# --------------------------------------------------------------------------------------


def test_claim_verification_prompt_version_is_still_the_frozen_sha():
    from app.agents.claim_verification_agent import CLAIM_VERIFICATION_PROMPT_VERSION

    assert CLAIM_VERIFICATION_PROMPT_VERSION == "sha256:49fcfbfaf2f6"


def test_verification_guard_functions_are_unchanged():
    """A digest over the source of every guard `apply_verification_guards` runs (plus every
    lexicon, band table, closed word list, span and overlap minimum a demotion decision
    passes through, added by value -- the same reasoning `_compute_guard_digest`'s own
    docstring states). Changing a guard is a deliberate act that updates this digest with
    its own reasoning; an accidental edit while working on something else fails here."""
    import hashlib
    import inspect

    from app.services import fulltext

    names = [
        "apply_verification_guards",
        "_guard_no_full_text_with_chunks",
        "_guard_attribution",
        "_guard_assertion_status_consistency",
        "_guard_numeric_tier_b",
        "_guard_numeric_value_absent",
        "_guard_quote_fidelity",
        "_guard_scale_fidelity",
        "scale_alignment_findings",
        "_scale_negation_scope_shift",
        "_scale_tokens",
        "_scale_stem",
        "_scale_is_content_token",
        "_scale_content_indices",
        "_scale_head_set",
        "_scale_context_set",
        "_scale_occurrences",
        "_scale_admissible_pairs",
        "_scale_align",
    ]
    digest = hashlib.sha256()
    for name in names:
        digest.update(inspect.getsource(getattr(fulltext, name)).encode("utf-8"))
    for value in (
        fulltext._SCALE_CURLY_FOLD,
        fulltext._SCALE_EXTENT_WORDS,
        fulltext._SCALE_DEGREE_WORDS,
        fulltext._SCALE_HEDGE_WORDS,
        fulltext._SCALE_BAND_BY_RANK,
    ):
        digest.update(repr(value).encode("utf-8"))
    digest.update(repr(sorted(fulltext._SCALE_FUNCTION_WORDS)).encode("utf-8"))
    digest.update(repr(fulltext._SCALE_EPISTEMIC_VERB_STEMS).encode("utf-8"))
    digest.update(repr(sorted(fulltext._SCALE_ABSOLUTE_NEGATORS)).encode("utf-8"))
    digest.update(repr(sorted(fulltext._SCALE_NEGATORS)).encode("utf-8"))
    for value in (
        fulltext._SCALE_HEAD_RIGHT_SPAN,
        fulltext._SCALE_HEAD_RIGHT_MAX,
        fulltext._SCALE_HEAD_LEFT_SPAN,
        fulltext._SCALE_HEAD_LEFT_MAX,
        fulltext._SCALE_CONTEXT_SPAN,
        fulltext._SCALE_CONTEXT_MAX,
        fulltext._SCALE_MIN_HEAD_OVERLAP,
        fulltext._SCALE_MIN_CONTEXT_OVERLAP,
    ):
        digest.update(repr(value).encode("utf-8"))
    assert digest.hexdigest()[:16] == fulltext.GUARD_DIGEST == "8a6c833ffc329f89"


# --------------------------------------------------------------------------------------
# Two consequences of counting the audit's own
# spans, each pinned here so a later change has to argue with a test rather than a
# comment.
# --------------------------------------------------------------------------------------


def test_extract_claims_from_document_reports_an_undated_citation_as_unresolved():
    """`citation_audit` counts "(Smith, n.d.)" as a citation and no key can ever be built
    for it (`_build_paper_lookup` only indexes real four-digit years), so it is reported
    unresolved. Before the redesign it was skipped outright and `found` was 0, which told
    the user a draft citing an undated source had no citation there at all."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"smith_2020": object()}
    sentence = "The approach is long established (Smith, n.d.)."
    doc = {"type": "doc", "content": [_doc_paragraph(sentence)]}
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == []
    assert _public_coverage(coverage) == {
        "found": 1,
        "linked": 0,
        "sent_to_verifier": 0,
        "unresolved": 1,
        "unresolved_citations": ["(Smith, n.d.)"],
        "by_source": {"mapping": 0, "author-year": 1, "numbered": 0},
    }


def test_extract_claims_from_document_counts_an_ignored_link_in_diagnostics():
    """A link that matches no unit AT ALL -- here one whose own ``citation_text`` the
    paragraph never renders -- contributes nothing and is counted, so a coverage line
    that looks wrong can be traced to the map rather than guessed at.

    The same link returned twice for one rendered citation is not one of these --
    the document-level unit model REUSES the unit an earlier, identical link already
    claimed (as an extra resolution carrying the same fallback claim text, so the two
    collapse to the one claim below by the pre-existing dedup rule) rather than dropping
    the duplicate. That case is exercised on its own in
    `test_citation_link_proposition.py::test_two_links_naming_the_same_rendered_citation_with_different_propositions_both_yield_a_claim`."""
    from app.services.fulltext import extract_claims_from_document

    paper_lookup = {"smith_2020": object()}
    sentence = "Tutoring improves outcomes (Smith, 2020)."
    link = {"sentence": sentence, "keys": ["smith_2020"], "citation_text": "(Smith, 2020)"}
    doc = {
        "type": "doc",
        "content": [
            _doc_paragraph(
                sentence,
                citation_links=[
                    link,
                    dict(link),
                    {"sentence": sentence, "keys": [], "citation_text": "(Smith, 2020)"},
                    {"sentence": "A sentence the user deleted (Jones, 2021).", "keys": [],
                     "citation_text": "(Jones, 2021)"},
                    {"sentence": sentence, "keys": ["nowhere_2099"],
                     "citation_text": "(Nowhere, 2099)"},
                ],
            )
        ],
    }
    claims, coverage = extract_claims_from_document(doc, paper_lookup)
    assert claims == [(sentence, "smith_2020", "(Smith, 2020)", sentence)]
    assert coverage["found"] == 1
    assert coverage["diagnostics"] == {
        "links_seen": 5,
        "links_stale": 1,
        "links_without_keys": 1,
        "links_ignored": 1,
        "units_linked": 1,
        "units_fallback": 0,
        "units_numbered": 0,
        "units_without_keys": 0,
    }

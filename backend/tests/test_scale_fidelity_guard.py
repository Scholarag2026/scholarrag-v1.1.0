"""Guard 7: scale-word fidelity. Every sub-rule, every decision constant and
every rejected variant gets its own test here, independent of the DB-backed pipeline
(mirrors ``test_verification_guards.py``'s own convention: one file per guard family,
in-memory, no database).
"""

from __future__ import annotations

import pytest

# --------------------------------------------------------------------------------------
# The gate (design section 5, point 1): guard 7 only ever runs on status == "verified".
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["needs_nuance", "unsupported", "no_full_text", "error"])
def test_guard_scale_fidelity_passes_through_a_non_verified_status_unchanged(status):
    from app.services.fulltext import _guard_scale_fidelity

    result = _guard_scale_fidelity(
        status,
        claim_text="Participants seldom critically engaged with local comments.",
        evidence_quote="None reported that they critically engaged with the comments.",
    )
    assert result == (status, [])


def test_guard_scale_fidelity_passes_when_there_is_no_evidence_at_all():
    """Guard 7 never treats "nothing to compare" as evidence of a mismatch -- that is guard
    3's (quote fidelity's) job, not this one's (design section 1: "nothing more" than a
    force comparison)."""
    from app.services.fulltext import _guard_scale_fidelity

    result = _guard_scale_fidelity(
        "verified",
        claim_text="Recall improved dramatically according to the authors.",
        evidence_quote=None,
        evidence_quotes=None,
    )
    assert result == ("verified", [])


# --------------------------------------------------------------------------------------
# Rule A, the four Yallop extent rows (design section 2, rows 3/12/17/23): the source's
# "none" against the claim's "seldom", reached across the complementizer in "none reported
# that they critically engaged" -- the escape a three-token head window misses and a
# six-token one catches (design section 4 and section 9).
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "claim_text",
    [
        "Participants seldom critically engaged with local comments.",
        "Students seldom engaged critically with local visible revision comments.",
        "Learners seldom critically engaged with local comments during the task.",
        "Writers seldom engaged critically with local visible revisions.",
    ],
)
def test_guard_scale_fidelity_demotes_the_four_yallop_extent_rows(claim_text):
    from app.services.fulltext import _guard_scale_fidelity

    quote = "None reported that they critically engaged with the comments."
    status, reasons = _guard_scale_fidelity(
        "verified", claim_text=claim_text, evidence_quote=quote
    )
    assert status == "needs_nuance"
    assert reasons == ["scale_word_mismatch"]


def test_guard_scale_fidelity_demotes_a_degree_mismatch():
    """Row 31 shape (design section 2): the source's "considerably" (band 2) against the
    claim's "dramatically" (band 3)."""
    from app.services.fulltext import _guard_scale_fidelity

    status, reasons = _guard_scale_fidelity(
        "verified",
        claim_text="The recall of the model improved dramatically after fine-tuning.",
        evidence_quote="The recall of the model considerably improved after fine-tuning.",
    )
    assert status == "needs_nuance"
    assert reasons == ["scale_word_mismatch"]


def test_guard_scale_fidelity_uses_content_anchors_not_the_nearest_same_word():
    """Design section 4: the alignment is over content-word anchors, not a nearest match --
    a nearest-match rule would let the claim's "seldom" pair with the quote's own "seldom"
    (present here on an unrelated predicate) and pass every one of the Yallop rows. The
    correct counterpart is "none", found through the shared "critically engaged ...
    comments" anchor, not through the repeated surface word."""
    from app.services.fulltext import _guard_scale_fidelity, scale_alignment_findings

    claim = "Participants seldom critically engaged with local comments."
    quote = (
        "Participants seldom viewed the interface, and none reported that they "
        "critically engaged with the comments."
    )
    status, reasons = _guard_scale_fidelity("verified", claim_text=claim, evidence_quote=quote)
    assert status == "needs_nuance"
    assert reasons == ["scale_word_mismatch"]
    findings = scale_alignment_findings(claim, [quote])
    assert len(findings) == 1
    assert findings[0]["claim_word"] == "seldom"
    assert findings[0]["quote_word"] == "none"


def test_guard_scale_fidelity_one_to_one_assignment_keeps_two_pairs_straight():
    """Design section 4: a claim carrying two scale words whose counterparts are both
    present must pair each occurrence once -- "all" with the quote's "all", "no" with the
    quote's "no" -- rather than either claim word taking a counterpart already spoken for."""
    from app.services.fulltext import _guard_scale_fidelity, scale_alignment_findings

    claim = "All participants completed the task and no participant reported any difficulty."
    quote = (
        "All six participants completed the reading task; none of them reported any "
        "difficulty completing it."
    )
    status, reasons = _guard_scale_fidelity("verified", claim_text=claim, evidence_quote=quote)
    assert status == "verified"
    assert reasons == []
    assert scale_alignment_findings(claim, [quote]) == []


# --------------------------------------------------------------------------------------
# Band folding (design section 3): near-synonym ranks collapse into the same band so the
# guard stays quiet on a step no reviewer would call a shift.
# --------------------------------------------------------------------------------------


def test_guard_scale_fidelity_does_not_fire_on_a_one_band_extent_step():
    """Extent ranks 2 (occasionally/sometimes/some/several) and 3 (often/frequently/
    regularly/commonly/repeatedly/many/much/numerous) both fold to band 2."""
    from app.services.fulltext import _guard_scale_fidelity

    status, reasons = _guard_scale_fidelity(
        "verified",
        claim_text="Participants often reported feeling more confident afterwards.",
        evidence_quote="Participants sometimes reported feeling more confident afterwards.",
    )
    assert status == "verified"
    assert reasons == []


def test_guard_scale_fidelity_does_not_fire_on_a_one_band_degree_step():
    """Design section 3: degree band 2 collapses considerable/substantial with marked/
    strong/sharp, so "substantial" against "strong" does not fire -- the rejected unbanded
    alternative fired here (design section 12, "banding hides one step of the scale")."""
    from app.services.fulltext import _guard_scale_fidelity

    status, reasons = _guard_scale_fidelity(
        "verified",
        claim_text="Motivation showed a substantial change in completion rates.",
        evidence_quote="Motivation showed a strong change in completion rates.",
    )
    assert status == "verified"
    assert reasons == []


# --------------------------------------------------------------------------------------
# The three exclusions (design section 3): "significant"/"significantly", reporting verbs,
# and number words are never lexicon members, each for its own stated reason.
# --------------------------------------------------------------------------------------


def test_guard_scale_fidelity_does_not_treat_significant_as_a_degree_word():
    from app.services.fulltext import _guard_scale_fidelity

    status, reasons = _guard_scale_fidelity(
        "verified",
        claim_text="The intervention had a significant effect on scores.",
        evidence_quote="The intervention had a considerable effect on scores.",
    )
    assert status == "verified"
    assert reasons == []


def test_guard_scale_fidelity_does_not_treat_a_reporting_verb_swap_as_a_hedge_shift():
    """The HSS constructed set's paraphrase items swap exactly these words (found/observed,
    showed/demonstrated) and are verified by construction; reporting verbs carry no
    epistemic force of their own (design section 3)."""
    from app.services.fulltext import _guard_scale_fidelity

    status, reasons = _guard_scale_fidelity(
        "verified",
        claim_text="The authors observed that recall improved after training.",
        evidence_quote="The authors found that recall improved after training.",
    )
    assert status == "verified"
    assert reasons == []


def test_guard_scale_fidelity_does_not_treat_a_number_word_as_a_scale_word():
    from app.services.fulltext import _guard_scale_fidelity

    status, reasons = _guard_scale_fidelity(
        "verified",
        claim_text="Three participants withdrew before the second session.",
        evidence_quote="Two participants withdrew before the second session.",
    )
    assert status == "verified"
    assert reasons == []


# --------------------------------------------------------------------------------------
# The admissibility thresholds (design section 4): head overlap at least one, context
# overlap at least two -- both needed, since head alone or context alone each mispair.
# --------------------------------------------------------------------------------------


def test_guard_scale_fidelity_does_not_align_across_an_unrelated_predicate():
    """Row 33 shape (design section 4): head alone paired the claim's "no division of
    labour" with a quote's "all six participants" on a single shared stem; the context
    condition (two shared stems within eight tokens) rejects it, so the guard passes."""
    from app.services.fulltext import _guard_scale_fidelity

    status, reasons = _guard_scale_fidelity(
        "verified",
        claim_text="The team reported no division of labour during the project.",
        evidence_quote="All six participants took part in the interview about the project.",
    )
    assert status == "verified"
    assert reasons == []


def test_guard_scale_fidelity_passes_an_unaligned_claim_scale_word():
    """Design section 4, "Unaligned occurrences are ignored": a claim scale word with no
    admissible counterpart in the quotes is not evidence of anything."""
    from app.services.fulltext import _guard_scale_fidelity

    status, reasons = _guard_scale_fidelity(
        "verified",
        claim_text="Recall improved dramatically according to the authors.",
        evidence_quote="The system produced a working prototype for the pilot study.",
    )
    assert status == "verified"
    assert reasons == []


def test_scale_admissible_pairs_requires_both_head_and_context_overlap():
    from app.services.fulltext import _scale_admissible_pairs, _scale_occurrences

    claim_occurrences = _scale_occurrences("The team rarely finished the assessment quickly.")
    quote_occurrences = _scale_occurrences(
        "The rival cohort never finished the assessment quickly."
    )
    pairs = _scale_admissible_pairs(claim_occurrences, quote_occurrences)
    assert len(pairs) == 1
    head_overlap, context_overlap, claim_index, quote_index = pairs[0]
    assert head_overlap >= 1
    assert context_overlap >= 2
    assert claim_index == 0 and quote_index == 0


# --------------------------------------------------------------------------------------
# Rule B, negation scope (design section 5, point 4; row 26): negation raising, the mirror
# image of Rule A's negated-absolute case.
# --------------------------------------------------------------------------------------


def test_guard_scale_fidelity_demotes_the_negation_scope_shift():
    from app.services.fulltext import _guard_scale_fidelity

    status, reasons = _guard_scale_fidelity(
        "verified",
        claim_text="The study established no causal role for self-reflection in learning.",
        evidence_quote=(
            "This study has not established a causal role for self-reflection in learning."
        ),
    )
    assert status == "needs_nuance"
    assert reasons == ["negation_scope_shift"]


def test_guard_scale_fidelity_does_not_fire_when_neither_side_is_negated():
    from app.services.fulltext import _guard_scale_fidelity

    status, reasons = _guard_scale_fidelity(
        "verified",
        claim_text="The study established a causal role for self-reflection in learning.",
        evidence_quote=(
            "This study has established a causal role for self-reflection in learning."
        ),
    )
    assert status == "verified"
    assert reasons == []


def test_guard_scale_fidelity_reasons_carry_both_slugs_when_both_rules_fire():
    """Design section 5, point 5: `reasons` is one slug, the other, or both in that order."""
    from app.services.fulltext import _guard_scale_fidelity

    claim = (
        "Participants seldom critically engaged with local comments, and the study "
        "established no causal role for self-reflection in learning."
    )
    quote = (
        "None reported that they critically engaged with the comments. This study has "
        "not established a causal role for self-reflection in learning."
    )
    status, reasons = _guard_scale_fidelity("verified", claim_text=claim, evidence_quote=quote)
    assert status == "needs_nuance"
    assert reasons == ["scale_word_mismatch", "negation_scope_shift"]


# --------------------------------------------------------------------------------------
# Multi-span quotes (design section 5, point 2): the whole quote pool is searched, exactly
# as `evidence_quotes or [evidence_quote]` already behaves for guard 3.
# --------------------------------------------------------------------------------------


def test_guard_scale_fidelity_searches_every_quote_span_in_the_pool():
    from app.services.fulltext import _guard_scale_fidelity

    status, reasons = _guard_scale_fidelity(
        "verified",
        claim_text="Participants seldom critically engaged with local comments.",
        evidence_quote=None,
        evidence_quotes=[
            "The interface displayed inline comments beside each paragraph of text.",
            "None reported that they critically engaged with the comments.",
        ],
    )
    assert status == "needs_nuance"
    assert reasons == ["scale_word_mismatch"]


# --------------------------------------------------------------------------------------
# Composition with the rest of `apply_verification_guards` (design section 6): placed
# immediately after guard 3, gated on `verified`, monotonic, never touches chunk texts.
# --------------------------------------------------------------------------------------


def test_guard_scale_fidelity_runs_inside_apply_verification_guards():
    from app.services.fulltext import apply_verification_guards

    claim = "Participants seldom critically engaged with local comments."
    quote = "None reported that they critically engaged with the comments."
    status, reasons, diagnostics = apply_verification_guards(
        "verified",
        claim_text=claim,
        evidence_quote=quote,
        evidence_quotes=None,
        assertions=[],
        chunk_texts=[quote],
        chunks=[{"text": quote}],
    )
    assert status == "needs_nuance"
    assert reasons == ["scale_word_mismatch"]
    assert diagnostics == []


def test_guard_scale_fidelity_never_runs_after_quote_fidelity_has_already_demoted():
    """Design section 6: guard 7's gate on `verified` means that once guard 3 has fired
    (quote not verbatim), guard 7 never runs, so it can never add a reason to a row guard 3
    has already decided -- and the repair-turn trigger's exact list equality against
    `["quote_not_verbatim"]` stays byte for byte unaffected by guard 7's presence."""
    from app.services.fulltext import apply_verification_guards

    claim = "Participants seldom critically engaged with local comments."
    # The chunk carries the claim's own terms (so guard 1, attribution, does not fire) but
    # not the fabricated quote below (so guard 3, quote fidelity, fires first).
    chunk = (
        "In this study, participants seldom critically engaged with local comments, "
        "according to the interviews."
    )
    fabricated_quote = (
        "A quote that never appears anywhere in the source text at all, invented whole "
        "for this fixture."
    )
    status, reasons, diagnostics = apply_verification_guards(
        "verified",
        claim_text=claim,
        evidence_quote=fabricated_quote,
        evidence_quotes=None,
        assertions=[],
        chunk_texts=[chunk],
        chunks=[{"text": chunk}],
    )
    assert status == "needs_nuance"
    assert reasons == ["quote_not_verbatim"]


# --------------------------------------------------------------------------------------
# GUARD_DIGEST (design section 10): moves with this guard's own function list and every
# lexicon/threshold added by value; the other frozen versions do not move.
# --------------------------------------------------------------------------------------


def test_guard_digest_is_a_16_character_hex_digest_that_moved_off_the_old_six_guard_value():
    from app.services.fulltext import GUARD_DIGEST

    assert GUARD_DIGEST is not None
    assert len(GUARD_DIGEST) == 16
    int(GUARD_DIGEST, 16)  # raises ValueError if not hex
    assert GUARD_DIGEST != "570a5b663e6140da"


def test_the_other_four_frozen_versions_do_not_move():
    from app.services.fulltext import (
        QUOTE_RELOCATION_VERSION,
        VERIFICATION_POLICY_VERSION,
    )
    from app.agents.claim_verification_agent import (
        CLAIM_VERIFICATION_PROMPT_VERSION,
        QUOTE_REPAIR_PROMPT_VERSION,
    )

    assert CLAIM_VERIFICATION_PROMPT_VERSION == "sha256:49fcfbfaf2f6"
    assert QUOTE_REPAIR_PROMPT_VERSION == "sha256:7ec7cffe852d"
    assert QUOTE_RELOCATION_VERSION == "0269a13351c0da7d"
    assert VERIFICATION_POLICY_VERSION == "667bcc9209126ddc"

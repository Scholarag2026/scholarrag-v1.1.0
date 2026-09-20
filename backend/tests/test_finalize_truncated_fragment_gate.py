"""A run can deliver one sentence fragment in extra section 5, "for whether ME's
disadvantage persists beyond four weeks or generalises to other proficiency levels." --
the second half of a sentence the writer opened with its own literal ``[NEEDS
CITATION]`` marker, continuing a previous sentence grammatically rather than starting a
new one. The citation-link retry never named it; the kept-unclassified fallback kept it
(deliberately, so a transient retry failure never destroys good text); the marker was
then stripped; what survived was a lower-case fragment. A second, otherwise identical
fragment in the same section's next paragraph never reached the delivered draft only
because that paragraph carried nothing else and was dropped whole for lacking any body.

The fix belongs at the point `_finalize_paragraph_text` is about to keep a sentence with
no citation of its own, regardless of which tag put it there (``"unclassified": True``,
a later heal's ``"framing"`` re-tag, or no entry at all): `_fragment_truncation_reason`
mirrors `demo/check_delivered.py` rule 6's own two structural signals, and a sentence it
flags is merged into the preceding sentence as its own continuation clause, or dropped
outright when there is no preceding sentence in the paragraph to merge into.

These tests are built directly from the real, saved artefacts of
``demo/output/20260915-151120/`` (read-only; not modified by this file) -- the exact
``content``, ``citation_links`` and ``uncited_sentences`` a real run produced
for extra sections 2 and 5, before this fix existed -- following the same
"re-finalize the real writing_result.json" pattern
`test_citation_link_coverage.py::test_resolve_and_finalize_keep_orphaned_sentences_from_writing_result_on_retry_failure`
already uses for a different coverage gap. Section 5's own two "unclassified" entries are
the two truncated fragments this fix repairs; section 2's own two "unclassified"
entries are complete, capitalised sentences that must survive completely unchanged, so
the fix does not overreach onto ordinary uncited findings.
"""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

import pytest  # noqa: E402

from app.services.fulltext import (  # noqa: E402
    _coverage_incomplete_reason,
    _finalize_paragraph_text,
    _fragment_truncation_reason,
    _merge_continuation_fragment,
    _split_on_sentence_boundaries,
    finalize_generated_section,
)
from tests.conftest import skip_unless_run_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "demo" / "output" / "20260915-151120"


def _load_writing_result(index: int) -> dict:
    run_dir = skip_unless_run_dir(RUN_DIR)
    return json.loads((run_dir / f"writing_result_{index}.json").read_text(encoding="utf-8"))


def _claim_status_verifying_every_link(citation_links: list[dict]) -> dict:
    """Every currently-linked sentence is already known-good: it survived into this
    same ``content``, the write job's own
    already-finalized text, so "verified" here mirrors that and re-finalizing does not
    remove text finalize already decided to keep."""
    return {
        (link["sentence"], link.get("proposition") or link["sentence"], key): "verified"
        for link in citation_links
        for key in link.get("keys") or []
    }


def _sentences_in(text: str) -> list[str]:
    """Every body sentence of *text*, split the same way `demo/check_delivered.py`'s
    own `split_sentences` does (a period/!/? followed by whitespace) -- good enough for
    these tests, which only need to ask "does any delivered sentence begin lower-case".
    """
    pieces = re.split(r"(?<=[.!?])\s+", text.replace("\n\n", " "))
    return [p.strip() for p in pieces if p.strip()]


def _no_lowercase_opening_sentence(text: str) -> None:
    for sentence in _sentences_in(text):
        first_letter = re.search(r"[A-Za-z]", sentence)
        assert not (first_letter and first_letter.group(0).islower()), (
            f"delivered a lower-case-opening fragment: {sentence!r}"
        )


# ---------------------------------------------------------------------------
# Section 5: the real defect -- both of its own two fragments repaired.
# ---------------------------------------------------------------------------


def test_section5_fragment_is_not_delivered_as_its_own_broken_sentence():
    """A marker-headed fragment (the marker sat in front of its own
    dependent clause, so stripping it left a lower-case opening) is dropped
    outright, never merged onto the sentence before it -- merging it there
    produced exactly the broken run-on this test used to assert as the fix (run
    11's own section 6 reproduced the same defect three times over: section 2)."""
    writing_result = _load_writing_result(5)
    text = writing_result["content"]
    citation_links = writing_result["citation_links"]
    uncited_sentences = writing_result["uncited_sentences"]
    assert any(
        entry.get("unclassified") and entry["sentence"].startswith("[NEEDS CITATION] for")
        for entry in uncited_sentences
    ), "fixture sanity: the real run's own marker-carrying fragment must still be present"

    claim_status = _claim_status_verifying_every_link(citation_links)

    result = finalize_generated_section(text, citation_links, uncited_sentences, claim_status)

    _no_lowercase_opening_sentence(result.text)
    assert "for whether ME's disadvantage persists beyond four weeks" not in result.text
    # The Mao et al. sentence is delivered exactly as the citation-link map
    # verified it, ending in its own period -- not turned into a comma-joined
    # run-on with the dropped fragment's own clause.
    assert (
        "research has focused predominantly on teacher WCF to the exclusion of "
        "peer and computational sources (Mao et al., 2024)."
    ) in result.text
    # The second fragment ("for how learners actually revise ...") is alone in its own
    # paragraph with nothing to merge into either, so it -- and the now-bodiless heading
    # in front of it -- are dropped entirely, exactly as the real (already-healed) draft
    # shows (`draft_content_5.json` carries no "Revision behaviour and engagement"
    # heading at all).
    assert "for how learners actually revise" not in result.text
    assert "Revision behaviour and engagement" not in result.text
    assert result.stats["sentences_truncated_fragment_repaired"] == 2
    assert result.stats["sentences_unclassified_kept"] == 0
    # Both fragments are marker-headed (a lower-case opening after their own
    # marker was stripped from the front), so both are named here, whether or not
    # either one had a preceding sentence in its own paragraph to merge into.
    assert len(result.truncated_fragment_reasons) == 2
    dropped_sentences = {entry["sentence"] for entry in result.truncated_fragment_reasons}
    assert (
        "for whether ME's disadvantage persists beyond four weeks or generalises to "
        "other proficiency levels."
    ) in dropped_sentences
    for entry in result.truncated_fragment_reasons:
        assert entry["reason"].startswith("begins with a lower-case letter")


def test_merge_continuation_fragment_joins_with_a_comma_and_keeps_one_terminal_period():
    previous = (
        "A synthesis of 50 classroom studies found that most were not theoretically "
        "motivated, with only 11 (22.0%) making explicit reference to theoretical "
        "tenets, and that research has focused predominantly on teacher WCF to the "
        "exclusion of peer and computational sources (Mao et al., 2024)."
    )
    fragment = (
        "for whether ME's disadvantage persists beyond four weeks or generalises to "
        "other proficiency levels."
    )

    merged = _merge_continuation_fragment(previous, fragment)

    assert merged == (
        "A synthesis of 50 classroom studies found that most were not theoretically "
        "motivated, with only 11 (22.0%) making explicit reference to theoretical "
        "tenets, and that research has focused predominantly on teacher WCF to the "
        "exclusion of peer and computational sources (Mao et al., 2024), for whether "
        "ME's disadvantage persists beyond four weeks or generalises to other "
        "proficiency levels."
    )
    # Exactly one real sentence boundary once merged -- the same splitter this
    # module's own claim extractor uses (`_split_on_sentence_boundaries`) no longer
    # cuts this in two, because the false boundary was a comma, not a period, all
    # along. (The text still carries two OTHER periods of its own, "22.0%" and the
    # "et al." abbreviation before a comma, neither of them a candidate split point.)
    assert len(_split_on_sentence_boundaries(merged)) == 1


def test_a_truncated_fragment_with_no_preceding_sentence_is_dropped_not_delivered():
    """Item (a)'s "otherwise removed": the second of section 5's own two fragments is
    the only sentence of its own paragraph, so there is nothing in that paragraph to
    merge it into."""
    fragment = (
        "[NEEDS CITATION] for how learners actually revise in response to ME codes, "
        "and for whether ME promotes deeper cognitive engagement despite higher "
        "perceived load."
    )
    uncited = [{"tag": "finding", "sentence": fragment, "unclassified": True}]

    rebuilt, surviving_links, stats, _healed, surviving_uncited = _finalize_paragraph_text(
        fragment, [], uncited, {}
    )

    assert rebuilt == ""
    assert surviving_links == []
    assert surviving_uncited == []
    assert stats["sentences_truncated_fragment_repaired"] == 1
    assert stats["sentences_unclassified_kept"] == 0


# ---------------------------------------------------------------------------
# Item (c): a later heal's own "framing" re-tag must not hide a fragment either --
# the gate fires on the sentence's own shape, not on whichever tag reached it.
# ---------------------------------------------------------------------------


def test_a_framing_tagged_fragment_is_still_repaired_not_delivered():
    """A verify-stage heal can re-tag the surviving fragment
    ``"framing"`` on the draft node, which is exactly why rule 1 (an uncited "finding"
    the retry positively classified) never sees it -- a framing-tagged, or altogether
    untagged, uncited sentence would otherwise be unconditionally kept by
    `_finalize_paragraph_text`. This must no longer be true for a sentence this
    predicate flags, whatever tag it carries.

    No literal ``[NEEDS CITATION]`` marker is in *fragment* here -- it stands for a
    fragment already marker-stripped by an earlier pass, before this later heal's
    own "framing" re-tag ever reached it, so this call's own ``marker_count`` for it
    is zero. The marker-headed drop still applies to it: the fragment reaches
    this branch at all only because *uncited* names it (``gate_applies``'s own
    second disjunct), which is the same origin as a sentence this pass marker-
    stripped itself, and its own text still opens lower case -- so it is dropped,
    not merged, exactly like the section 5 fixture above."""
    previous_sentence = "A synthesis of 50 studies found gains (Mao et al., 2024)."
    proposition = "A synthesis of 50 studies found gains"
    fragment = (
        "for whether ME's disadvantage persists beyond four weeks or generalises to "
        "other proficiency levels."
    )
    text = f"{previous_sentence} {fragment}"
    links = [{
        "sentence": previous_sentence,
        "proposition": proposition,
        "keys": ["mao_2024"],
        "citation_text": "(Mao et al., 2024)",
    }]
    uncited = [{"sentence": fragment, "tag": "framing"}]
    claim_status = {(previous_sentence, proposition, "mao_2024"): "verified"}

    rebuilt, _links, stats, _healed, surviving_uncited = _finalize_paragraph_text(
        text, links, uncited, claim_status
    )

    _no_lowercase_opening_sentence(rebuilt)
    # The fragment is dropped outright, not merged: the previous sentence survives
    # verbatim, ending in its own period, and the fragment's own clause never
    # reaches the delivered text at all.
    assert rebuilt == previous_sentence
    assert len(_split_on_sentence_boundaries(rebuilt)) == 1
    assert "for whether ME's disadvantage" not in rebuilt
    assert stats["sentences_truncated_fragment_repaired"] == 1
    assert stats["truncated_fragment_reasons"] == [
        {
            "sentence": fragment,
            "reason": _fragment_truncation_reason(fragment, marker_stripped=False),
        }
    ]
    assert surviving_uncited == []


# ---------------------------------------------------------------------------
# The tail case: a marker stripped from the position of a missing predicate
# complement rather than from in front of a dependent clause -- the shape the head
# case above (section 5) does not cover. English licenses a bare "remains" (and the
# rest of `_MARKER_STRIPPED_BARE_VERBS`) only with a nominal subject, not a clausal
# one, so a whether-clause subject needs a complement after the verb; the writer put
# its own marker exactly where that complement belonged, and stripping it left the
# verb standing alone.
# ---------------------------------------------------------------------------


def test_a_marker_stripped_bare_verb_ending_is_removed_and_the_reason_is_recorded():
    sentence = (
        "Whether AWCF produces durable accuracy gains, rather than momentary error "
        "flagging, remains [NEEDS CITATION]."
    )
    uncited = [{"tag": "finding", "sentence": sentence, "unclassified": True}]

    rebuilt, surviving_links, stats, _healed, surviving_uncited = _finalize_paragraph_text(
        sentence, [], uncited, {}
    )

    assert rebuilt == ""
    assert surviving_links == []
    assert surviving_uncited == []
    assert stats["sentences_truncated_fragment_repaired"] == 1
    reason = _fragment_truncation_reason(
        "Whether AWCF produces durable accuracy gains, rather than momentary error "
        "flagging, remains.",
        marker_stripped=True,
    )
    assert reason == (
        "a stripped [NEEDS CITATION] marker leaves 'remains.' as a bare verb with no "
        "complement"
    )


def test_a_second_marker_stripped_bare_verb_ending_is_also_removed():
    """A second, independently produced sentence with the same tail shape: a
    different clausal subject, the same bare "remains" with nothing after it."""
    sentence = (
        "Whether such approval translates into durable accuracy gains remains "
        "[NEEDS CITATION]."
    )
    uncited = [{"tag": "finding", "sentence": sentence, "unclassified": True}]

    rebuilt, _links, stats, _healed, surviving_uncited = _finalize_paragraph_text(
        sentence, [], uncited, {}
    )

    assert rebuilt == ""
    assert surviving_uncited == []
    assert stats["sentences_truncated_fragment_repaired"] == 1


def test_a_marker_stripped_sentence_with_a_complete_predicate_is_kept():
    """The marker sits after a complement this time ("an open question"), so
    stripping it leaves a complete sentence, not a fragment. The third signal must
    not overreach onto every marker-stripped sentence ending in one of these verbs,
    only the ones with nothing left after the verb."""
    sentence = (
        "Whether AWCF produces durable accuracy gains remains an open question "
        "[NEEDS CITATION]."
    )
    uncited = [{"tag": "finding", "sentence": sentence, "unclassified": True}]

    rebuilt, _links, stats, _healed, surviving_uncited = _finalize_paragraph_text(
        sentence, [], uncited, {}
    )

    assert rebuilt == "Whether AWCF produces durable accuracy gains remains an open question."
    assert stats["sentences_truncated_fragment_repaired"] == 0
    assert len(surviving_uncited) == 1


def test_the_bare_verb_signal_never_fires_when_no_marker_was_stripped():
    """`_fragment_truncation_reason`'s own default (``marker_stripped=False``) must
    never flag an ordinary complete sentence just because it happens to end on one
    of these verbs."""
    assert _fragment_truncation_reason("The effect remains.") is None
    assert _fragment_truncation_reason("The effect remains.", marker_stripped=False) is None


def test_an_ordinary_sentence_ending_in_one_of_these_verbs_with_no_marker_is_kept():
    """The same sentence, run through the actual gate this module applies
    (`_finalize_paragraph_text`) rather than the predicate alone: no marker, no
    uncited entry naming it, so the gate never applies and the sentence is delivered
    unchanged."""
    sentence = "The effect remains."

    rebuilt, _links, stats, _healed, _uncited = _finalize_paragraph_text(sentence, [], [], {})

    assert rebuilt == sentence
    assert stats["sentences_truncated_fragment_repaired"] == 0


# ---------------------------------------------------------------------------
# Section 2: non-regression -- two complete, capitalised "unclassified" sentences
# must survive completely unchanged.
# ---------------------------------------------------------------------------


def test_section2_complete_unclassified_sentences_are_unaffected_by_the_gate():
    writing_result = _load_writing_result(2)
    text = writing_result["content"]
    citation_links = writing_result["citation_links"]
    uncited_sentences = writing_result["uncited_sentences"]
    unclassified = [e for e in uncited_sentences if e.get("unclassified")]
    assert len(unclassified) == 2, "fixture sanity: section 2's own two kept sentences"
    for entry in unclassified:
        assert entry["sentence"][0].isupper(), "fixture sanity: these are whole sentences"

    claim_status = _claim_status_verifying_every_link(citation_links)

    result = finalize_generated_section(text, citation_links, uncited_sentences, claim_status)

    for entry in unclassified:
        assert entry["sentence"] in result.text
    assert result.stats["sentences_truncated_fragment_repaired"] == 0
    assert result.stats["sentences_unclassified_kept"] == 2


# ---------------------------------------------------------------------------
# Zero-drift fixture: the backend's own predicate and `demo/check_delivered.py` rule
# 6's predicate must agree, sentence for sentence, or this test fails.
# ---------------------------------------------------------------------------


def _load_check_delivered():
    spec = importlib.util.spec_from_file_location(
        "check_delivered_zero_drift", REPO_ROOT / "demo" / "check_delivered.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


DRIFT_FIXTURE_SENTENCES = [
    "for whether ME's disadvantage persists beyond four weeks or generalises to other "
    "proficiency levels.",
    "for how learners actually revise in response to ME codes, and for whether ME "
    "promotes deeper cognitive engagement despite higher perceived load.",
    "Smith (2020) reported large gains in accuracy.",
    "Their results showed that although direct corrections and codes were both "
    "effective, a long-term advantage was evident only for direct corrections.",
    "Precision remained high across prompts, exceeding Grammarly (Luo et al., 2025).",
    "Luo et al. (2025) found ChatGPT's precision exceeded Grammarly's (94-98% vs. 85%).",
    "",
    "   ",
]


def test_backend_truncation_predicate_matches_check_delivered_rule6_zero_drift():
    check_delivered = _load_check_delivered()
    for sentence in DRIFT_FIXTURE_SENTENCES:
        backend_reason = _fragment_truncation_reason(sentence)
        checker_reasons = check_delivered.truncation_reasons(sentence, check_lowercase_start=True)
        checker_reason = checker_reasons[0] if checker_reasons else None
        assert backend_reason == checker_reason, (
            f"drift on {sentence!r}: backend={backend_reason!r} checker={checker_reason!r}"
        )


# ---------------------------------------------------------------------------
# Item (d): the write stage's own record of why a coverage-incomplete removal fired.
# ---------------------------------------------------------------------------


def test_coverage_incomplete_reason_names_the_residue():
    reason = _coverage_incomplete_reason(
        "In a small ethnographic case study of four L2 PhD students, engagement with "
        "feedback varied widely (Lopez, 2019).",
        bad_links=[],
        non_frame_residues=["In a small ethnographic case study of four L2 PhD students"],
        has_meta_evaluation=False,
    )
    assert reason["reasons"] == [
        "unresolved residue 'In a small ethnographic case study of four L2 PhD students'"
    ]


def test_coverage_incomplete_reason_names_an_unverified_proposition():
    reason = _coverage_incomplete_reason(
        "A claim with one bad link (Smith, 2020).",
        bad_links=[{"proposition": "a bad proposition", "keys": ["smith_2020"]}],
        non_frame_residues=[],
        has_meta_evaluation=False,
    )
    assert reason["reasons"] == [
        "proposition 'a bad proposition' against smith_2020 carried no verified verdict"
    ]


def test_finalize_paragraph_text_records_a_coverage_incomplete_cause_for_a_leading_residue():
    """A leading (not trailing) residue has no safe deterministic cut
    (`sentence_coverage.trailing_excision_cut` returns ``None``), so the whole sentence
    is removed, and this pass must say why."""
    sentence = (
        "In a small ethnographic case study of four L2 PhD students, engagement "
        "with feedback varied widely (Lopez, 2019)."
    )
    links = [{
        "sentence": sentence,
        "proposition": "engagement with feedback varied widely",
        "keys": ["lopez_2019"],
        "citation_text": "(Lopez, 2019)",
    }]
    claim_status = {(sentence, "engagement with feedback varied widely", "lopez_2019"): "verified"}

    _rebuilt, _links, stats, _healed, _uncited = _finalize_paragraph_text(
        sentence, links, [], claim_status
    )

    assert stats["sentences_removed_coverage_incomplete"] == 1
    assert len(stats["coverage_incomplete_reasons"]) == 1
    reason = stats["coverage_incomplete_reasons"][0]
    assert reason["sentence"] == sentence
    assert any("residue" in r for r in reason["reasons"])


def test_finalize_generated_section_carries_coverage_incomplete_reasons_and_pops_them_from_stats():
    sentence = (
        "In a small ethnographic case study of four L2 PhD students, engagement "
        "with feedback varied widely (Lopez, 2019)."
    )
    links = [{
        "sentence": sentence,
        "proposition": "engagement with feedback varied widely",
        "keys": ["lopez_2019"],
        "citation_text": "(Lopez, 2019)",
    }]
    claim_status = {(sentence, "engagement with feedback varied widely", "lopez_2019"): "verified"}

    result = finalize_generated_section(sentence, links, [], claim_status)

    assert len(result.coverage_incomplete_reasons) == 1
    assert result.coverage_incomplete_reasons[0]["sentence"] == sentence
    # Never miscounted as though it were a number, and never left inside the plain,
    # int-only stats dict every existing reader of `.stats`/`finalize_stats` expects.
    assert "coverage_incomplete_reasons" not in result.stats
    assert result.stats["sentences_removed_coverage_incomplete"] == 1


def test_finalize_paragraph_text_with_no_coverage_incomplete_removal_reports_an_empty_list():
    text = "Smith (2020) reported large gains in accuracy."
    links = [{"sentence": text, "proposition": text, "keys": ["smith_2020"]}]
    claim_status = {(text, text, "smith_2020"): "verified"}

    _rebuilt, _links, stats, _healed, _uncited = _finalize_paragraph_text(
        text, links, [], claim_status
    )

    assert stats["coverage_incomplete_reasons"] == []

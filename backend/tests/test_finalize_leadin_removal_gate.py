"""A coverage-incomplete sentence whose every proposition already verified,
blocked from a permitted frame only by its own leading attribution carrying a
numeral ("a 16-week case study", "seventy ESL students") or a noun outside
`app.services.sentence_coverage.ATTRIB` ("a case"), is removed whole by the
ordinary coverage-incomplete path -- never rewritten. An earlier fix
(`is_attribution_leadin` / `trim_attribution_leadin`) trimmed such a lead-in to
a deterministic skeleton instead of discarding the finding; that mechanism has
since been removed, because a rewrite that manufactures a subject, or keeps one
the lead-in itself did not carry, is not part of this module's own guarantee
(deliver only what is verified, or omit the sentence). A lead-in this narrow
cannot in general be told apart from a genuine editorial generalisation or
inference asserted on its own account, so no lead-in is rewritten here,
regardless of shape.

Two real, previously delivered sentences confirm the current behaviour:

- Run 4 (`demo/output/20260915-211700/writing_result.json`): "Teng and Ma
  (2024) similarly relied on self-report data, cautioning that their scale may
  not fully reflect learners' feedback literacy in academic writing." was
  trimmed, under the old mechanism, to a bare-participle sentence with no
  finite verb ("Teng and Ma (2024) cautioning that ..."). It is now removed.
- Run 5 (`demo/output/20260915-234219/writing_result_5.json`): "A separate
  case comparison showed that one student resubmitted her essay 13 times while
  another made one resubmission (Zhang & Hyland, 2018)." was trimmed, under
  the old mechanism, to a sentence with no subject at all ("Showed that one
  student resubmitted ..."). It is now removed at the same point; a second,
  independent guard (`sentence_coverage.opens_with_reporting_verb`,
  `sentences_removed_verb_initial`) also removes that already-subjectless text
  directly, should it ever reach finalize by any other path.
"""

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

import pytest  # noqa: E402

from app.services import sentence_coverage  # noqa: E402
from app.services.fulltext import _finalize_paragraph_text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Four sentences whose own leading attribution carries a numeral or an
# out-of-lexicon noun, reconstructed from a real run's own saved
# `writing_result*.json` (read-only): every proposition on each sentence was
# verified, none in `bad_links`.
# ---------------------------------------------------------------------------

CASE_DEMONSTRATED_NUMERAL_LEADIN = {
    "sentence": (
        "Zhang and Hyland (2018) demonstrated through a 16-week case study that "
        "engagement mediates outcomes: the highly engaged learner resubmitted 13 "
        "times versus one resubmission by the moderately engaged peer."
    ),
    "proposition": (
        "engagement mediates outcomes: the highly engaged learner resubmitted 13 "
        "times versus one resubmission by the moderately engaged peer"
    ),
    "citation_text": "Zhang and Hyland (2018)",
    "key": "zhang_2018",
}

CASE_SURVEYED_NUMERAL_LEADIN = {
    "sentence": (
        "Liu and Wu (2019) surveyed seventy ESL students and found that students "
        "favored direct correction in conjunction with metalinguistic explanations."
    ),
    "proposition": (
        "students favored direct correction in conjunction with metalinguistic "
        "explanations"
    ),
    "citation_text": "Liu and Wu (2019)",
    "key": "liu_2019",
}

CASE_DOCUMENTED_OUT_OF_LEXICON_NOUN_LEADIN = {
    "sentence": (
        "Zhang and Hyland (2018) documented a case in which teacher feedback "
        "covered more error types than automated writing evaluation, and "
        "concluded that integrating the two sources may be more effective than "
        "either alone."
    ),
    "proposition": (
        "teacher feedback covered more error types than automated writing "
        "evaluation, and concluded that integrating the two sources may be more "
        "effective than either alone"
    ),
    "citation_text": "Zhang and Hyland (2018)",
    "key": "zhang_2018",
}

CASE_STUDY_OF_NUMERAL_LEADIN = {
    "sentence": (
        "In a study of six graduate teaching associates using Grammarly to "
        "complement their feedback, teachers provided feedback on both global "
        "and local aspects of writing, with no division of labour between "
        "higher-order and lower-order concerns (Koltovskaia, 2022)."
    ),
    "proposition": (
        "teachers provided feedback on both global and local aspects of "
        "writing, with no division of labour between higher-order and "
        "lower-order concerns"
    ),
    "citation_text": "(Koltovskaia, 2022)",
    "key": "koltovskaia_2022",
}

# Run 4's own delivered sentence (`demo/output/20260915-211700/writing_result.json`
# `loop_stats.leadin_trim_reasons`): its residue carries only an `-ing`
# (present-participle) reporting-verb token, "cautioning", ahead of an
# incidental lexicon match inside the compound noun "self-report".
CASE_RUN4_TENG_MA = {
    "sentence": (
        "Teng and Ma (2024) similarly relied on self-report data, cautioning that "
        "their scale may not fully reflect learners' feedback literacy in academic "
        "writing."
    ),
    "proposition": (
        "their scale may not fully reflect learners' feedback literacy in academic "
        "writing"
    ),
    "citation_text": "Teng and Ma (2024)",
    "key": "teng_2024",
}

# Run 5's own delivered sentence (`demo/output/20260915-234219/writing_result_5.json`
# `loop_stats.leadin_trim_reasons`): its own citation trails the sentence instead of
# sitting in front of the residue's finite reporting verb.
CASE_RUN5_ZHANG_HYLAND = {
    "sentence": (
        "A separate case comparison showed that one student resubmitted her "
        "essay 13 times while another made one resubmission (Zhang & Hyland, "
        "2018)."
    ),
    "proposition": (
        "one student resubmitted her essay 13 times while another made one "
        "resubmission"
    ),
    "citation_text": "(Zhang & Hyland, 2018)",
    "key": "zhang_2018",
}

ALL_CASES = [
    CASE_DEMONSTRATED_NUMERAL_LEADIN,
    CASE_SURVEYED_NUMERAL_LEADIN,
    CASE_DOCUMENTED_OUT_OF_LEXICON_NOUN_LEADIN,
    CASE_STUDY_OF_NUMERAL_LEADIN,
    CASE_RUN4_TENG_MA,
    CASE_RUN5_ZHANG_HYLAND,
]


def _links_and_status(case: dict) -> tuple[list[dict], dict]:
    sentence = case["sentence"]
    proposition = case["proposition"]
    key = case["key"]
    links = [{
        "sentence": sentence,
        "proposition": proposition,
        "keys": [key],
        "citation_text": case["citation_text"],
    }]
    claim_status = {(sentence, proposition, key): "verified"}
    return links, claim_status


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c["citation_text"])
def test_attribution_leadin_blocked_sentence_is_removed_not_rewritten(case):
    """A fully verified finding blocked only by its own leading attribution is
    removed whole -- the pipeline never manufactures or relocates a subject to
    rescue it."""
    links, claim_status = _links_and_status(case)

    rebuilt, kept_links, stats, _healed, _uncited = _finalize_paragraph_text(
        case["sentence"], links, [], claim_status
    )

    assert rebuilt == ""
    assert kept_links == []
    assert stats["sentences_removed_coverage_incomplete"] == 1
    assert stats["sentences_removed_verb_initial"] == 0
    assert "sentences_leadin_trimmed" not in stats
    assert len(stats["coverage_incomplete_reasons"]) == 1


def test_run5_delivered_subjectless_sentence_is_removed_by_the_verb_initial_guard():
    """Even when a sentence somehow reaches finalize already in the broken,
    subject-less shape the old trim used to produce -- "Showed that one student
    resubmitted her essay 13 times while another made one resubmission (Zhang &
    Hyland, 2018)." -- the residue "Showed that" carries zero content tokens, so
    it passes the coverage decision as a permitted frame with no regard for
    whether a subject precedes the verb. The final, unconditional guard
    (`sentence_coverage.opens_with_reporting_verb`) still removes it, with the
    reason recorded."""
    sentence = (
        "Showed that one student resubmitted her essay 13 times while another "
        "made one resubmission (Zhang & Hyland, 2018)."
    )
    proposition = (
        "one student resubmitted her essay 13 times while another made one "
        "resubmission"
    )
    citation_text = "(Zhang & Hyland, 2018)"
    key = "zhang_2018"
    links = [{
        "sentence": sentence,
        "proposition": proposition,
        "keys": [key],
        "citation_text": citation_text,
    }]
    claim_status = {(sentence, proposition, key): "verified"}

    rebuilt, kept_links, stats, _healed, _uncited = _finalize_paragraph_text(
        sentence, links, [], claim_status
    )

    assert rebuilt == ""
    assert kept_links == []
    assert stats["sentences_removed_coverage_incomplete"] == 0
    assert stats["sentences_removed_verb_initial"] == 1
    assert len(stats["verb_initial_reasons"]) == 1
    reason = stats["verb_initial_reasons"][0]
    assert reason["sentence"] == sentence
    assert reason["delivered"] == sentence


# ---------------------------------------------------------------------------
# Control: a numeral inside the PROPOSITION itself (an unverified one, not the
# lead-in) still results in the whole sentence being removed.
# ---------------------------------------------------------------------------


def test_a_numeral_in_an_unverified_proposition_not_the_leadin_is_still_removed():
    sentence = (
        "Ferris (2006) reported that 24 students improved, and Lee (2020) found "
        "that overall accuracy increased significantly."
    )
    bad_proposition = "24 students improved"
    good_proposition = "overall accuracy increased significantly"
    links = [
        {
            "sentence": sentence,
            "proposition": bad_proposition,
            "keys": ["ferris_2006"],
            "citation_text": "Ferris (2006)",
        },
        {
            "sentence": sentence,
            "proposition": good_proposition,
            "keys": ["lee_2020"],
            "citation_text": "Lee (2020)",
        },
    ]
    claim_status = {
        (sentence, bad_proposition, "ferris_2006"): "unsupported",
        (sentence, good_proposition, "lee_2020"): "verified",
    }

    rebuilt, _kept_links, stats, _healed, _uncited = _finalize_paragraph_text(
        sentence, links, [], claim_status
    )

    assert rebuilt == ""
    assert stats["sentences_removed_coverage_incomplete"] == 1
    assert len(stats["coverage_incomplete_reasons"]) == 1
    reason = stats["coverage_incomplete_reasons"][0]
    assert any("no verified verdict" in r for r in reason["reasons"])


# ---------------------------------------------------------------------------
# Non-regression: a genuine, unverified editorial inference blocked by the same
# residue shapes (a numeral, an out-of-lexicon noun) must still be removed.
# ---------------------------------------------------------------------------


def test_a_genuine_unverified_inference_leadin_is_not_rescued():
    """A genuine unverified inference, not an attribution, correctly removed
    whole even though its own leading residue also carries a numeral."""
    sentence = (
        "Their design, random assignment across conditions with a delayed "
        "posttest, supports the inference that feedback form, not learner "
        "proficiency, drove the differences, since baseline proficiency did not "
        "differ across groups (Bonilla Lopez et al., 2018)."
    )
    proposition = "baseline proficiency did not differ across groups"
    links = [{
        "sentence": sentence,
        "proposition": proposition,
        "keys": ["lopez_2018"],
        "citation_text": "(Bonilla Lopez et al., 2018)",
    }]
    claim_status = {(sentence, proposition, "lopez_2018"): "verified"}

    rebuilt, _kept_links, stats, _healed, _uncited = _finalize_paragraph_text(
        sentence, links, [], claim_status
    )

    assert rebuilt == ""
    assert stats["sentences_removed_coverage_incomplete"] == 1


def test_an_editorial_generalisation_leadin_is_not_rescued():
    """An editorial generalisation ("engagement is uneven") plus a method
    detail, not a bare attribution, correctly removed whole even though it also
    carries a numeral ("two", "16") and the noun "case"."""
    sentence = (
        "Yet engagement is uneven: in an earlier naturalistic case study of two "
        "students over 16 weeks, Zhang and Hyland (2018) reported that the "
        "highly engaged learner resubmitted her essay 13 times while the "
        "moderately engaged learner made one resubmission."
    )
    proposition = (
        "the highly engaged learner resubmitted her essay 13 times while the "
        "moderately engaged learner made one resubmission"
    )
    links = [{
        "sentence": sentence,
        "proposition": proposition,
        "keys": ["zhang_2018"],
        "citation_text": "Zhang and Hyland (2018)",
    }]
    claim_status = {(sentence, proposition, "zhang_2018"): "verified"}

    rebuilt, _kept_links, stats, _healed, _uncited = _finalize_paragraph_text(
        sentence, links, [], claim_status
    )

    assert rebuilt == ""
    assert stats["sentences_removed_coverage_incomplete"] == 1


# ---------------------------------------------------------------------------
# `demo/check_delivered.py` rule 12's own shared predicate
# (`opens_with_participle_after_attribution`/`_opens_with_participle_after_
# attribution`) must agree with `app.services.sentence_coverage`'s copy, sentence
# for sentence, or this test fails -- the same zero-drift pattern
# `test_finalize_uncited_meta_evaluation_gate.py` uses for rule 11. Independent of
# the lead-in removal above: rule 12 is a standalone delivered-output check, never
# called from `_finalize_paragraph_text` itself.
# ---------------------------------------------------------------------------


def _load_check_delivered():
    spec = importlib.util.spec_from_file_location(
        "check_delivered_zero_drift_leadin", REPO_ROOT / "demo" / "check_delivered.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


PARTICIPLE_AFTER_ATTRIBUTION_ZERO_DRIFT_FIXTURE_SENTENCES = [
    # A participle immediately after a citation, with no finite reporting verb (positive).
    "Teng and Ma (2024) cautioning that their scale may not fully reflect learners' "
    "feedback literacy in academic writing.",
    # One `-ing` inflection per citation shape: narrative and bracketed (positive).
    "Zhang and Hyland (2018) warning that engagement effects may not generalise.",
    "(Liu and Wu, 2019) conceding that their design was not randomised.",
    # A finite verb, not a participle, immediately follows the citation (negative).
    "Teng and Ma (2024) report that their scale may not fully reflect learners' "
    "feedback literacy in academic writing.",
    # A finite verb governing the participle as its own object -- the participle no
    # longer directly abuts the citation (negative).
    "Teng and Ma (2024) reported cautioning that their scale may not fully reflect "
    "learners' feedback literacy in academic writing.",
    # An ordinary, grammatical cited sentence (negative).
    "Li and Hébert (2023) found that peer feedback produced meaningful improvements "
    "in scores between drafts.",
    # A discourse connective precedes the citation, so the sentence does not OPEN
    # with it (negative).
    "However, Zhang and Hyland (2018) argued that engagement mediates outcomes.",
    "",
]


def test_backend_participle_predicate_matches_check_delivered_rule12_zero_drift():
    check_delivered = _load_check_delivered()
    for sentence in PARTICIPLE_AFTER_ATTRIBUTION_ZERO_DRIFT_FIXTURE_SENTENCES:
        backend_flag = sentence_coverage.opens_with_participle_after_attribution(sentence)
        checker_flag = check_delivered._opens_with_participle_after_attribution(sentence)
        assert backend_flag == checker_flag, (
            f"drift on {sentence!r}: backend={backend_flag} checker={checker_flag}"
        )


def test_check_delivered_rule12_flags_the_real_run_sentence_only():
    check_delivered = _load_check_delivered()
    flags = [
        check_delivered._opens_with_participle_after_attribution(s)
        for s in PARTICIPLE_AFTER_ATTRIBUTION_ZERO_DRIFT_FIXTURE_SENTENCES
    ]
    assert flags == [True, True, True, False, False, False, False, False]


# ---------------------------------------------------------------------------
# Rule 13 (`verb_initial_sentence`): a delivered sentence whose own first two
# tokens are a past-tense or third-person reporting verb followed by "that",
# with no subject before it at all. `demo/check_delivered.py`'s own shared
# predicate (`_opens_with_reporting_verb`) must agree with
# `app.services.sentence_coverage`'s copy (`opens_with_reporting_verb`), sentence
# for sentence, the same zero-drift pattern used for rule 12 above.
# ---------------------------------------------------------------------------

VERB_INITIAL_SENTENCE_ZERO_DRIFT_FIXTURE_SENTENCES = [
    # The real defect (positive).
    "Showed that one student resubmitted her essay 13 times while another made "
    "one resubmission (Zhang & Hyland, 2018).",
    # Any past/third-person reporting-verb inflection followed by "that" opening a
    # sentence is a violation too (positive), case-insensitive.
    "showed that engagement mediates outcomes (Zhang & Hyland, 2018).",
    "SHOWED that engagement mediates outcomes (Zhang & Hyland, 2018).",
    "Cautioned that automated programmes failed to identify many important "
    "errors (Hyland, 2025).",
    # An ordinary, grammatical cited sentence opening with its own citation
    # subject, not a bare verb (negative).
    "Zhang and Hyland (2018) showed that one student resubmitted her essay 13 "
    "times while another made one resubmission.",
    # This participle shape opens with a citation subject, not a bare verb,
    # which is rule 12's own business, not rule 13's (negative).
    "Teng and Ma (2024) cautioning that their scale may not fully reflect "
    "learners' feedback literacy in academic writing.",
    # A finite reporting verb, correctly attributed, is not a violation (negative).
    "Teng and Ma (2024) report that their scale may not fully reflect "
    "learners' feedback literacy in academic writing.",
    # A discourse connective precedes the verb, so the sentence does not open
    # with it (negative).
    "However, showed that engagement mediates outcomes (Zhang & Hyland, 2018).",
    # An ordinary sentence whose subject is a plural noun that merely resembles a
    # reporting-verb inflection, not the verb itself, followed by an unrelated
    # word rather than "that" (negative) -- the exact false-positive shape a
    # first-word-only check would miss.
    "Reports from teachers suggest direct correction improves accuracy.",
    # The base (dictionary) form immediately followed by "that" is not a shape
    # any part of the pipeline produces (negative).
    "Show that direct correction improved accuracy.",
    "",
]


def test_backend_reporting_verb_predicate_matches_check_delivered_rule13_zero_drift():
    check_delivered = _load_check_delivered()
    for sentence in VERB_INITIAL_SENTENCE_ZERO_DRIFT_FIXTURE_SENTENCES:
        backend_flag = sentence_coverage.opens_with_reporting_verb(sentence)
        checker_flag = check_delivered._opens_with_reporting_verb(sentence)
        assert backend_flag == checker_flag, (
            f"drift on {sentence!r}: backend={backend_flag} checker={checker_flag}"
        )


def test_check_delivered_rule13_flags_the_expected_sentences_only():
    check_delivered = _load_check_delivered()
    flags = [
        check_delivered._opens_with_reporting_verb(s)
        for s in VERB_INITIAL_SENTENCE_ZERO_DRIFT_FIXTURE_SENTENCES
    ]
    assert flags == [
        True, True, True, True, False, False, False, False, False, False, False,
    ]

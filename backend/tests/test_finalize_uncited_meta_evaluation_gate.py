"""An uncited framing sentence can carry a comparative meta-evaluation of the
literature -- for example "The strongest comparative evidence concerns feedback
explicitness." or "The most direct evidence on coded feedback comes from
experimental comparisons of comprehensive feedback forms." -- which
`demo/check_delivered.py` rule 11 (`comparative_meta_evaluation`) flags.

`_finalize_paragraph_text`'s own write-time gate runs
`app.services.sentence_coverage.has_meta_evaluation` over `kept_propositions` -- the
CITED-sentence path. An uncited sentence (tagged "framing", "unclassified", or
carrying no entry at all) has no proposition of its own, so without checking it too, a
comparative meta-evaluation written as a paragraph-opening framing sentence would pass
the write-time gate untouched and reach the delivered draft, only for
`check_delivered.py`'s own, independently written `_META_EVALUATION_RE` to catch it
downstream, on the delivered text.

The gate applies the SAME predicate (`has_meta_evaluation`) to every uncited sentence
about to be kept, whatever tag it carries, and drops any that fires
(`app.services.fulltext._finalize_paragraph_text`, counter
`sentences_removed_meta_evaluation_uncited`).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

import pytest  # noqa: E402

from app.services import sentence_coverage  # noqa: E402
from app.services.fulltext import (  # noqa: E402
    _finalize_paragraph_text,
    finalize_generated_section,
)
from tests.conftest import skip_unless_run_dir  # noqa: E402
from tests.test_sentence_coverage import (  # noqa: E402
    ORDINARY_COMPARATIVE_SENTENCES_FROM_DELIVERED_DRAFTS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "demo" / "output" / "20260915-182437"

VIOLATING_SENTENCES = [
    "The strongest comparative evidence concerns feedback explicitness.",
    "The most direct evidence on coded feedback comes from experimental comparisons "
    "of comprehensive feedback forms.",
]

INNOCENT_FRAMING_SENTENCE = "Learner preferences point in a different direction."


# ---------------------------------------------------------------------------
# Minimal, isolated reproduction: each violating sentence, alone, as its own
# single-sentence "paragraph", the smallest input that exercises the gate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sentence", VIOLATING_SENTENCES)
def test_meta_evaluation_framing_sentence_is_dropped_not_delivered(sentence):
    uncited = [{"tag": "framing", "sentence": sentence}]

    rebuilt, links, stats, _healed, surviving_uncited = _finalize_paragraph_text(
        sentence, [], uncited, {}
    )

    assert rebuilt == ""
    assert links == []
    assert surviving_uncited == []
    assert stats["sentences_removed_meta_evaluation_uncited"] == 1


def test_an_innocent_framing_sentence_survives():
    """Non-regression: an uncited framing sentence carrying no comparative
    meta-evaluation of the literature -- the real run's own next paragraph opener,
    `writing_result_2.json` -- is kept exactly as written."""
    uncited = [{"tag": "framing", "sentence": INNOCENT_FRAMING_SENTENCE}]

    rebuilt, _links, stats, _healed, surviving_uncited = _finalize_paragraph_text(
        INNOCENT_FRAMING_SENTENCE, [], uncited, {}
    )

    assert rebuilt == INNOCENT_FRAMING_SENTENCE
    assert surviving_uncited == [{"tag": "framing", "sentence": INNOCENT_FRAMING_SENTENCE}]
    assert stats["sentences_removed_meta_evaluation_uncited"] == 0


# ---------------------------------------------------------------------------
# The real run: re-finalizing the run's own saved `writing_result_*.json` (read-only)
# drops each violating sentence and keeps its innocent neighbours untouched.
# ---------------------------------------------------------------------------


def _load_writing_result(index: int) -> dict:
    run_dir = skip_unless_run_dir(RUN_DIR)
    return json.loads((run_dir / f"writing_result_{index}.json").read_text(encoding="utf-8"))


def _claim_status_verifying_every_link(citation_links: list[dict]) -> dict:
    """Every currently-linked sentence is already known-good -- it survived into this
    same `content`, the write job's own already-finalized text -- so re-finalizing
    with every link re-affirmed "verified" does not remove any text finalize already
    decided to keep (the same template `test_finalize_truncated_fragment_gate.py`'s
    own `_claim_status_verifying_every_link` uses)."""
    return {
        (link["sentence"], link.get("proposition") or link["sentence"], key): "verified"
        for link in citation_links
        for key in link.get("keys") or []
    }


def test_section2_real_run_drops_the_meta_evaluation_sentence_keeps_its_neighbours():
    writing_result = _load_writing_result(2)
    text = writing_result["content"]
    citation_links = writing_result["citation_links"]
    uncited_sentences = writing_result["uncited_sentences"]
    assert VIOLATING_SENTENCES[0] in text, (
        "fixture sanity: the real run's own violating sentence must still be present"
    )
    claim_status = _claim_status_verifying_every_link(citation_links)

    result = finalize_generated_section(text, citation_links, uncited_sentences, claim_status)

    assert VIOLATING_SENTENCES[0] not in result.text
    assert INNOCENT_FRAMING_SENTENCE in result.text
    assert "Automated feedback introduces a further complication." in result.text
    assert result.stats["sentences_removed_meta_evaluation_uncited"] == 1


def test_section5_real_run_drops_the_meta_evaluation_sentence():
    writing_result = _load_writing_result(5)
    text = writing_result["content"]
    citation_links = writing_result["citation_links"]
    uncited_sentences = writing_result["uncited_sentences"]
    assert VIOLATING_SENTENCES[1] in text, (
        "fixture sanity: the real run's own violating sentence must still be present"
    )
    claim_status = _claim_status_verifying_every_link(citation_links)

    result = finalize_generated_section(text, citation_links, uncited_sentences, claim_status)

    assert VIOLATING_SENTENCES[1] not in result.text
    assert result.stats["sentences_removed_meta_evaluation_uncited"] == 1


# ---------------------------------------------------------------------------
# Zero-drift fixture: the backend's own predicate and `demo/check_delivered.py`
# rule 11's own predicate must agree, sentence for sentence, or this test fails.
# ---------------------------------------------------------------------------


def _load_check_delivered():
    spec = importlib.util.spec_from_file_location(
        "check_delivered_zero_drift_meta_eval", REPO_ROOT / "demo" / "check_delivered.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


ZERO_DRIFT_FIXTURE_SENTENCES = [
    *VIOLATING_SENTENCES,
    INNOCENT_FRAMING_SENTENCE,
    "Automated feedback introduces a further complication.",
    "Revision behaviour is treated most directly by Yallop et al. (2021).",
    "The strongest direct evidence comes from Bonilla Lopez et al. (2018).",
    "Recall improved dramatically compared with the baseline (Smith, 2020).",
    "",
]


def test_backend_meta_evaluation_predicate_matches_check_delivered_rule11_zero_drift():
    check_delivered = _load_check_delivered()
    for sentence in ZERO_DRIFT_FIXTURE_SENTENCES:
        backend_flag = sentence_coverage.has_meta_evaluation(sentence)
        checker_flag = bool(check_delivered._META_EVALUATION_RE.search(sentence))
        assert backend_flag == checker_flag, (
            f"drift on {sentence!r}: backend={backend_flag} checker={checker_flag}"
        )


# ---------------------------------------------------------------------------
# A CITED sentence can carry the same guard's own harm too, and the coverage
# gate's own proposition-only check cannot see it when the comparison sits
# inside an accepted frame span rather than inside any verified proposition:
# an observed "Li and Hebert (2023) provide stronger evidence: ..." sentence
# never trips `has_meta_evaluation(kept_propositions)` above, because
# "provide stronger evidence" is the sentence's own accepted colon lead-in
# (frame class F2), never part of either of its two verified propositions.
# `demo/check_delivered.py` rule 11 catches it downstream, on the delivered
# text, exactly as it caught the two uncited sentences above; the app's own
# gate did not, until this fix checks the sentence's own final, whole delivered
# text once more, after the coverage decision and the lead-in trim.
# ---------------------------------------------------------------------------

STRONGER_EVIDENCE_SENTENCE = (
    "Li and Hébert (2023) provide stronger evidence: a paired t-test on 12 L2 "
    "writers showed paper ratings improved by a mean of 1.1 points (Cohen's d "
    "= 1.62) after peer feedback, though four students ignored unity-related "
    "feedback and two ignored word-use feedback."
)
STRONGER_EVIDENCE_PROPOSITIONS = [
    "a paired t-test on 12 L2 writers showed paper ratings improved by a mean "
    "of 1.1 points (Cohen's d = 1.62) after peer feedback",
    "though four students ignored unity-related feedback and two ignored "
    "word-use feedback",
]


def _stronger_evidence_links_and_status():
    links = [
        {
            "sentence": STRONGER_EVIDENCE_SENTENCE,
            "proposition": proposition,
            "keys": ["li_2023"],
            "citation_text": "Li and Hébert (2023)",
        }
        for proposition in STRONGER_EVIDENCE_PROPOSITIONS
    ]
    claim_status = {
        (STRONGER_EVIDENCE_SENTENCE, proposition, "li_2023"): "verified"
        for proposition in STRONGER_EVIDENCE_PROPOSITIONS
    }
    return links, claim_status


def test_meta_evaluation_folded_into_an_accepted_frame_span_removes_a_cited_sentence():
    links, claim_status = _stronger_evidence_links_and_status()

    rebuilt, kept_links, stats, _healed, _uncited = _finalize_paragraph_text(
        STRONGER_EVIDENCE_SENTENCE, links, [], claim_status
    )

    assert rebuilt == ""
    assert kept_links == []
    assert stats["sentences_removed_meta_evaluation_uncited"] == 1
    assert stats["sentences_removed_coverage_incomplete"] == 0
    assert stats["meta_evaluation_reasons"] == [
        {"sentence": STRONGER_EVIDENCE_SENTENCE, "matched": "stronger evidence"}
    ]


def _cited_case(sentence: str) -> tuple[list[dict], dict]:
    """One citation link whose own proposition is the sentence's own full text --
    trivially, verbatim covered, so the coverage decision leaves no residue at all
    and the sentence reaches the new whole-text gate exactly as written, the
    narrowest possible harness for checking that gate alone."""
    links = [{
        "sentence": sentence,
        "proposition": sentence,
        "keys": ["k"],
        "citation_text": "",
    }]
    claim_status = {(sentence, sentence, "k"): "verified"}
    return links, claim_status


@pytest.mark.parametrize("sentence", ORDINARY_COMPARATIVE_SENTENCES_FROM_DELIVERED_DRAFTS)
def test_ordinary_comparative_cited_sentence_survives_the_whole_text_gate(sentence):
    links, claim_status = _cited_case(sentence)

    rebuilt, _kept_links, stats, _healed, _uncited = _finalize_paragraph_text(
        sentence, links, [], claim_status
    )

    assert rebuilt == sentence
    assert stats["sentences_removed_meta_evaluation_uncited"] == 0

"""The evidence step's targets, measured on the twelve demo seed papers.

The unit tests in ``test_deep_analysis_evidence.py`` pin one acceptance rule each against
a chunk written to exercise it. These pin the whole rule set against real extracted text
and real model output: the stored fixture under ``tests/fixtures/evidence/seeds12`` is
every evidence item the analysis agent proposed for the twelve seed papers of
``demo/protocol.json`` in a real run, together with excerpts of those papers' own
stored ``fulltext_chunks`` (see that directory's README for how it was captured and
reduced).

The two targets: at least six accepted evidence items for every paper with full text,
and a verbatim rate of at least 0.8 across the twelve.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

import pytest  # noqa: E402

from app.services.deep_analysis import validate_evidence_items  # noqa: E402

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "evidence" / "seeds12"

#: Six items give the writer's evidence context a real choice for a
#: paper, across more than one concept, rather than one quote it has to use or drop the
#: citation.
TARGET_ITEMS_PER_PAPER = 6
TARGET_VERBATIM_RATE = 0.8


def _manifest() -> dict:
    return json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


def _papers() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(FIXTURE_DIR.glob("*.json"))
        if path.name != "manifest.json"
    ]


def _accept(paper: dict) -> tuple[list[dict], int]:
    return validate_evidence_items(paper["proposed_items"], paper["chunks"])


def test_the_fixture_holds_all_twelve_seed_papers():
    papers = _papers()
    assert len(papers) == 12
    assert {p["paper_id"] for p in papers} == {
        p["paper_id"] for p in _manifest()["per_paper"]
    }
    assert _manifest()["analysis_prompt_version"].startswith("sha256:")


@pytest.mark.parametrize("paper", _papers(), ids=lambda p: p["paper_id"][:8])
def test_every_seed_paper_yields_at_least_six_accepted_evidence_items(paper):
    accepted, _ = _accept(paper)
    assert len(accepted) >= TARGET_ITEMS_PER_PAPER, paper["title"]


@pytest.mark.parametrize("paper", _papers(), ids=lambda p: p["paper_id"][:8])
def test_no_seed_paper_accepts_less_than_it_did_when_the_fixture_was_captured(paper):
    """A change to the acceptance rules may accept more of the same proposed items; it
    may not quietly accept fewer."""
    accepted, _ = _accept(paper)
    assert len(accepted) >= paper["accepted_at_capture"], paper["title"]


def test_the_verbatim_rate_over_the_twelve_seeds_clears_the_target():
    accepted_total = 0
    rejected_total = 0
    for paper in _papers():
        accepted, rejected = _accept(paper)
        accepted_total += len(accepted)
        rejected_total += rejected
    rate = accepted_total / (accepted_total + rejected_total)
    assert rate >= TARGET_VERBATIM_RATE, (accepted_total, rejected_total, rate)


@pytest.mark.parametrize("paper", _papers(), ids=lambda p: p["paper_id"][:8])
def test_every_accepted_quote_is_verbatim_in_the_chunk_it_names(paper):
    """Checked a second time, and by other code than the rule that accepted it: the
    canonical fold the claim-verification guards use
    (``app.services.fulltext._normalise_for_match``) has to find every stored quote inside
    the chunk the stored ``chunk_index`` points at."""
    from app.services.fulltext import _normalise_for_match

    accepted, _ = _accept(paper)
    assert accepted
    for row in accepted:
        chunk_text = paper["chunks"][row["chunk_index"]]["text"]
        assert _normalise_for_match(row["quote"]) in _normalise_for_match(chunk_text), (
            row["quote"][:80]
        )
        assert row["section"] == paper["chunks"][row["chunk_index"]]["section"]


@pytest.mark.parametrize("paper", _papers(), ids=lambda p: p["paper_id"][:8])
def test_every_accepted_quote_ends_a_sentence_and_is_long_enough(paper):
    from app.services.deep_analysis import MIN_EVIDENCE_WORDS

    accepted, _ = _accept(paper)
    for row in accepted:
        assert len(row["quote"].split()) >= MIN_EVIDENCE_WORDS
        assert row["quote"].rstrip("\"')]}»›”’")[-1] in ".!?", row["quote"][-40:]

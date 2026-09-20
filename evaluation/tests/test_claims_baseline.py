"""claims/baseline_lexical: covers only the sixth `over_specified` HSS bucket.

The SciFact and HSS lexical-baseline tests already live in test_claims_scifact.py; this
file covers only the sixth `over_specified` HSS bucket, no network, no LLM.
"""

from __future__ import annotations

import baseline_lexical as bl


def test_hss_rules_extended_to_six_matches_summarize():
    assert bl.HSS_RULES == (
        "verbatim", "paraphrase", "altered", "over_specified", "wrong_paper", "no_full_text",
    )


def test_evaluate_lexical_hss_scores_zero_on_over_specified_by_construction():
    """The lexical baseline can never answer needs_nuance, so an over_specified item (whose
    only correct answer is needs_nuance) is always wrong, regardless of lexical overlap."""
    src = "Scores improved significantly for the treatment group over the semester overall."
    items = [
        {"item_id": "hss-over_specified-01", "rule": "over_specified",
         "expected": ["needs_nuance"], "claim": src + ", among adult learners.",
         "chunks": [src]},
    ]
    out = bl.evaluate_lexical_hss(items, 0.5)
    assert "over_specified" in out["by_rule"]
    assert out["by_rule"]["over_specified"]["n"] == 1
    assert out["by_rule"]["over_specified"]["correct"] == 0


def test_default_results_dir_is_the_result_of_record():
    """RESULTS_DIR must point at claims/results/v6 (the directory of record,
    evaluation/README.md "Results directories"), not a superseded version."""
    assert bl.RESULTS_DIR == bl.HERE / "results" / "v6"

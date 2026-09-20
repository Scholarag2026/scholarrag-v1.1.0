"""M2 regression: the numeric-consistency diagnostic must fire on exactly one ``altered``
item and none of the ``verbatim``/``paraphrase``/``wrong_paper`` items, using a redacted
fixture of the retired 30-item HSS set's verifier run and the real cached full text. No LLM
call.

An unconditional times-100 numeral expansion (``_numeral_candidates``) would let a claim
numeral such as 1.62 collide with unrelated source numerals such as 161/163 through the
scaled form, firing on 0 of 10 HSS ``altered`` items instead of the expected 1
(``hss-altered-01``). This test pins the guard's bucket counts against the real data (not a
synthetic fixture) so that collision cannot silently come back. Skipped outright when the
full-text cache is absent, mirroring ``test_claims_quote_fidelity_guard.py`` (the M1
regression).

Tier A was renamed ``_guard_numeric_tier_a`` -> ``_diagnose_numeric_not_in_source`` and
demoted to a diagnostic: it no longer takes or returns a ``status``, only the reason list.
But the pinned firing buckets below are unchanged, since the underlying arithmetic did not
change, only what the caller does with the result.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import pytest

from common import read_jsonl, scifact_gold_for_doc

CLAIMS_DIR = Path(__file__).resolve().parents[1] / "claims"
RESULTS_DIR = CLAIMS_DIR / "results"
# Redacted fixture of the retired 30-item HSS set's verifier run this M2 regression is
# pinned to (see evaluation/tests/fixtures/hss_v1_guards/README.md).
V1_RESULTS_DIR = Path(__file__).resolve().parent / "fixtures" / "hss_v1_guards"
CACHE_DIR = CLAIMS_DIR / "data" / "hss_fulltext"
BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
SCIFACT_DATA_DIR = CLAIMS_DIR / "data" / "data"

# Brief section 2, M2 (measured read-only on the frozen HSS items and cached full texts;
# reproduced here as a regression against the real, fixed guard). "no_full_text" items are
# excluded: they never reach the guard (no chunks), so 5+5+10+5 = 25 chunk-bearing items.
EXPECTED_FIRES: dict[str, int] = {
    "verbatim": 0,
    "paraphrase": 0,
    "altered": 1,
    "wrong_paper": 0,
}

pytestmark = pytest.mark.skipif(
    not CACHE_DIR.is_dir() or not any(CACHE_DIR.glob("*.json")),
    reason="HSS full-text cache (claims/data/hss_fulltext/) is gitignored and absent",
)


def _import_guard():
    """Import the real ``_diagnose_numeric_not_in_source`` from
    ``backend/app/services/fulltext.py`` (renamed from ``_guard_numeric_tier_a``).

    Restores the working directory afterwards, mirroring
    ``test_claims_quote_fidelity_guard.py``: ``app.config.Settings`` reads ``.env`` relative
    to the cwd, so the import briefly needs ``cwd == backend/``.
    """
    original_cwd = Path.cwd()
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    try:
        os.chdir(BACKEND_ROOT)
        from app.services.fulltext import _diagnose_numeric_not_in_source
    finally:
        os.chdir(original_cwd)
    return _diagnose_numeric_not_in_source


def _load_fulltext_cache() -> dict[str, dict]:
    import build_hss_set as bh  # evaluation/claims; already on sys.path via conftest.py

    return bh.load_cache(CACHE_DIR)


def _chunk_texts_for(row: dict, cache: dict[str, dict]) -> list[str]:
    paper = cache.get(row.get("chunk_doi"))
    if paper is None:
        return []
    return [c["text"] for c in paper["chunks"]]


@pytest.mark.parametrize("run", ["A", "B"])
def test_numeric_tier_a_fires_on_exactly_the_brief_m2_items(run: str) -> None:
    diagnose_numeric_not_in_source = _import_guard()
    cache = _load_fulltext_cache()
    rows = read_jsonl(V1_RESULTS_DIR / f"hss_run{run}.jsonl")
    assert rows, f"hss_run{run}.jsonl is empty or missing"

    chunk_bearing = [r for r in rows if r.get("rule") != "no_full_text"]
    assert len(chunk_bearing) == 25, "5 verbatim + 5 paraphrase + 10 altered + 5 wrong_paper"

    fires_by_rule: Counter[str] = Counter()
    fired_altered_ids: list[str] = []
    for row in chunk_bearing:
        chunk_texts = _chunk_texts_for(row, cache)
        assert chunk_texts, f"{row.get('item_id')}: no cached full text found"
        reasons = diagnose_numeric_not_in_source(
            claim_text=row["claim"], chunk_texts=chunk_texts
        )
        if reasons:
            fires_by_rule[row["rule"]] += 1
            if row["rule"] == "altered":
                fired_altered_ids.append(row["item_id"])

    for rule, expected in EXPECTED_FIRES.items():
        assert fires_by_rule.get(rule, 0) == expected, (
            f"tier A fired on {fires_by_rule.get(rule, 0)} of the {rule} items in run {run}; "
            f"the expected count is {expected} (bucket counts: {dict(fires_by_rule)})"
        )
    assert fired_altered_ids == ["hss-altered-01"], (
        "The single altered-bucket fire must be hss-altered-01 (missing 1.62), "
        f"got {fired_altered_ids} in run {run}"
    )


# --------------------------------------------------------------------------------------
# tier A's SUPPORT-gold false-fire rate on the SciFact **train** split (the designated
# development data; read-only, no model call: the fixture is the raw claim/corpus data, not
# a model-scored results file). Without ``_numeral_is_identifier_fragment`` (a structural
# filter, not a lexicon of specific names), tier A would cap 15 of 370 SUPPORT-gold pairs
# to `unsupported` regardless of the model's answer, a hard ceiling on verified recall with
# no compensating gain (bare digits inside identifiers such as "NIH 3T3" or "A-769662" have
# no notion of a quantity). The filter reduces this to 9. This test pins the reduced count,
# not zero: the residual 9 are documented, real limitations of a literal-numeral guard (see
# that function's docstring). Skipped outright when the SciFact train/corpus data is absent
# (gitignored; `fetch_scifact.py` downloads it), mirroring the HSS-cache skip above.
#
# The identifier filter applies only to the CLAIM's own numerals, not symmetrically to the
# source-side candidate pool: applying it to both sides could only ever manufacture a
# spurious "absent" verdict (never prevent a real one), since a claim and its source
# routinely spell the same identifier or fused unit differently. Restricting the filter to
# the CLAIM's own numerals (``_extract_numerals(..., skip_identifiers=False)`` on every
# chunk-text call) leaves the SUPPORT and CONTRADICT firing counts unchanged and removes
# exactly 3 spurious NOT_ENOUGH_INFO fires this asymmetry would otherwise manufacture
# (44 -> 41).
# --------------------------------------------------------------------------------------

SCIFACT_TRAIN_EXPECTED_FIRES: dict[str, int] = {
    "SUPPORT": 9,
    "CONTRADICT": 5,
    "NOT_ENOUGH_INFO": 41,
}
SCIFACT_TRAIN_EXPECTED_TOTALS: dict[str, int] = {
    "SUPPORT": 370,
    "CONTRADICT": 194,
    "NOT_ENOUGH_INFO": 355,
}
#: The exact fired-SUPPORT set, so a future change to the filter is deliberate.
SCIFACT_TRAIN_EXPECTED_SUPPORT_FIRES: list[str] = sorted(
    [
        "44:56893404",
        "426:16728949",
        "428:16728949",
        "506:7433668",
        "955:2078658",
        "955:30507607",
        "1028:13923140",
        "1040:25254425",
        "1297:9167230",
    ]
)


def _scifact_train_data_present() -> bool:
    return (SCIFACT_DATA_DIR / "claims_train.jsonl").is_file() and (
        SCIFACT_DATA_DIR / "corpus.jsonl"
    ).is_file()


@pytest.mark.skipif(
    not _scifact_train_data_present(),
    reason="SciFact train/corpus data (claims/data/data/) is gitignored and absent",
)
def test_numeric_tier_a_support_gold_firing_rate_on_scifact_train_dev_data() -> None:
    diagnose_numeric_not_in_source = _import_guard()
    claims = read_jsonl(SCIFACT_DATA_DIR / "claims_train.jsonl")
    corpus = {int(d["doc_id"]): d for d in read_jsonl(SCIFACT_DATA_DIR / "corpus.jsonl")}

    fires: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    fired_support: list[str] = []
    n_pairs = 0
    for claim in claims:
        for doc_id in claim.get("cited_doc_ids") or []:
            doc = corpus.get(int(doc_id))
            if doc is None:
                continue
            n_pairs += 1
            chunk = " ".join(
                s.strip() for s in doc.get("abstract") or [] if s and s.strip()
            )
            gold = scifact_gold_for_doc(claim, doc_id)
            totals[gold] += 1
            reasons = diagnose_numeric_not_in_source(
                claim_text=claim["claim"],
                chunk_texts=[chunk] if chunk else [],
            )
            if reasons:
                fires[gold] += 1
                if gold == "SUPPORT":
                    fired_support.append(f"{claim['id']}:{doc_id}")

    assert n_pairs == 919, f"brief section 5.1: expected 919 claim-document pairs, got {n_pairs}"
    assert dict(totals) == SCIFACT_TRAIN_EXPECTED_TOTALS
    for gold, expected in SCIFACT_TRAIN_EXPECTED_FIRES.items():
        assert fires.get(gold, 0) == expected, (
            f"tier A fired on {fires.get(gold, 0)} of the {gold} SciFact-train pairs; "
            f"expected {expected} (all fires: {dict(fires)})"
        )
    assert sorted(fired_support) == SCIFACT_TRAIN_EXPECTED_SUPPORT_FIRES

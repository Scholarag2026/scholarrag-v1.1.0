"""M1 regression: the shared Unicode-typography normaliser
(``app.services.fulltext._normalise_for_match`` /
``_quote_segments``) must recover almost all "non verbatim" HSS quote locations without
ever demoting a genuinely correct "verified" answer.

Runs against a redacted fixture of the retired 30-item HSS set's verifier run
(``evaluation/tests/fixtures/hss_v1_guards/hss_run{A,B}.jsonl``) and the real cached full
text (``claims/data/hss_fulltext/``, gitignored). No LLM call. Skipped outright when the
full-text cache is absent, since it is not present on a fresh checkout.

This intentionally imports the real ``app.services.fulltext`` guard functions rather than
reimplementing them, so the regression tracks the shipped guard, not a copy of it. That
import needs the ambient ``python`` that also runs the backend suite (which has FastAPI /
SQLAlchemy / pymupdf installed), not the lean ``evaluation/.venv`` used for paid harness
runs; ``app.config.Settings`` fields all default, so no ``.env`` or API key is required
just to import the module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from common import read_jsonl

CLAIMS_DIR = Path(__file__).resolve().parents[1] / "claims"
RESULTS_DIR = CLAIMS_DIR / "results"
# Redacted fixture of the retired 30-item HSS set's verifier run this M1 regression is
# pinned to (see evaluation/tests/fixtures/hss_v1_guards/README.md).
V1_RESULTS_DIR = Path(__file__).resolve().parent / "fixtures" / "hss_v1_guards"
CACHE_DIR = CLAIMS_DIR / "data" / "hss_fulltext"
BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"

# Brief section 2, M1 (measured on the frozen v1 runs; reproduced here as a regression).
EXPECTED = {
    "A": {"with_quote": 23, "old_verbatim": 11, "new_located": 20},
    "B": {"with_quote": 25, "old_verbatim": 10, "new_located": 20},
}

pytestmark = pytest.mark.skipif(
    not CACHE_DIR.is_dir() or not any(CACHE_DIR.glob("*.json")),
    reason="HSS full-text cache (claims/data/hss_fulltext/) is gitignored and absent",
)


def _import_guard_functions():
    """Import the real guard functions from ``backend/app/services/fulltext.py``.

    Restores the working directory afterwards: ``app.config.Settings`` reads ``.env``
    relative to the cwd, so the import briefly needs ``cwd == backend/``, but nothing
    else in this test suite may be left running from there.
    """
    original_cwd = Path.cwd()
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    try:
        os.chdir(BACKEND_ROOT)
        from app.services.fulltext import _find_evidence_location, _guard_quote_fidelity
    finally:
        os.chdir(original_cwd)
    return _find_evidence_location, _guard_quote_fidelity


def _load_fulltext_cache() -> dict[str, dict]:
    import build_hss_set as bh  # evaluation/claims; already on sys.path via conftest.py

    return bh.load_cache(CACHE_DIR)


def _chunks_for(row: dict, cache: dict[str, dict]) -> list[dict]:
    paper = cache.get(row.get("chunk_doi"))
    if paper is None:
        return []
    return [{"section": c.get("section"), "text": c["text"]} for c in paper["chunks"]]


@pytest.mark.parametrize("run", ["A", "B"])
def test_quote_fidelity_guard_recovers_almost_all_hss_verified_quotes(run: str) -> None:
    find_evidence_location, guard_quote_fidelity = _import_guard_functions()
    cache = _load_fulltext_cache()
    rows = read_jsonl(V1_RESULTS_DIR / f"hss_run{run}.jsonl")
    assert rows, f"hss_run{run}.jsonl is empty or missing"

    verified_rows = [r for r in rows if r.get("predicted_status") == "verified"]
    assert len(verified_rows) == 10, "the frozen v1 run always answers verified on 10/10"

    quoted_rows = [r for r in rows if r.get("evidence_quote")]
    old_verbatim = sum(1 for r in quoted_rows if r.get("quote_is_verbatim"))
    new_located = sum(
        1
        for r in quoted_rows
        if find_evidence_location(r["evidence_quote"], _chunks_for(r, cache)) is not None
    )

    expected = EXPECTED[run]
    assert len(quoted_rows) == expected["with_quote"]
    assert old_verbatim == expected["old_verbatim"]
    assert new_located == expected["new_located"]

    guard_fires_on_verified = 0
    for row in verified_rows:
        chunk_texts = [c["text"] for c in _chunks_for(row, cache)]
        _status, reasons = guard_quote_fidelity(
            "verified", evidence_quote=row.get("evidence_quote"), chunk_texts=chunk_texts
        )
        if reasons:
            guard_fires_on_verified += 1
    assert guard_fires_on_verified == 0, (
        "the quote fidelity guard must fire on 0 of the 10 correct HSS verified rows "
        f"in run {run} (brief section 10, risk 2), or it would drive accuracy down"
    )


# --------------------------------------------------------------------------------------
# The guard normaliser must fold a PDF line-break hyphenation ("post-\ntest", surviving
# whitespace collapse as "post- test") to match however the model happened to quote the
# same word (joined or contiguously hyphenated). Four frozen-run misses this construction
# defect caused on a genuinely correct `verified` answer: `hss-verbatim-01`,
# `hss-paraphrase-04` (HSS v3 set), `real-test-12` and `real-test-20` (real set). Runs
# against a redacted fixture of the pre-fix rows that documented the miss (the run itself
# was superseded once the re-run under the fixed guard, whose rows no longer reproduce the
# miss this test pins, took over `claims/results/v3/`), and the real cached full text
# (`claims/data/hss_test_v3_fulltext/`, `claims/data/real_claims_fulltext/`, both
# gitignored). No LLM call.
# --------------------------------------------------------------------------------------

V3_MISS_RESULTS_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "v3_line_break_hyphenation_misses"
)

V3_HSS_CLAIMS = CLAIMS_DIR / "hss_test_v3_claims.jsonl"
V3_REAL_CLAIMS = CLAIMS_DIR / "real_claims_test.jsonl"
V3_HSS_CACHE_DIR = CLAIMS_DIR / "data" / "hss_test_v3_fulltext"
V3_REAL_CACHE_DIR = CLAIMS_DIR / "data" / "real_claims_fulltext"

# (item_id, dataset): dataset selects which claims file, run files and cache directory to
# use. Every named row is a `quote_not_verbatim` miss on a `model_status == "verified"` row
# in at least one of its two runs.
LINE_BREAK_HYPHENATION_MISSES = [
    ("hss-verbatim-01", "hss"),
    ("hss-paraphrase-04", "hss"),
    ("real-test-12", "real"),
    ("real-test-20", "real"),
]

v3_cache_missing = not V3_HSS_CACHE_DIR.is_dir() or not V3_REAL_CACHE_DIR.is_dir()


def _load_v3_chunk_doi(item_id: str, dataset: str) -> str:
    claims_path = V3_HSS_CLAIMS if dataset == "hss" else V3_REAL_CLAIMS
    for row in read_jsonl(claims_path):
        if row.get("item_id") == item_id:
            return row["chunk_doi"]
    raise AssertionError(f"{item_id} not found in {claims_path}")


def _load_v3_chunk_texts(chunk_doi: str, dataset: str) -> list[str]:
    import build_hss_set as bh  # evaluation/claims; already on sys.path via conftest.py

    cache_dir = V3_HSS_CACHE_DIR if dataset == "hss" else V3_REAL_CACHE_DIR
    cache = bh.load_cache(cache_dir, dois=[chunk_doi])
    paper = cache.get(chunk_doi)
    assert paper is not None, f"{chunk_doi} not found in {cache_dir}"
    return [c["text"] for c in paper["chunks"]]


def _v3_miss_rows(item_id: str, dataset: str) -> list[dict]:
    """Every row for *item_id*, across both frozen runs, that reproduces the documented
    miss: the model answered ``verified`` and the guard demoted it on `quote_not_verbatim`
    alone."""
    run_names = ["hss_runA", "hss_runB"] if dataset == "hss" else ["real_runA", "real_runB"]
    rows = []
    for run_name in run_names:
        for row in read_jsonl(V3_MISS_RESULTS_DIR / f"{run_name}.jsonl"):
            if (
                row.get("item_id") == item_id
                and row.get("model_status") == "verified"
                and row.get("machine_reasons") == ["quote_not_verbatim"]
            ):
                rows.append(row)
    return rows


@pytest.mark.skipif(
    v3_cache_missing,
    reason=(
        "v3 full-text caches (claims/data/hss_test_v3_fulltext/, "
        "claims/data/real_claims_fulltext/) are gitignored and absent"
    ),
)
@pytest.mark.parametrize("item_id, dataset", LINE_BREAK_HYPHENATION_MISSES)
def test_quote_fidelity_guard_no_longer_fires_on_the_v3_line_break_hyphenation_misses(
    item_id: str, dataset: str
) -> None:
    _find_evidence_location, guard_quote_fidelity = _import_guard_functions()
    chunk_doi = _load_v3_chunk_doi(item_id, dataset)
    chunk_texts = _load_v3_chunk_texts(chunk_doi, dataset)

    miss_rows = _v3_miss_rows(item_id, dataset)
    assert miss_rows, (
        f"{item_id}: expected at least one frozen run with model_status=verified and "
        "machine_reasons=['quote_not_verbatim'] (the documented miss); none found -- "
        "either the fixture drifted or the row schema changed"
    )
    for row in miss_rows:
        status, reasons = guard_quote_fidelity(
            "verified",
            evidence_quote=row.get("evidence_quote"),
            evidence_quotes=row.get("evidence_quotes"),
            chunk_texts=chunk_texts,
        )
        assert reasons == [], (
            f"{item_id}: quote fidelity guard still fires ({reasons}) on the committed "
            f"evidence_quotes {row.get('evidence_quotes')!r} against the cached text for "
            f"{chunk_doi} -- the line-break-hyphenation fold did not fix this row"
        )
        assert status == "verified"

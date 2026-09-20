"""`claims/freeze_v6.py`: the pure per-row decision logic, and an end-to-end determinism
check of the real guard replay against the committed `results/v6/inputs/hss_runA.jsonl` file.

The determinism check imports the real `app.services.fulltext` (no network, no LLM call --
guard 7 and `verify_claim_with_policy` are pure functions of their stored-answer input) and
restores the working directory afterwards, mirroring the one other test in this suite that
imports the real backend guard directly
(`test_claims_verify_common.py::test_harness_guard_entry_point_has_parity_with_the_real_backend_normaliser`).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import freeze_v6 as fv

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"


def _import_fulltext():
    """Import the real backend guard/policy module, chdir'd into `backend/` as production
    code requires; the caller must restore the original working directory afterwards."""
    os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    os.chdir(BACKEND_ROOT)
    from app.services import fulltext
    from app.services.fulltext import GUARD_DIGEST, ModelPassAnswer, verify_claim_with_policy

    return fulltext, ModelPassAnswer, verify_claim_with_policy, GUARD_DIGEST


# --------------------------------------------------------------------------------------
# Pure helpers: no backend import
# --------------------------------------------------------------------------------------


def test_quotes_pool_prefers_evidence_quotes_over_the_singular_field():
    row = {"evidence_quotes": ["a", "b"], "evidence_quote": "a"}
    assert fv.quotes_pool(row) == ["a", "b"]


def test_quotes_pool_falls_back_to_the_singular_field():
    row = {"evidence_quotes": [], "evidence_quote": "only one"}
    assert fv.quotes_pool(row) == ["only one"]


def test_quotes_pool_drops_falsy_entries():
    row = {"evidence_quotes": ["a", "", None, "b"]}
    assert fv.quotes_pool(row) == ["a", "b"]


def _replay(before_status, before_reasons, after_status, after_reasons, *, anomaly=False):
    return {
        "before": {
            "status": before_status, "machine_reasons": before_reasons,
            "repair_turn_would_fire": anomaly,
        },
        "after": {
            "status": after_status, "machine_reasons": after_reasons,
            "repair_turn_would_fire": anomaly,
        },
    }


def test_apply_v6_status_keeps_v5_status_when_guard_7_finds_nothing():
    v5_row = {"item_id": "x", "predicted_status": "verified", "machine_reasons": []}
    replay = _replay("verified", [], "verified", [])
    applied = fv.apply_v6_status(v5_row, replay)
    assert applied["status"] == "verified"
    assert applied["machine_reasons"] == []
    assert applied["provenance"]["status_changed"] is False
    assert applied["provenance"]["guard7_examined"] is True


def test_apply_v6_status_applies_the_demotion_when_v5_status_was_verified():
    v5_row = {"item_id": "x", "predicted_status": "verified", "machine_reasons": []}
    replay = _replay("verified", [], "needs_nuance", ["scale_word_mismatch"])
    applied = fv.apply_v6_status(v5_row, replay)
    assert applied["status"] == "needs_nuance"
    assert applied["machine_reasons"] == ["scale_word_mismatch"]
    assert applied["provenance"]["status_changed"] is True
    assert applied["provenance"]["guard7_findings"] == ["scale_word_mismatch"]


def test_apply_v6_status_ignores_the_delta_when_v5_status_was_not_verified():
    """Production's real guard chain, run against the real source text, already demoted
    this row before guard 7 could examine it; the substitution-based replay's own before/
    after (whatever it happens to show) must never override that real, authoritative
    result."""
    v5_row = {
        "item_id": "x", "predicted_status": "unsupported",
        "machine_reasons": ["quote_not_verbatim"],
    }
    replay = _replay("verified", [], "needs_nuance", ["scale_word_mismatch"])
    applied = fv.apply_v6_status(v5_row, replay)
    assert applied["status"] == "unsupported"
    assert applied["machine_reasons"] == ["quote_not_verbatim"]
    assert applied["provenance"]["status_changed"] is False


def test_apply_v6_status_marks_a_repair_turn_anomaly_and_keeps_the_v5_status():
    v5_row = {"item_id": "x", "predicted_status": "verified", "machine_reasons": []}
    replay = _replay(None, None, None, None, anomaly=True)
    applied = fv.apply_v6_status(v5_row, replay)
    assert applied["status"] == "verified"
    assert applied["provenance"]["repair_turn_would_fire"] is True
    assert applied["provenance"]["guard7_examined"] is None


# --------------------------------------------------------------------------------------
# End-to-end determinism, against the real backend and the committed v5 data
# --------------------------------------------------------------------------------------


def test_replay_run_file_is_deterministic_and_changes_nothing_on_hss_run_a():
    original_cwd = Path.cwd()
    try:
        fulltext, model_pass_answer, verify, guard_digest = _import_fulltext()
        rows1, counters1 = asyncio.run(
            fv.replay_run_file("hss", "A", fulltext, model_pass_answer, verify, guard_digest)
        )
        rows2, counters2 = asyncio.run(
            fv.replay_run_file("hss", "A", fulltext, model_pass_answer, verify, guard_digest)
        )
    finally:
        os.chdir(original_cwd)

    assert counters1 == counters2
    # Byte-for-byte equal serialisation, the same claim the module docstring/
    # FREEZE_DIGESTS_v6.md make about running freeze_v6.py twice.
    assert json.dumps(rows1, ensure_ascii=False) == json.dumps(rows2, ensure_ascii=False)
    # Pinned against this data: guard 7 changes no row's status here (FREEZE_DIGESTS_v6.md).
    assert counters1["n_status_changed"] == 0
    assert counters1["n_examined_by_guard7"] == 14
    assert counters1["n_repair_turn_anomalies"] == 0

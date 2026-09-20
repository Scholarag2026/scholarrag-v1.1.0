"""claims/replay_real_test_18: pure helpers only (no network, no backend import).

The script this module tests replays the one cached v3 answer that licenses the shared
verification policy's bounded second pass (`real-test-18`, `real_runA`) through the real
`verify_claim_with_policy` at HEAD, making exactly one live model call. That live call is
out of scope for this test file (it is not faked here, and never will be; see the module's
own docstring for why); only the deterministic, network-free helpers are covered.
"""

from __future__ import annotations

import pytest
import replay_real_test_18 as rrt


def test_item_id_and_paths_target_the_right_row_and_directory() -> None:
    assert rrt.ITEM_ID == "real-test-18"
    assert rrt.OUT_PATH.parent.name == "v4"
    assert rrt.OUT_PATH.name == "real-test-18_policy_replay.json"
    assert rrt.V3_RUN_PATH.name == "real_runA.jsonl"
    assert rrt.V3_RUN_PATH.parent.name == "v3"


def test_find_row_returns_the_matching_row() -> None:
    rows = [{"item_id": "a", "x": 1}, {"item_id": "real-test-18", "x": 2}]
    found = rrt._find_row(rows, "real-test-18", rrt.CLAIMS_PATH)
    assert found == {"item_id": "real-test-18", "x": 2}


def test_find_row_raises_when_the_item_is_missing() -> None:
    with pytest.raises(SystemExit, match="not found"):
        rrt._find_row([{"item_id": "a"}], "real-test-18", rrt.CLAIMS_PATH)


def test_cached_first_pass_kwargs_reads_the_v3_row_shape() -> None:
    cached = {
        "model_status": "verified",
        "evidence_quote": "Q1",
        "evidence_quotes": ["Q1", "Q2"],
        "assertions": [{"text": "t", "verdict": "supported"}],
        "explanation": "expl",
        "suggested_revision": None,
        "unstated_details": [],
        "input_tokens": 30525,
        "output_tokens": 4114,
        "system_fingerprint": "a26a7955944dc5c60445bff77fac9c8e",
        "model_reported": "deepseek-v4-flash",
    }
    kwargs = rrt.cached_first_pass_kwargs(cached)
    assert kwargs == {
        "status": "verified",
        "evidence_quote": "Q1",
        "evidence_quotes": ["Q1", "Q2"],
        "assertions": [{"text": "t", "verdict": "supported"}],
        "explanation": "expl",
        "suggested_revision": None,
        "unstated_details": [],
        "input_tokens": 30525,
        "output_tokens": 4114,
        "system_fingerprint": "a26a7955944dc5c60445bff77fac9c8e",
        "model_reported": "deepseek-v4-flash",
    }
    # Mutating the returned dict's lists must never alias the input row's own lists.
    kwargs["evidence_quotes"].append("Q3")
    kwargs["assertions"][0]["verdict"] = "changed"
    assert cached["evidence_quotes"] == ["Q1", "Q2"]
    assert cached["assertions"][0]["verdict"] == "supported"


def test_cached_first_pass_kwargs_defaults_missing_optional_fields() -> None:
    cached = {
        "model_status": "needs_nuance",
        "evidence_quote": None,
        "input_tokens": None,
        "output_tokens": None,
        "system_fingerprint": None,
        "model_reported": None,
    }
    kwargs = rrt.cached_first_pass_kwargs(cached)
    assert kwargs["evidence_quotes"] == []
    assert kwargs["assertions"] == []
    assert kwargs["unstated_details"] == []
    assert kwargs["suggested_revision"] is None
    assert kwargs["explanation"] is None

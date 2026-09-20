"""Tests for Cohen's kappa inter-coder reliability computation."""

import pytest

from app.services.qualitative import compute_inter_coder_reliability


def test_perfect_agreement():
    ai = {"s1": ["c1"], "s2": ["c2"], "s3": ["c1"]}
    human = {"s1": ["c1"], "s2": ["c2"], "s3": ["c1"]}
    report = compute_inter_coder_reliability(ai, human, ["c1", "c2"])
    assert report["overall_kappa"] == pytest.approx(1.0)
    assert report["agreement_pct"] == pytest.approx(100.0)


def test_no_agreement():
    # Systematic disagreement: each rater uses both codes but always opposite
    ai = {"s1": ["c1"], "s2": ["c2"], "s3": ["c1"], "s4": ["c2"]}
    human = {"s1": ["c2"], "s2": ["c1"], "s3": ["c2"], "s4": ["c1"]}
    report = compute_inter_coder_reliability(ai, human, ["c1", "c2"])
    assert report["overall_kappa"] < 0.0
    assert report["agreement_pct"] == pytest.approx(0.0)


def test_partial_agreement():
    ai = {"s1": ["c1"], "s2": ["c2"], "s3": ["c1"], "s4": ["c2"]}
    human = {"s1": ["c1"], "s2": ["c1"], "s3": ["c1"], "s4": ["c2"]}
    report = compute_inter_coder_reliability(ai, human, ["c1", "c2"])
    assert 0.0 < report["overall_kappa"] < 1.0


def test_disagreement_segments_listed():
    ai = {"s1": ["c1"], "s2": ["c2"]}
    human = {"s1": ["c1"], "s2": ["c1"]}
    report = compute_inter_coder_reliability(ai, human, ["c1", "c2"])
    assert len(report["disagreement_segments"]) == 1
    assert report["disagreement_segments"][0]["segment_id"] == "s2"


def test_per_code_kappa():
    ai = {"s1": ["c1"], "s2": ["c2"], "s3": ["c1"]}
    human = {"s1": ["c1"], "s2": ["c2"], "s3": ["c2"]}
    report = compute_inter_coder_reliability(ai, human, ["c1", "c2"])
    assert len(report["per_code_kappa"]) > 0


# ---------- D5: undefined kappa is null, never fabricated ----------


def test_overall_kappa_is_none_when_undefined():
    """Both coders using one identical label for every segment -> kappa is undefined."""
    ai = {"s1": ["c1"], "s2": ["c1"], "s3": ["c1"]}
    human = {"s1": ["c1"], "s2": ["c1"], "s3": ["c1"]}
    report = compute_inter_coder_reliability(ai, human, ["c1", "c2"])
    assert report["overall_kappa"] is None
    assert report["overall_kappa_explanation"]
    assert report["agreement_pct"] == pytest.approx(100.0)


def test_overall_kappa_is_none_when_no_common_segments():
    report = compute_inter_coder_reliability({"s1": ["c1"]}, {"s2": ["c1"]}, ["c1"])
    assert report["overall_kappa"] is None
    assert "both coders" in report["overall_kappa_explanation"].lower()


def test_per_code_kappa_is_none_when_code_unused_by_both():
    ai = {"s1": ["c1"], "s2": ["c2"]}
    human = {"s1": ["c1"], "s2": ["c2"]}
    report = compute_inter_coder_reliability(ai, human, ["c1", "c2", "c3"])
    by_code = {row["code_name"]: row for row in report["per_code_kappa"]}
    assert by_code["c3"]["kappa"] is None
    assert by_code["c3"]["agreement_pct"] == pytest.approx(100.0)


def test_report_has_no_nan_and_is_json_serializable():
    import json
    import math

    ai = {"s1": ["c1"], "s2": ["c1"]}
    human = {"s1": ["c1"], "s2": ["c1"]}
    report = compute_inter_coder_reliability(ai, human, ["c1", "c2"])
    text = json.dumps(report)
    assert "NaN" not in text
    for row in report["per_code_kappa"]:
        assert row["kappa"] is None or math.isfinite(row["kappa"])


def test_defined_kappa_still_returned_as_float():
    ai = {"s1": ["c1"], "s2": ["c2"], "s3": ["c1"]}
    human = {"s1": ["c1"], "s2": ["c2"], "s3": ["c1"]}
    report = compute_inter_coder_reliability(ai, human, ["c1", "c2"])
    assert report["overall_kappa"] == pytest.approx(1.0)
    assert report["overall_kappa_explanation"] is None

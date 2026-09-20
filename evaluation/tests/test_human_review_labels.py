"""`claims/human_review_labels.py`: the majority-label rule, the two inter-rater agreement
statistics, and the v6-vs-human-label scorer, all on synthetic fixtures -- the real
workbooks are gitignored and not part of any checkout, so nothing here depends on them.
"""

from __future__ import annotations

import json

import human_review_labels as hrl
import pytest

# --------------------------------------------------------------------------------------
# Majority rule
# --------------------------------------------------------------------------------------


def test_majority_label_two_of_three_agree():
    assert hrl.majority_label(["verified", "verified", "needs_nuance"]) == "verified"


def test_majority_label_unanimous():
    assert hrl.majority_label(["unsupported", "unsupported", "unsupported"]) == "unsupported"


def test_majority_label_three_way_split_is_no_majority():
    assert hrl.majority_label(["verified", "needs_nuance", "unsupported"]) is None


# --------------------------------------------------------------------------------------
# Inter-rater agreement
# --------------------------------------------------------------------------------------


def test_fleiss_kappa_is_one_when_every_rater_agrees_on_every_item():
    table = [[3, 0], [0, 3], [3, 0]]  # 3 items, 2 categories, unanimous each time
    assert hrl.fleiss_kappa(table) == 1.0


def test_fleiss_kappa_hand_computed_example():
    # 4 items, 3 raters, 2 categories: two unanimous A, one unanimous B, one 2-1 split.
    # P_i = [1, 1, 1, 1/3], Pbar = 5/6; p_A = 8/12, p_B = 4/12, Pe = (2/3)**2 + (1/3)**2 =
    # 5/9; kappa = (5/6 - 5/9) / (1 - 5/9) = (5/18) / (4/9) = 0.625.
    table = [[3, 0], [3, 0], [0, 3], [2, 1]]
    assert hrl.fleiss_kappa(table) == pytest.approx(0.625, abs=1e-9)


def test_fleiss_kappa_none_when_fewer_than_two_raters():
    assert hrl.fleiss_kappa([[1, 0]]) is None


def test_pairwise_agreement_reports_mean_percent_and_kappa():
    cols = {1: ["a", "a", "b"], 2: ["a", "a", "b"], 3: ["a", "b", "b"]}
    out = hrl.pairwise_agreement(cols)
    assert out["mean_pairwise_percent_agreement"] > 0.6
    assert len(out["pairwise_percent_agreement"]) == 3


# --------------------------------------------------------------------------------------
# Scoring v6 against the human majority label
# --------------------------------------------------------------------------------------


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_score_v6_against_labels_computes_accuracy_kappa_and_falsely_verified(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(hrl, "V6_DIR", tmp_path)
    human_labels = {
        "items": [
            {"item_id": "c1", "set": "constructed", "majority_label": "verified"},
            {"item_id": "c2", "set": "constructed", "majority_label": "unsupported"},
            {"item_id": "c3", "set": "constructed", "majority_label": "needs_nuance"},
            {"item_id": "r1", "set": "real", "majority_label": "verified"},
        ],
    }
    _write_jsonl(tmp_path / "hss_runA.jsonl", [
        {"item_id": "c1", "predicted_status": "verified"},
        {"item_id": "c2", "predicted_status": "verified"},  # falsely verified
        {"item_id": "c3", "predicted_status": "needs_nuance"},
    ])
    _write_jsonl(tmp_path / "hss_runB.jsonl", [
        {"item_id": "c1", "predicted_status": "verified"},
        {"item_id": "c2", "predicted_status": "unsupported"},
        {"item_id": "c3", "predicted_status": "needs_nuance"},
    ])
    _write_jsonl(tmp_path / "real_runA.jsonl", [{"item_id": "r1", "predicted_status": "verified"}])
    _write_jsonl(tmp_path / "real_runB.jsonl", [{"item_id": "r1", "predicted_status": "verified"}])

    scores = hrl.score_v6_against_labels(human_labels)

    run_a = scores["by_set"]["constructed"]["run_A"]
    assert run_a["n"] == 3
    assert run_a["n_correct"] == 2
    assert run_a["accuracy"] == pytest.approx(2 / 3, abs=1e-9)
    assert run_a["falsely_verified"] == 1

    run_b = scores["by_set"]["constructed"]["run_B"]
    assert run_b["n_correct"] == 3
    assert run_b["accuracy"] == 1.0
    assert run_b["falsely_verified"] == 0

    real_a = scores["by_set"]["real"]["run_A"]
    assert real_a["n"] == 1
    assert real_a["accuracy"] == 1.0
    assert real_a["verified_precision"] == 1.0
    assert real_a["verified_recall"] == 1.0


# --------------------------------------------------------------------------------------
# CLI: no hard-coded workbook paths, a no-argument run refuses
# --------------------------------------------------------------------------------------


def test_main_refuses_with_no_arguments(capsys):
    rc = hrl.main([])
    assert rc != 0
    assert "ERROR" in capsys.readouterr().out


def test_main_refuses_with_fewer_than_three_workbooks(tmp_path, capsys):
    rc = hrl.main(["--workbook", str(tmp_path / "r1.xlsx"), "--key", str(tmp_path / "k.csv")])
    assert rc != 0
    assert "ERROR" in capsys.readouterr().out


def test_score_v6_against_labels_excludes_no_majority_items(tmp_path, monkeypatch):
    monkeypatch.setattr(hrl, "V6_DIR", tmp_path)
    human_labels = {"items": [
        {"item_id": "c1", "set": "constructed", "majority_label": None},
        {"item_id": "c2", "set": "constructed", "majority_label": "verified"},
    ]}
    _write_jsonl(tmp_path / "hss_runA.jsonl", [
        {"item_id": "c1", "predicted_status": "unsupported"},
        {"item_id": "c2", "predicted_status": "verified"},
    ])
    _write_jsonl(tmp_path / "hss_runB.jsonl", [
        {"item_id": "c1", "predicted_status": "unsupported"},
        {"item_id": "c2", "predicted_status": "verified"},
    ])
    _write_jsonl(tmp_path / "real_runA.jsonl", [])
    _write_jsonl(tmp_path / "real_runB.jsonl", [])

    scores = hrl.score_v6_against_labels(human_labels)
    run_a = scores["by_set"]["constructed"]["run_A"]
    assert run_a["n"] == 1  # the no_majority item is excluded, not scored as wrong
    assert run_a["accuracy"] == 1.0

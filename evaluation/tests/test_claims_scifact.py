"""claims/fetch_scifact, claims/run_scifact and claims/baseline_lexical (no network, no LLM)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

import baseline_lexical as bl
import fetch_scifact as fs
import run_scifact as rs
from common import SCIFACT_TO_STATUS, append_jsonl

CLAIMS = [
    {
        "id": 1,
        "claim": "Vitamin D reduces fracture risk in adults.",
        "evidence": {"10": [{"sentences": [0], "label": "SUPPORT"}]},
        "cited_doc_ids": [10, 20],
    },
    {
        "id": 2,
        "claim": "Coffee causes cancer.",
        "evidence": {"20": [{"sentences": [1], "label": "CONTRADICT"}]},
        "cited_doc_ids": [20],
    },
    {"id": 3, "claim": "Nothing to see here.", "evidence": {}, "cited_doc_ids": [30, 99]},
]
CORPUS = {
    10: {
        "doc_id": 10,
        "title": "Vitamin D",
        "abstract": ["Vitamin D reduces fracture risk in adults.", "  ", "It was a trial."],
    },
    20: {"doc_id": 20, "title": "Coffee", "abstract": ["Coffee was studied.", "No cancer."]},
    30: {"doc_id": 30, "title": "Empty", "abstract": []},
}
REAL_DEV = Path(__file__).resolve().parents[1] / "claims" / "data" / "data" / "claims_dev.jsonl"


def write_fixture_data(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    append_jsonl(folder / "claims_dev.jsonl", CLAIMS)
    append_jsonl(folder / "corpus.jsonl", list(CORPUS.values()))
    return folder


# ---------------------------------------------------------------- run_scifact


def test_build_items_one_per_pair_with_joined_chunk_and_gold(capsys):
    items = rs.build_items(CLAIMS, CORPUS)
    assert [it["item_id"] for it in items] == ["1:10", "1:20", "2:20", "3:30"]
    assert items[0]["chunks"] == ["Vitamin D reduces fracture risk in adults. It was a trial."]
    assert items[0]["gold"] == "SUPPORT" and items[1]["gold"] == "NOT_ENOUGH_INFO"
    assert items[2]["gold"] == "CONTRADICT" and items[3]["gold"] == "NOT_ENOUGH_INFO"
    assert items[3]["chunks"] == [] and items[0]["title"] == "Vitamin D"
    assert items[0]["authors"] is None and items[0]["claim_id"] == 1 and items[0]["doc_id"] == 10
    assert "1 cited documents missing" in capsys.readouterr().err
    assert [it["item_id"] for it in rs.build_items(CLAIMS, CORPUS, limit=1)] == ["1:10", "1:20"]


@pytest.mark.skipif(not REAL_DEV.exists(), reason="SciFact dev set not fetched")
def test_real_dev_set_yields_340_pairs():
    claims, corpus = rs.load_scifact(REAL_DEV.parent)
    items = rs.build_items(claims, corpus)
    assert len(claims) == 300 and len(items) == 340
    counts = {g: sum(1 for it in items if it["gold"] == g) for g in SCIFACT_TO_STATUS}
    assert counts == {"SUPPORT": 138, "CONTRADICT": 71, "NOT_ENOUGH_INFO": 131}
    assert all(len(it["chunks"]) == 1 for it in items)


TRAIN_CLAIMS = [
    {"id": 101, "claim": "A supports X.",
     "evidence": {"10": [{"sentences": [0], "label": "SUPPORT"}]}, "cited_doc_ids": [10]},
    {"id": 102, "claim": "B supports X too.",
     "evidence": {"20": [{"sentences": [0], "label": "SUPPORT"}]}, "cited_doc_ids": [20]},
    {"id": 103, "claim": "C contradicts X.",
     "evidence": {"10": [{"sentences": [0], "label": "CONTRADICT"}]}, "cited_doc_ids": [10]},
    {"id": 104, "claim": "D contradicts X too.",
     "evidence": {"20": [{"sentences": [0], "label": "CONTRADICT"}]}, "cited_doc_ids": [20]},
    {"id": 105, "claim": "E is NEI.", "evidence": {}, "cited_doc_ids": [10]},
    {"id": 106, "claim": "F is NEI too.", "evidence": {}, "cited_doc_ids": [20]},
]


def write_train_fixture_data(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    append_jsonl(folder / "claims_train.jsonl", TRAIN_CLAIMS)
    append_jsonl(folder / "corpus.jsonl", list(CORPUS.values()))
    return folder


def test_load_scifact_train_split_reads_claims_train(tmp_path: Path):
    data = write_train_fixture_data(tmp_path / "data")
    claims, corpus = rs.load_scifact(data, "train")
    assert len(claims) == 6 and claims[0]["id"] == 101
    assert set(corpus) == {10, 20, 30}


def test_stratified_train_sample_deterministic_and_stratified():
    items = (
        [{"item_id": f"s{i}", "claim_id": i, "doc_id": 100 + i, "gold": "SUPPORT"}
         for i in range(5)]
        + [{"item_id": f"c{i}", "claim_id": 10 + i, "doc_id": 200 + i, "gold": "CONTRADICT"}
           for i in range(5)]
        + [{"item_id": f"n{i}", "claim_id": 20 + i, "doc_id": 300 + i, "gold": "NOT_ENOUGH_INFO"}
           for i in range(5)]
    )
    ids = rs.stratified_train_sample(items, n_per_bucket=3, seed=42)
    assert len(ids) == 9 and len(set(ids)) == 9
    assert all(i.startswith("s") for i in ids[:3])
    assert all(i.startswith("c") for i in ids[3:6])
    assert all(i.startswith("n") for i in ids[6:9])
    assert rs.stratified_train_sample(items, n_per_bucket=3, seed=42) == ids  # deterministic
    assert rs.stratified_train_sample(items, n_per_bucket=3, seed=7) != ids  # seed-sensitive


def test_stratified_train_sample_raises_when_a_bucket_is_too_small():
    items = [{"item_id": "s0", "claim_id": 0, "doc_id": 1, "gold": "SUPPORT"}]
    with pytest.raises(ValueError):
        rs.stratified_train_sample(items, n_per_bucket=2, seed=1)


def test_run_scifact_cli_train_split_builds_and_reuses_frozen_sample(tmp_path: Path):
    data = write_train_fixture_data(tmp_path / "data")
    sample_file = tmp_path / "sample.json"
    results_a = tmp_path / "results_a"
    argv = ["--run", "A", "--dry-run", "--split", "train", "--data-dir", str(data),
            "--results-dir", str(results_a), "--sample-n", "1", "--sample-seed", "42",
            "--sample-file", str(sample_file)]
    assert rs.main(argv) == 0
    assert sample_file.exists()
    ids = json.loads(sample_file.read_text(encoding="utf-8"))
    assert len(ids) == 3  # one item per gold bucket
    rows_a = [json.loads(line)
              for line in (results_a / "scifact-train_runA.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    assert {r["item_id"] for r in rows_a} == set(ids)
    # a second run with only --sample-file (already present) reuses the frozen ids, ignoring
    # --sample-n/--sample-seed entirely (not even passed here)
    results_b = tmp_path / "results_b"
    argv2 = ["--run", "A", "--dry-run", "--split", "train", "--data-dir", str(data),
             "--results-dir", str(results_b), "--sample-file", str(sample_file)]
    assert rs.main(argv2) == 0
    rows_b = [json.loads(line)
              for line in (results_b / "scifact-train_runA.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    assert {r["item_id"] for r in rows_b} == set(ids)


def test_run_scifact_cli_sample_n_without_seed_is_a_readable_error(tmp_path: Path):
    data = write_train_fixture_data(tmp_path / "data")
    argv = ["--run", "A", "--dry-run", "--split", "train", "--data-dir", str(data),
            "--results-dir", str(tmp_path / "results"), "--sample-n", "1"]
    with pytest.raises(SystemExit):
        rs.main(argv)


def test_run_scifact_cli_help_and_dry_run(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        rs.main(["--help"])
    assert exc.value.code == 0
    data = write_fixture_data(tmp_path / "data")
    results = tmp_path / "results"
    argv = ["--run", "A", "--dry-run", "--data-dir", str(data), "--results-dir", str(results)]
    assert rs.main(argv) == 0
    lines = (results / "scifact_runA.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    by_id = {r["item_id"]: r for r in rows}
    assert set(by_id) == {"1:10", "1:20", "2:20", "3:30"}
    assert by_id["1:10"]["predicted_status"] == "verified" and by_id["1:10"]["quote_is_verbatim"]
    assert by_id["3:30"]["predicted_status"] == "no_full_text" and by_id["3:30"]["deterministic"]
    meta = json.loads((results / "scifact_runA.meta.json").read_text(encoding="utf-8"))
    assert meta["dry_run"] is True and meta["n_rows"] == 4 and meta["n_deterministic"] == 1


# ---------------------------------------------------------------- baseline_lexical


def test_evaluate_lexical_on_fixture():
    items = rs.build_items(CLAIMS, CORPUS)
    result = bl.evaluate_lexical(items, 0.5)
    assert result["n"] == 4
    conf = result["confusion_gold_raw_x_predicted"]
    assert set(conf) == {"SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"}
    assert set(conf["SUPPORT"]) == {"verified", "unsupported"}
    assert conf["SUPPORT"]["verified"] == 1 and conf["CONTRADICT"]["unsupported"] == 1
    assert result["per_class"]["verified"]["precision"] == 1.0
    assert result["per_class"]["verified"]["recall"] == 1.0
    assert result["accuracy"] == 1.0 and result["predicted_counts"]["verified"] == 1


def test_baseline_match_run_filters_items(tmp_path: Path):
    data = write_fixture_data(tmp_path / "data")
    run_file = tmp_path / "scifact_runA.jsonl"
    append_jsonl(run_file, [{"item_id": "1:10"}, {"item_id": "2:20"}])
    with pytest.raises(SystemExit):
        bl.main(["--help"])
    argv = ["--data-dir", str(data), "--results-dir", str(tmp_path), "--match-run", str(run_file)]
    assert bl.main(argv) == 0
    out = json.loads((tmp_path / "scifact_baseline.json").read_text(encoding="utf-8"))
    assert out["n"] == 2 and out["matched_run"] == "scifact_runA.jsonl"


# ---------------------------------------------------------------- fetch_scifact


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _lines(n: int) -> bytes:
    return b"".join(b'{"id": %d}\n' % i for i in range(n))


def test_safe_members_rejects_traversal_and_absolute(tmp_path: Path):
    path = tmp_path / "t.tar.gz"
    _write_tar(path, {"data/a.jsonl": b"{}\n", "../x": b"", "/abs": b"", "data/../../y": b""})
    with tarfile.open(path) as tar:
        names = [m.name for m in fs.safe_members(tar)]
    assert names == ["data/a.jsonl"]


def test_verify_layout_and_extract(tmp_path: Path):
    tar_path = tmp_path / "scifact_data.tar.gz"
    _write_tar(
        tar_path,
        {
            "data/claims_dev.jsonl": _lines(3),
            "data/claims_train.jsonl": _lines(2),
            "data/claims_test.jsonl": _lines(1),
            "data/corpus.jsonl": _lines(4),
            "._data": b"apple",
        },
    )
    fs.extract(tar_path, tmp_path)
    assert (tmp_path / "data" / "corpus.jsonl").exists()
    counts = fs.verify_layout(tmp_path / "data")
    assert counts == {"claims_dev": 3, "claims_train": 2, "claims_test": 1, "corpus": 4}
    assert fs.count_lines(tmp_path / "data" / "corpus.jsonl") == 4
    with pytest.raises(SystemExit):
        fs.check_counts(counts)
    with pytest.raises(SystemExit):
        fs.verify_layout(tmp_path / "missing")


def test_fetch_main_skips_download_when_present(tmp_path: Path, monkeypatch):
    def no_download(*a, **k):
        raise AssertionError("download must not happen")

    monkeypatch.setattr(fs, "download", no_download)
    with pytest.raises(SystemExit) as exc:
        fs.main(["--help"])
    assert exc.value.code == 0
    tar_path = tmp_path / "scifact_data.tar.gz"
    tar_path.write_bytes(b"x" * 10)
    zero = {"claims_dev": 0, "claims_train": 0, "claims_test": 0, "corpus": 0}
    monkeypatch.setattr(fs, "EXPECTED_BYTES", 10)
    monkeypatch.setattr(fs, "EXPECTED_COUNTS", zero)
    monkeypatch.setattr(fs, "extract", lambda *a, **k: None)
    monkeypatch.setattr(fs, "verify_layout", lambda d: dict(zero))
    monkeypatch.setattr(fs, "dev_gold_pairs", lambda d: dict(fs.EXPECTED_GOLD_PAIRS))
    assert fs.main(["--out-dir", str(tmp_path)]) == 0
    info = json.loads((tmp_path / "scifact.fetch.json").read_text(encoding="utf-8"))
    assert info["bytes"] == 10 and info["citation_doi"] == "10.18653/v1/2020.emnlp-main.609"
    assert info["license"].startswith("claims CC BY 4.0") and len(info["sha256"]) == 64
    assert info["url"] == fs.DEFAULT_URL and info["counts"] == zero
    assert info["dev_gold_pairs"] == {"SUPPORT": 138, "CONTRADICT": 71, "NOT_ENOUGH_INFO": 131}
    assert info["fetched_at"]


# ---------------------------------------------------------------- additional cases


def test_evaluate_lexical_hss_per_rule(tmp_path: Path):
    src = ("The learners in the treatment group improved their writing scores substantially "
           "over the semester.")
    items = [
        {"item_id": "hss-verbatim-01", "rule": "verbatim", "expected": ["verified"],
         "claim": src, "chunks": [f"Intro text here. {src} More text follows."]},
        {"item_id": "hss-altered-01", "rule": "altered",
         "expected": ["unsupported", "needs_nuance"],
         "claim": src.replace("improved", "worsened"), "chunks": [src]},
        {"item_id": "hss-wrong_paper-01", "rule": "wrong_paper", "expected": ["unsupported"],
         "claim": src, "chunks": ["An unrelated paper about soil chemistry and nitrogen."]},
        {"item_id": "hss-no_full_text-01", "rule": "no_full_text",
         "expected": ["no_full_text"], "claim": src, "chunks": []},
    ]
    out = bl.evaluate_lexical_hss(items, 0.5)
    assert out["n"] == 4 and out["threshold"] == 0.5
    assert out["by_rule"]["verbatim"]["correct"] == 1
    assert out["by_rule"]["altered"]["correct"] == 0  # single-word change: Jaccard ~0.9
    assert out["by_rule"]["altered"]["predicted_counts"] == {"verified": 1}
    assert out["by_rule"]["wrong_paper"]["correct"] == 1
    assert out["by_rule"]["no_full_text"]["correct"] == 1  # deterministic path, no lexical call
    assert out["accuracy"] == pytest.approx(3 / 4)
    vb = out["verified_binary"]
    assert vb["tp"] == 1 and vb["fp"] == 1 and vb["recall"] == 1.0
    # the no-chunk item is a by-construction rejection, not a true negative
    assert vb["n"] == 3 and vb["tn"] == 1 and vb["n_excluded_deterministic"] == 1
    assert out["n"] == 4
    assert out["items"][1]["max_jaccard"] > 0.8
    args = bl.parse_args(["--set", "hss", "--claims", "c.jsonl", "--cache-dir", "d"])
    assert args.set == "hss" and args.claims.is_absolute()
    assert bl.parse_args([]).set == "scifact"

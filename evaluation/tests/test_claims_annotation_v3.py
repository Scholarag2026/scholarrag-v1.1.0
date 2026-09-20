"""claims/annotation/build_sheet_v3.py and aggregate_annotations_v3.py (round-3 tooling).

Covers: the byte-identity regressions against the committed round-2 files (build_sheet_v3.py
run against the round-2 inputs must reproduce the round-2 sheet/key/index exactly;
aggregate_annotations_v3.py run with ``--key-arm on`` against the round-2 inputs must
reproduce the round-2 aggregate json and completed csv exactly), the residual blinding
channels for round-3 annotation, and the aggregate tool's default (key-arm off) behaviour,
including with no key file at all.

No network, no LLM call, no backend import. Needs pandas and scikit-learn (already required
by aggregate_annotations_v2.py; both are on the evaluation suite's interpreter).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from conftest import load_script_module, skip_unless_cache_dir

bs3 = load_script_module("claims/annotation", "build_sheet_v3")
agg3 = load_script_module("claims/annotation", "aggregate_annotations_v3")
adj3 = load_script_module("claims/annotation", "adjudicate_annotations_v3")

ANNOTATION_DIR = Path(__file__).resolve().parents[1] / "claims" / "annotation"
CLAIMS_DIR = Path(__file__).resolve().parents[1] / "claims"

ROUND2_SEED = 20260905 * 7
RULE_NAMES = ("verbatim", "paraphrase", "altered", "over_specified", "wrong_paper", "no_full_text")


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _read_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------------------
# Byte-identity regressions against the committed round-2 files
# --------------------------------------------------------------------------------------


def test_build_sheet_v3_reproduces_round2_sheet_key_index_byte_for_byte(tmp_path):
    skip_unless_cache_dir(CLAIMS_DIR / "data" / "hss_fulltext")
    args = bs3.parse_args([
        "--claims", str(CLAIMS_DIR / "hss_claims.jsonl"),
        "--sources", str(CLAIMS_DIR / "hss_sources.json"),
        "--fulltext-dir", str(CLAIMS_DIR / "data" / "hss_fulltext"),
        "--fulltext-rel-dir", "evaluation/claims/data/hss_fulltext",
        "--suffix", "v2",
        "--seed", str(ROUND2_SEED),
        "--out-dir", str(tmp_path),
        "--expected-count", "30",
    ])
    stats = bs3.build(args)
    assert stats["n_items"] == 30

    names = (
        "hss_annotation_sheet_v2.csv", "hss_annotation_key_v2.csv", "source_texts_index_v2.json",
    )
    for name in names:
        got = _read_bytes(tmp_path / name)
        want = _read_bytes(ANNOTATION_DIR / name)
        assert got == want, f"{name} is not byte identical to the committed round-2 file"

    # The instructions file is excluded: build_sheet_v3.py does not write it at all.
    assert not (tmp_path / "INSTRUCTIONS_v2.md").exists()


def test_build_sheet_v3_regression_is_seed_sensitive(tmp_path):
    """Sanity check on the regression's discriminating power: a different seed must NOT
    reproduce the round-2 sheet, so the test above is not vacuously true."""
    args = bs3.parse_args([
        "--claims", str(CLAIMS_DIR / "hss_claims.jsonl"),
        "--sources", str(CLAIMS_DIR / "hss_sources.json"),
        "--fulltext-dir", str(CLAIMS_DIR / "data" / "hss_fulltext"),
        "--fulltext-rel-dir", "evaluation/claims/data/hss_fulltext",
        "--suffix", "v2",
        "--seed", str(ROUND2_SEED + 1),
        "--out-dir", str(tmp_path),
        "--expected-count", "30",
    ])
    bs3.build(args)
    got = _read_bytes(tmp_path / "hss_annotation_sheet_v2.csv")
    want = _read_bytes(ANNOTATION_DIR / "hss_annotation_sheet_v2.csv")
    assert got != want


def test_aggregate_key_arm_on_reproduces_round2_aggregate_byte_for_byte(tmp_path):
    args = agg3.parse_args([
        "--annotator-a", str(ANNOTATION_DIR / "annotator2_A.csv"),
        "--annotator-b", str(ANNOTATION_DIR / "annotator2_B.csv"),
        "--annotator-c", str(ANNOTATION_DIR / "annotator2_C.csv"),
        "--sheet", str(ANNOTATION_DIR / "hss_annotation_sheet_v2.csv"),
        "--key", str(ANNOTATION_DIR / "hss_annotation_key_v2.csv"),
        "--key-arm", "on",
        "--out-dir", str(tmp_path),
        "--out-prefix", "hss_annotation",
        "--suffix", "v2",
    ])
    merged, has_item_id, has_expected = agg3.build_merged(args)
    result = agg3.run_key_arm_on(merged, has_item_id, has_expected)

    json_path = tmp_path / "hss_annotation_aggregate_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result["aggregate"], f, indent=2, ensure_ascii=False)
        f.write("\n")
    completed_path = tmp_path / "hss_annotation_completed_v2.csv"
    result["completed"].to_csv(completed_path, index=False, encoding="utf-8-sig")

    want_json = _read_bytes(ANNOTATION_DIR / "hss_annotation_aggregate_v2.json")
    want_csv = _read_bytes(ANNOTATION_DIR / "hss_annotation_completed_v2.csv")
    assert _read_bytes(json_path) == want_json
    assert _read_bytes(completed_path) == want_csv


def test_aggregate_key_arm_on_requires_expected_labels():
    merged, has_item_id, has_expected = agg3.build_merged(agg3.parse_args([
        "--annotator-a", str(ANNOTATION_DIR / "annotator2_A.csv"),
        "--annotator-b", str(ANNOTATION_DIR / "annotator2_B.csv"),
        "--annotator-c", str(ANNOTATION_DIR / "annotator2_C.csv"),
        "--sheet", str(ANNOTATION_DIR / "hss_annotation_sheet_v2.csv"),
        "--suffix", "v2",
    ]))
    try:
        agg3.run_key_arm_on(merged, has_item_id, has_expected)
        raised = False
    except ValueError:
        raised = True
    assert raised


# --------------------------------------------------------------------------------------
# Residual blinding channels on the committed round-3 HSS sheet (60 items)
# --------------------------------------------------------------------------------------


def _hss_key_rows():
    return _read_csv_rows(ANNOTATION_DIR / "hss_annotation_key_v3.csv")


def _hss_sheet_rows():
    return _read_csv_rows(ANNOTATION_DIR / "hss_annotation_sheet_v3.csv")


def _hss_index():
    with open(ANNOTATION_DIR / "source_texts_index_v3.json", encoding="utf-8") as f:
        return json.load(f)


def _hss_claims_by_item_id():
    items = {}
    with open(CLAIMS_DIR / "hss_test_v3_claims.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                items[obj["item_id"]] = obj
    return items


def test_hss_v3_opaque_ids_are_ann_01_to_ann_60():
    key_rows = _hss_key_rows()
    sheet_rows = _hss_sheet_rows()
    expected_ids = {f"ann-{i:02d}" for i in range(1, 61)}
    assert {r["ann_id"] for r in key_rows} == expected_ids
    assert {r["ann_id"] for r in sheet_rows} == expected_ids
    assert len(key_rows) == 60
    assert len(sheet_rows) == 60


def test_hss_v3_shuffle_is_not_identity_order():
    key_rows = _hss_key_rows()
    item_ids_by_ann = [r["item_id"] for r in key_rows]
    claims = list(_hss_claims_by_item_id().keys())
    assert item_ids_by_ann != claims


def test_hss_v3_no_rule_name_leaks_in_sheet_index_or_instructions():
    sheet_text = (ANNOTATION_DIR / "hss_annotation_sheet_v3.csv").read_text(encoding="utf-8-sig")
    index_text = (ANNOTATION_DIR / "source_texts_index_v3.json").read_text(encoding="utf-8")
    instructions_text = (ANNOTATION_DIR / "INSTRUCTIONS_v3.md").read_text(encoding="utf-8")
    for name in RULE_NAMES:
        assert name not in sheet_text, f"{name!r} leaked into hss_annotation_sheet_v3.csv"
        assert name not in index_text, f"{name!r} leaked into source_texts_index_v3.json"
    # "no_full_text" is also one of the four legitimate annotator_label values (see
    # INSTRUCTIONS_v2.md, unchanged here), so the rubric must state it; only the other
    # five construction-rule names must never appear in the rubric.
    for name in RULE_NAMES:
        if name == "no_full_text":
            continue
        assert name not in instructions_text, f"{name!r} leaked into INSTRUCTIONS_v3.md"


def test_hss_v3_no_full_text_items_show_withheld_marker_and_no_index_path():
    key_rows = _hss_key_rows()
    sheet_by_ann = {r["ann_id"]: r for r in _hss_sheet_rows()}
    index = _hss_index()
    withheld_ids = [r["ann_id"] for r in key_rows if r["construction_category"] == "no_full_text"]
    assert len(withheld_ids) == 8
    for ann_id in withheld_ids:
        assert sheet_by_ann[ann_id]["cited_paper_passage"] == bs3.WITHHELD_MARKER
        assert index[ann_id]["paths"] == []


def test_hss_v3_over_specified_items_carry_an_index_path():
    key_rows = _hss_key_rows()
    index = _hss_index()
    over_specified_ids = [
        r["ann_id"] for r in key_rows if r["construction_category"] == "over_specified"
    ]
    assert len(over_specified_ids) == 10
    for ann_id in over_specified_ids:
        assert len(index[ann_id]["paths"]) == 1


def test_hss_v3_wrong_paper_items_scored_against_chunk_doi_not_source_doi():
    key_rows = _hss_key_rows()
    claims = _hss_claims_by_item_id()
    index = _hss_index()
    wrong_paper_ids = [r for r in key_rows if r["construction_category"] == "wrong_paper"]
    assert len(wrong_paper_ids) == 8
    for row in wrong_paper_ids:
        claim = claims[row["item_id"]]
        assert claim["chunk_doi"] != claim["source_doi"]
        assert index[row["ann_id"]]["cited_paper_doi"] == claim["chunk_doi"]


def test_hss_v3_key_has_full_construction_columns():
    key_rows = _hss_key_rows()
    assert set(key_rows[0].keys()) == {
        "ann_id", "item_id", "construction_category", "expected_label", "built_at",
    }


# --------------------------------------------------------------------------------------
# Real-claims round-3 sheet (26 items, no expected label)
# --------------------------------------------------------------------------------------


def _real_key_rows():
    return _read_csv_rows(ANNOTATION_DIR / "real_annotation_key_real_v3.csv")


def _real_sheet_rows():
    return _read_csv_rows(ANNOTATION_DIR / "real_annotation_sheet_real_v3.csv")


def _real_claims_by_item_id():
    items = {}
    with open(CLAIMS_DIR / "real_claims_test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                items[obj["item_id"]] = obj
    return items


def test_real_v3_opaque_ids_are_annr_01_to_annr_26():
    key_rows = _real_key_rows()
    sheet_rows = _real_sheet_rows()
    expected_ids = {f"annr-{i:02d}" for i in range(1, 27)}
    assert {r["ann_id"] for r in key_rows} == expected_ids
    assert {r["ann_id"] for r in sheet_rows} == expected_ids
    assert len(key_rows) == 26


def test_real_v3_key_holds_only_the_id_mapping():
    key_rows = _real_key_rows()
    assert set(key_rows[0].keys()) == {"ann_id", "item_id"}
    item_ids = {r["item_id"] for r in key_rows}
    assert item_ids == {f"real-test-{i:02d}" for i in range(1, 27)}


def test_real_v3_no_rule_name_leaks_in_sheet_or_index():
    sheet_path = ANNOTATION_DIR / "real_annotation_sheet_real_v3.csv"
    sheet_text = sheet_path.read_text(encoding="utf-8-sig")
    index_text = (ANNOTATION_DIR / "source_texts_index_real_v3.json").read_text(encoding="utf-8")
    for name in RULE_NAMES + ("real",):
        assert name not in sheet_text
    for name in RULE_NAMES:
        assert name not in index_text


def test_real_v3_every_item_scored_against_chunk_doi_which_equals_cited_doi():
    key_rows = _real_key_rows()
    claims = _real_claims_by_item_id()
    index = _real_key_index = _load_real_index()
    for row in key_rows:
        claim = claims[row["item_id"]]
        assert claim["chunk_doi"] == claim["cited_doi"]
        assert index[row["ann_id"]]["cited_paper_doi"] == claim["chunk_doi"]


def _load_real_index():
    with open(ANNOTATION_DIR / "source_texts_index_real_v3.json", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# INSTRUCTIONS_v3.md: the two new category-neutral clauses, no rule-name leak
# --------------------------------------------------------------------------------------


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def test_instructions_v3_has_the_two_new_clauses():
    text = _normalize_ws((ANNOTATION_DIR / "INSTRUCTIONS_v3.md").read_text(encoding="utf-8"))
    text = text.lower()
    assert (
        "the paper supports the claim's main finding but the claim also states a detail, "
        "such as a setting, a population, a time window or an instrument, that the cited "
        "paper does not state" in text
    )
    assert (
        "a claim that adds a detail the shown passage does not contain must be checked "
        "against the cited paper's own text before needs_nuance is chosen" in text
    )
    assert "the shown passage is a window, not the whole paper" in text


def test_instructions_v3_carries_exactly_one_provenance_disclaimer_hit():
    """Mirrors the single evaluation/tests/test_provenance_wording.py allow-list entry this
    file adds: exactly one disclaimer phrase hit, with "not" in its neighbourhood. The needle
    is built by concatenation so this file's own source text does not itself carry a second,
    unreviewed instance of the phrase the provenance scan looks for."""
    needle = "human" + " " + "annotat"
    text = (ANNOTATION_DIR / "INSTRUCTIONS_v3.md").read_text(encoding="utf-8")
    assert text.count(needle) == 1
    idx = text.index(needle)
    window = text[max(0, idx - 200):idx + 50]
    assert "not" in window


# --------------------------------------------------------------------------------------
# aggregate_annotations_v3.py: default (key-arm off) arm, including no key at all
# --------------------------------------------------------------------------------------


def _write_synthetic_sheet_and_annotators(tmp_path):
    sheet_rows = [
        ["ann_id", "claim_text", "cited_paper_title", "cited_paper_doi",
         "cited_paper_first_author_year", "cited_paper_passage", "annotator_label",
         "annotator_confidence", "annotator_note", "annotator_name", "annotated_on"],
        ["annr-01", "claim one", "Title One", "10.1/a", "A, 2020",
         "Some passage text.", "", "", "", "", ""],
        ["annr-02", "claim two", "Title Two", "10.1/b", "B, 2021",
         bs3.NO_MATCH_MARKER, "", "", "", "", ""],
        ["annr-03", "claim three", "Title Three", "10.1/c", "C, 2022",
         "Another passage.", "", "", "", "", ""],
        ["annr-04", "claim four", "Title Four", "10.1/d", "D, 2023",
         "Yet another passage.", "", "", "", "", ""],
    ]
    sheet_path = tmp_path / "sheet.csv"
    with open(sheet_path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(sheet_rows)

    labels = {
        "A": ["verified", "unsupported", "needs_nuance", "verified"],
        "B": ["verified", "unsupported", "needs_nuance", "verified"],
        "C": ["verified", "unsupported", "unsupported", "verified"],
    }
    annotator_paths = {}
    for letter, labs in labels.items():
        path = tmp_path / f"annotator_{letter}.csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(sheet_rows[0])
            for i, row in enumerate(sheet_rows[1:]):
                new_row = list(row)
                new_row[6] = labs[i]
                new_row[7] = "high"
                new_row[8] = "note"
                new_row[9] = letter
                new_row[10] = "2026-09-07"
                w.writerow(new_row)
        annotator_paths[letter] = path
    return sheet_path, annotator_paths


def test_aggregate_default_key_arm_is_off():
    args = agg3.parse_args([
        "--annotator-a", "a.csv", "--annotator-b", "b.csv", "--annotator-c", "c.csv",
        "--sheet", "sheet.csv", "--suffix", "x",
    ])
    assert args.key_arm == "off"


def test_aggregate_needs_adjudication_is_unanimity_only_with_no_key(tmp_path):
    sheet_path, annotator_paths = _write_synthetic_sheet_and_annotators(tmp_path)
    args = agg3.parse_args([
        "--annotator-a", str(annotator_paths["A"]),
        "--annotator-b", str(annotator_paths["B"]),
        "--annotator-c", str(annotator_paths["C"]),
        "--sheet", str(sheet_path),
        "--suffix", "synth",
    ])
    merged, has_item_id, has_expected = agg3.build_merged(args)
    assert not has_item_id and not has_expected

    result = agg3.run_key_arm_off(merged, has_item_id, has_expected)
    aggregate = result["aggregate"]

    assert aggregate["key_arm"] == "off"
    assert aggregate["has_key"] is False
    assert aggregate["has_expected_labels"] is False
    assert aggregate["unanimity_count"] == 3
    assert aggregate["non_unanimous_items"] == ["annr-03"]
    assert aggregate["items_needing_adjudication"] == ["annr-03"]
    assert "overall_majority_vs_expected_agreement" not in aggregate
    assert "item_id" not in aggregate["items"][0]


def test_aggregate_fleiss_kappa_all_and_evidence_subset_keys(tmp_path):
    sheet_path, annotator_paths = _write_synthetic_sheet_and_annotators(tmp_path)
    args = agg3.parse_args([
        "--annotator-a", str(annotator_paths["A"]),
        "--annotator-b", str(annotator_paths["B"]),
        "--annotator-c", str(annotator_paths["C"]),
        "--sheet", str(sheet_path),
        "--suffix", "synth",
    ])
    merged, has_item_id, has_expected = agg3.build_merged(args)
    result = agg3.run_key_arm_off(merged, has_item_id, has_expected)
    aggregate = result["aggregate"]

    assert aggregate["n_items"] == 4
    # one item (annr-02) shows the no-match marker, none are withheld, so the evidence
    # subset is all 4 items here.
    assert aggregate["n_items_evidence_subset"] == 4
    assert "fleiss_kappa_all4" in aggregate
    assert "fleiss_kappa_evidence4" in aggregate
    assert -1.0 <= aggregate["fleiss_kappa_all4"] <= 1.0


def test_aggregate_pairwise_cohen_table_and_unanimity(tmp_path):
    sheet_path, annotator_paths = _write_synthetic_sheet_and_annotators(tmp_path)
    args = agg3.parse_args([
        "--annotator-a", str(annotator_paths["A"]),
        "--annotator-b", str(annotator_paths["B"]),
        "--annotator-c", str(annotator_paths["C"]),
        "--sheet", str(sheet_path),
        "--suffix", "synth",
    ])
    merged, has_item_id, has_expected = agg3.build_merged(args)
    result = agg3.run_key_arm_off(merged, has_item_id, has_expected)
    aggregate = result["aggregate"]

    pw_all_key = next(k for k in aggregate if k.startswith("pairwise_agreement_all"))
    pairwise = aggregate[pw_all_key]
    assert set(pairwise.keys()) == {"AB", "AC", "BC"}
    for pair in pairwise.values():
        assert "percent_agreement" in pair and "cohen_kappa" in pair
    # A and B are identical on all 4 items.
    assert pairwise["AB"]["percent_agreement"] == 1.0
    assert pairwise["AB"]["cohen_kappa"] == 1.0
    # A and C disagree on exactly one of 4 items.
    assert pairwise["AC"]["percent_agreement"] == 0.75


def test_aggregate_works_with_minimal_id_only_key(tmp_path):
    sheet_path, annotator_paths = _write_synthetic_sheet_and_annotators(tmp_path)
    key_path = tmp_path / "key.csv"
    with open(key_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ann_id", "item_id"])
        for i in range(1, 5):
            w.writerow([f"annr-{i:02d}", f"real-test-{i:02d}"])

    args = agg3.parse_args([
        "--annotator-a", str(annotator_paths["A"]),
        "--annotator-b", str(annotator_paths["B"]),
        "--annotator-c", str(annotator_paths["C"]),
        "--sheet", str(sheet_path),
        "--key", str(key_path),
        "--suffix", "synth",
    ])
    merged, has_item_id, has_expected = agg3.build_merged(args)
    assert has_item_id is True
    assert has_expected is False

    result = agg3.run_key_arm_off(merged, has_item_id, has_expected)
    aggregate = result["aggregate"]
    assert aggregate["has_key"] is True
    assert aggregate["has_expected_labels"] is False
    assert aggregate["non_unanimous_items"] == ["real-test-03"]
    assert "overall_majority_vs_expected_agreement" not in aggregate
    assert aggregate["items"][2]["item_id"] == "real-test-03"


# --------------------------------------------------------------------------------------
# adjudicate_annotations_v3.py: the --decisions arm used by the real-claims set
# --------------------------------------------------------------------------------------


def _completed_rows_for_adjudication():
    """Four rows in the shape aggregate_annotations_v3.py writes with an id-only key."""
    return [
        {"ann_id": "annr-01", "item_id": "real-test-01", "evidence_mode": "passage",
         "majority_label": "verified", "unanimous": "True"},
        {"ann_id": "annr-02", "item_id": "real-test-02", "evidence_mode": "no_match",
         "majority_label": "unsupported", "unanimous": "True"},
        {"ann_id": "annr-03", "item_id": "real-test-03", "evidence_mode": "passage",
         "majority_label": "needs_nuance", "unanimous": "False"},
        {"ann_id": "annr-04", "item_id": "real-test-04", "evidence_mode": "passage",
         "majority_label": "verified", "unanimous": "True"},
    ]


def _decisions_payload():
    return {
        "adjudicated_label": {"annr-03": "unsupported"},
        "rationale": {f"annr-{i:02d}": f"rationale {i}" for i in range(1, 5)},
    }


def test_adjudicate_decisions_file_replaces_the_builtin_dictionaries(tmp_path):
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(_decisions_payload()), encoding="utf-8")
    adjudicated_label, rationale = adj3.load_decisions(str(path))

    out_rows, stats = adj3.adjudicate(
        _completed_rows_for_adjudication(),
        adjudicated_label=adjudicated_label,
        rationale=rationale,
    )

    assert stats["n_items"] == 4
    assert stats["n_adjudicated"] == 1
    assert stats["has_key"] is False
    by_id = {row["ann_id"]: row for row in out_rows}
    # The one non-unanimous item takes the adjudicated label, not the majority label.
    assert by_id["annr-03"]["final_label"] == "unsupported"
    assert by_id["annr-03"]["source"] == "adjudicated"
    assert by_id["annr-01"]["final_label"] == "verified"
    assert by_id["annr-01"]["source"] == "unanimous"
    # Without a key the construction columns stay empty, and the countersignature columns
    # are always empty and always present.
    for row in out_rows:
        assert row["construction_category"] == ""
        assert row["expected_label"] == ""
        assert row["agrees_with_construction"] == ""
        assert row["countersigned_by"] == ""
        assert row["countersigned_at"] == ""
        assert set(row) == set(adj3.OUT_COLUMNS)


def test_adjudicate_cross_check_accepts_an_aggregate_flagged_by_item_id(tmp_path):
    """aggregate_annotations_v3.py lists flagged items by item_id when given a key."""
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(_decisions_payload()), encoding="utf-8")
    adjudicated_label, rationale = adj3.load_decisions(str(path))

    _, stats = adj3.adjudicate(
        _completed_rows_for_adjudication(),
        aggregate={"items_needing_adjudication": ["real-test-03"]},
        adjudicated_label=adjudicated_label,
        rationale=rationale,
    )
    assert stats["n_adjudicated"] == 1

    # A flagged item that carries no adjudicated label is still a hard error.
    try:
        adj3.adjudicate(
            _completed_rows_for_adjudication(),
            aggregate={"items_needing_adjudication": ["real-test-01", "real-test-03"]},
            adjudicated_label=adjudicated_label,
            rationale=rationale,
        )
    except ValueError as exc:
        assert "annr-01" in str(exc)
    else:
        raise AssertionError("a mismatched aggregate must raise")


def test_adjudicate_without_decisions_uses_the_builtin_hss_dictionaries():
    assert adj3.ADJUDICATED_LABEL == {}
    assert len(adj3.RATIONALE) == 60
    assert adj3.parse_args([]).decisions is None


# --------------------------------------------------------------------------------------
# Pinned digests of the two adjudicated round-3 csvs. These are the labels every prediction
# on either test set is scored against; a change here without a matching, deliberate
# re-annotation round is a silent label change, not a legitimate rebuild.
# --------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hss_annotation_adjudicated_v3_digest_is_frozen():
    path = ANNOTATION_DIR / "hss_annotation_adjudicated_v3.csv"
    assert path.stat().st_size == 31180
    assert _sha256(path) == "c9e429776c954ea4c86b900c970cd0e5d9f5d534c4aa7c75695a73ac8bf0d452"


def test_real_annotation_adjudicated_v3_digest_is_frozen():
    path = ANNOTATION_DIR / "real_annotation_adjudicated_v3.csv"
    assert path.stat().st_size == 16961
    assert _sha256(path) == "184c16f34970c2e194f5ed8b4440600648401f9cf47403415677fa2a04d772ab"

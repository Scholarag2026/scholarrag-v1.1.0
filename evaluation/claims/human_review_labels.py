"""Human-label majority verdicts, inter-rater agreement and v6 scoring. Reads the three
completed human-review workbooks and the workbook key, all read-only (`openpyxl`,
`read_only=True`); writes nothing back to any of them and never copies the key. No reviewer
identity is read from or written to any output of this script: the three workbooks are
joined only by the order their `--workbook` flags are given (reviewer 1, 2, 3).

Inputs, given on the command line (never hard-coded; both outside `evaluation/claims/` and
never committed):

* ``--workbook`` (repeat exactly three times, in reviewer order) -- each reviewer's own
  filled Review-and-Screening workbook. This script reads only the ``Review`` sheet (86
  items: 60 constructed, 26 real, one row per claim item judged against the stored source
  text). The three workbooks are local, gitignored files the author holds; this script never
  names them. The ``Delivered`` sheet of a promoted demo run is scored separately by
  ``delivered_human_scores.py``, the tool of record for that scoring.
* ``--key`` -- the workbook key CSV: the one-way join from a workbook's opaque ``item_id``
  (e.g. ``R-000``) back to the evaluation's own ``item_id`` (e.g. ``item-000``) and its
  ``set``. Read-only; nothing from it beyond that join is ever written anywhere.

Majority rule: the label of record for an item is the status at least two of the three
reviewers returned; a three-way split is `no_majority`, excluded from every label-based
metric and reported as a count (zero on this data). Inter-rater agreement is reported two
ways: the mean of the three pairwise percent-agreement rates, and Fleiss' kappa (equal to
three decimals against the mean pairwise Cohen's kappa on this data).

Outputs, under ``evaluation/claims/results/v6/``:

* ``human_labels.json`` -- one entry per Review item: the evaluation's own item id, the
  human majority label, each reviewer's own verdict as ``reviewer_1``..``reviewer_3`` (no
  names), and whether the item is one of the eight protocol-fixed ``no_full_text`` rows.
  Nothing from the key beyond the ``item_id``/``set`` join.
* ``human_review_scores.json`` -- the v6 (`results/v6/{hss,real}_run{A,B}.jsonl`) verifier
  statuses scored against the human majority label: accuracy (with a 95% Wilson interval),
  Cohen's kappa, `verified` precision/recall, and the falsely-`verified` count, per set
  (constructed, real) and per run (A, B).

This script also rewrites ``results/v6/summary.json``/``summary.md`` (already written by
``freeze_v6.py``) so the human-majority scores are the primary label for the constructed and
real sets, with the existing ``hss_vs_annotation``/``real_vs_annotation`` blocks (scored
against the earlier model-authored/adjudicated labels) kept in place as a secondary
comparison row rather than removed.

Usage (the evaluation venv's python, for ``openpyxl``; no backend import, no network, no
LLM), from ``evaluation/claims/``::

    python human_review_labels.py --workbook R1.xlsx --workbook R2.xlsx \\
        --workbook R3.xlsx --key key.csv

A run with no ``--workbook``/``--key`` arguments refuses: this script has no default
workbook location.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import openpyxl

HERE = Path(__file__).resolve().parent  # evaluation/claims
EVAL_ROOT = HERE.parent
REPO_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    cohens_kappa,
    now_iso,
    per_class_prf,
    percent_agreement,
    read_json,
    read_jsonl,
    wilson_interval,
    write_json,
)

V6_DIR = HERE / "results" / "v6"

CANONICAL_STATUSES: tuple[str, ...] = ("verified", "needs_nuance", "unsupported", "no_full_text")
SET_TO_V6_PREFIX: dict[str, str] = {"constructed": "hss", "real": "real"}
RUNS: tuple[str, ...] = ("A", "B")


def _read_sheet(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    """Every non-blank row of *sheet_name*, read-only, as a header-keyed dict of stripped
    strings (blank cells become ``""``, never ``None``, so a missing field is unambiguous)."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[sheet_name]
        rows_iter = ws.iter_rows(values_only=True)
        header = [str(c) if c is not None else "" for c in next(rows_iter)]
        out = []
        for row in rows_iter:
            if all(c is None or str(c).strip() == "" for c in row):
                continue
            values = ("" if c is None else str(c).strip() for c in row)
            out.append(dict(zip(header, values, strict=True)))
        return out
    finally:
        wb.close()


def load_key(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {row["opaque_id"]: row for row in csv.DictReader(fh)}


def majority_label(verdicts: list[str]) -> str | None:
    """The status at least two of three reviewers returned, or ``None`` for a three-way
    split (`no_majority`)."""
    top, count = Counter(verdicts).most_common(1)[0]
    return top if count >= 2 else None


def fleiss_kappa(table: list[list[int]]) -> float | None:
    """Fleiss' kappa for a fixed number of raters per item, ``table[i][c]`` the count of
    raters who assigned item ``i`` to category ``c``. ``None`` when undefined (fewer than 2
    items or 0 raters)."""
    n_items = len(table)
    if n_items == 0:
        return None
    n_raters = sum(table[0])
    if n_raters < 2:
        return None
    n_cats = len(table[0])
    p_i = [(sum(x * x for x in row) - n_raters) / (n_raters * (n_raters - 1)) for row in table]
    p_bar = sum(p_i) / n_items
    p_j = [sum(row[c] for row in table) / (n_items * n_raters) for c in range(n_cats)]
    p_e = sum(p * p for p in p_j)
    if p_e >= 1.0:
        return None
    return (p_bar - p_e) / (1.0 - p_e)


def pairwise_agreement(cols: dict[int, list[str]]) -> dict[str, Any]:
    """Mean pairwise percent agreement and mean pairwise Cohen's kappa over the three
    reviewer columns (keys ``1``, ``2``, ``3``), reported alongside Fleiss' kappa (the task's
    "pairwise percent agreement and Fleiss kappa") so both readings are on record."""
    pct, kap = [], []
    for a, b in itertools.combinations((1, 2, 3), 2):
        pct.append(percent_agreement(cols[a], cols[b]))
        kap.append(cohens_kappa(cols[a], cols[b]))
    return {
        "mean_pairwise_percent_agreement": sum(pct) / len(pct),
        "pairwise_percent_agreement": [round(p, 4) for p in pct],
        "mean_pairwise_cohens_kappa": (
            (sum(kap) / len(kap)) if all(k is not None for k in kap) else None
        ),
    }


# --------------------------------------------------------------------------------------
# Review sheet: majority labels + inter-rater agreement
# --------------------------------------------------------------------------------------


def build_human_labels(workbook_paths: tuple[Path, Path, Path], key_path: Path) -> dict[str, Any]:
    key = load_key(key_path)
    review_by_reviewer = {
        i: _read_sheet(p, "Review") for i, p in enumerate(workbook_paths, start=1)
    }
    opaque_ids = [r["item_id"] for r in review_by_reviewer[1]]
    verdicts: dict[str, dict[int, str]] = {oid: {} for oid in opaque_ids}
    for reviewer, rows in review_by_reviewer.items():
        for row in rows:
            verdicts[row["item_id"]][reviewer] = row["human_verdict"]

    items: list[dict[str, Any]] = []
    no_majority_ids: list[str] = []
    for oid in opaque_ids:
        k = key[oid]
        v = verdicts[oid]
        m = majority_label([v[1], v[2], v[3]])
        if m is None:
            no_majority_ids.append(k["real_item_id"])
        items.append({
            "item_id": k["real_item_id"],
            "set": k["set"],
            "majority_label": m,
            "reviewer_1": v[1],
            "reviewer_2": v[2],
            "reviewer_3": v[3],
            "unanimous": len({v[1], v[2], v[3]}) == 1,
            "is_protocol_fixed_no_full_text": (
                k["expected_label"] == "no_full_text" and v[1] == v[2] == v[3] == "no_full_text"
            ),
        })

    def _agreement_over(id_subset: list[str]) -> dict[str, Any]:
        cols = {i: [verdicts[oid][i] for oid in id_subset] for i in (1, 2, 3)}
        table = [
            [sum(1 for i in (1, 2, 3) if verdicts[oid][i] == c) for c in CANONICAL_STATUSES]
            for oid in id_subset
        ]
        agr = pairwise_agreement(cols)
        agr["fleiss_kappa"] = fleiss_kappa(table)
        agr["n_items"] = len(id_subset)
        agr["n_unanimous"] = sum(
            1 for oid in id_subset if len({verdicts[oid][i] for i in (1, 2, 3)}) == 1
        )
        return agr

    fixed_opaque_ids = {
        oid for oid in opaque_ids
        if key[oid]["expected_label"] == "no_full_text"
        and verdicts[oid][1] == verdicts[oid][2] == verdicts[oid][3] == "no_full_text"
    }
    judged_ids = [oid for oid in opaque_ids if oid not in fixed_opaque_ids]

    # Report one number per sheet, computed on the 78 freely judged items (the 86 minus the
    # eight protocol-fixed no_full_text rows, on which every reviewer trivially agrees by
    # construction and which would otherwise inflate agreement without testing anything).
    # `with_withheld` (all 86) is kept alongside for the record.
    agreement = _agreement_over(judged_ids)
    agreement["population"] = "judged_only (86 minus the 8 protocol-fixed no_full_text rows)"
    agreement["n_no_majority"] = sum(1 for oid in judged_ids if majority_label(
        [verdicts[oid][1], verdicts[oid][2], verdicts[oid][3]]
    ) is None)
    agreement["with_withheld"] = _agreement_over(opaque_ids)
    agreement["with_withheld"]["population"] = (
        "all 86 Review items, including the 8 protocol-fixed rows"
    )
    agreement["with_withheld"]["n_no_majority"] = len(no_majority_ids)
    agreement["n_protocol_fixed_excluded"] = len(fixed_opaque_ids)

    return {
        "generated": now_iso(),
        "source": (
            "three human-review workbooks (Review sheet) joined through the workbook key; "
            "no reviewer identity read or recorded"
        ),
        "n_items": len(items),
        "n_constructed": sum(1 for it in items if it["set"] == "constructed"),
        "n_real": sum(1 for it in items if it["set"] == "real"),
        "no_majority_item_ids": no_majority_ids,
        "inter_rater_agreement": agreement,
        "items": items,
    }


# --------------------------------------------------------------------------------------
# Score v6 against the human majority label
# --------------------------------------------------------------------------------------


def score_v6_against_labels(human_labels: dict[str, Any]) -> dict[str, Any]:
    by_set: dict[str, dict[str, str]] = {"constructed": {}, "real": {}}
    for item in human_labels["items"]:
        label = item["majority_label"]
        if label is not None:
            by_set[item["set"]][item["item_id"]] = label

    out: dict[str, Any] = {}
    for set_name, prefix in SET_TO_V6_PREFIX.items():
        gold_by_id = by_set[set_name]
        set_out: dict[str, Any] = {}
        for run in RUNS:
            rows = read_jsonl(V6_DIR / f"{prefix}_run{run}.jsonl")
            pairs = [(gold_by_id[r["item_id"]], str(r.get("predicted_status")))
                     for r in rows if r["item_id"] in gold_by_id]
            gold = [g for g, _ in pairs]
            pred = [p for _, p in pairs]
            n = len(pairs)
            n_correct = sum(1 for g, p in pairs if g == p)
            lo, hi = wilson_interval(n_correct, n) if n else (None, None)
            prf = per_class_prf(gold, pred, list(CANONICAL_STATUSES))
            verified = prf["verified"]
            falsely_verified = verified["predicted"] - verified["tp"]
            set_out[f"run_{run}"] = {
                "n": n,
                "n_matched_to_human_label": n,
                "n_missing_human_label": len(rows) - n,
                "accuracy": (n_correct / n) if n else None,
                "n_correct": n_correct,
                "accuracy_wilson_95": [lo, hi],
                "kappa": cohens_kappa(gold, pred),
                "verified_precision": verified["precision"],
                "verified_recall": verified["recall"],
                "falsely_verified": falsely_verified,
                "confusion": {
                    f"{g}->{p}": c
                    for (g, p), c in sorted(Counter(zip(gold, pred, strict=True)).items())
                },
            }
        out[set_name] = set_out
    return {
        "generated": now_iso(),
        "label_of_record": "human majority verdict (results/v6/human_labels.json)",
        "note": (
            "Scored against results/v6/*.jsonl (the guard-7 replay of the cached model "
            "answers), not against the workbook's own stored verifier_status column."
        ),
        "by_set": out,
    }


# --------------------------------------------------------------------------------------
# Fold into results/v6/summary.json + summary.md
# --------------------------------------------------------------------------------------


def _md_human_review_table(scores: dict[str, Any]) -> list[str]:
    lines = [
        "## Table E2-h reviewer majority, primary label of record",
        "",
        "The constructed and real claim sets are scored here against the three reviewers' "
        "majority verdict (`results/v6/human_labels.json`), the label of record. Tables "
        "E2-d/E2-g below score the same v6 runs against the earlier model-authored/"
        "adjudicated label and are kept as a secondary comparison.",
        "",
        "| set | run | n | accuracy | wilson 95% | kappa | verified precision | "
        "verified recall | falsely verified |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for set_name, runs in scores["by_set"].items():
        for run_key, block in runs.items():
            lo, hi = block["accuracy_wilson_95"]
            lines.append(
                f"| {set_name} | {run_key} | {block['n']} | {block['accuracy']:.4f} | "
                f"{lo:.3f}-{hi:.3f} | {block['kappa']:.4f} | "
                f"{block['verified_precision']:.4f} | {block['verified_recall']:.4f} | "
                f"{block['falsely_verified']} |"
            )
    lines.append("")
    return lines


def update_v6_summary(human_labels: dict[str, Any], scores: dict[str, Any]) -> None:
    import summarize as sm  # noqa: PLC0415

    summary = read_json(V6_DIR / "summary.json")
    summary["human_review"] = {
        "label_of_record": True,
        "majority_rule": (
            "at least two of three reviewers; a three-way split is no_majority and excluded"
        ),
        "inter_rater_agreement": human_labels["inter_rater_agreement"],
        "scores": scores,
        "secondary_comparison": (
            "hss_vs_annotation and real_vs_annotation below score the same v6 runs against "
            "the earlier model-authored/adjudicated label and are kept as a secondary, "
            "non-primary comparison"
        ),
    }
    write_json(V6_DIR / "summary.json", summary)

    base_markdown = sm.render_markdown(summary)
    header, _, rest = base_markdown.partition("\n\n")
    note = (
        "\n\nPrimary label of record: Table E2-h (the three reviewers' majority verdict). "
        "Tables E2-d and E2-g below score the same runs against the earlier "
        "model-authored/adjudicated label and are a secondary comparison, kept because "
        "summary.json's own hss_vs_annotation/real_vs_annotation blocks are read by other "
        "consumers.\n"
    )
    human_section = "\n".join(_md_human_review_table(scores))
    markdown = header + note + "\n" + rest.rstrip() + "\n\n" + human_section
    (V6_DIR / "summary.md").write_text(markdown, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--workbook", action="append", type=Path, dest="workbooks", default=None,
        help=(
            "one reviewer's filled Review-and-Screening workbook, read-only. Repeat "
            "exactly three times, in reviewer order (reviewer 1, 2, 3). No default: the "
            "three workbooks are local, gitignored files the author holds; this script "
            "never names them."
        ),
    )
    ap.add_argument(
        "--key", type=Path, default=None,
        help="the workbook key CSV (opaque_id -> real item_id/set join), read-only.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.workbooks or len(args.workbooks) != 3 or args.key is None:
        print(
            "ERROR: --workbook (repeated exactly three times, in reviewer order) and --key "
            "are both required; this script has no default workbook location."
        )
        return 2

    workbook_paths = (args.workbooks[0], args.workbooks[1], args.workbooks[2])
    human_labels = build_human_labels(workbook_paths, args.key)
    write_json(V6_DIR / "human_labels.json", human_labels)

    scores = score_v6_against_labels(human_labels)
    write_json(V6_DIR / "human_review_scores.json", scores)

    update_v6_summary(human_labels, scores)
    print(f"wrote {V6_DIR / 'human_labels.json'}, human_review_scores.json; "
          f"updated summary.json/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

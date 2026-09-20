"""Aggregate round-3 blind triple annotation (argparse copy of aggregate_annotations_v2.py).

Reads the three independent annotator sheets, an optional withheld key, and the blind
sheet itself (used only to recover which items showed a real passage, the no-match
marker, or the withheld marker), then computes:
  - per-item labels from annotators A/B/C
  - majority label per item (None when there is a three-way split)
  - unanimity count (items where A == B == C) and the list of non-unanimous items
  - Fleiss' kappa over all items, and separately over the subset of items that show
    evidence (a real passage or the no-match marker; withheld items are excluded because
    there is nothing for annotators to judge)
  - raw pairwise percent agreement and Cohen's kappa (via
    sklearn.metrics.cohen_kappa_score) for the AB, AC, BC pairs, for both the full set and
    the evidence-only subset

Two arms, selected by ``--key-arm``:

``--key-arm off`` (the default, and the only arm usable when there is no key, or a key
that carries no expected label, as for the real-claims set). ``needs_adjudication`` is
``not unanimous``: a unanimous blind majority stands as the label without reference to any
construction key. When a key with ``construction_category``/``expected_label`` columns is
given, ``expected_agree`` and the confusion/per-category tables are still computed and
written as reported diagnostics, but they do not affect ``needs_adjudication``. Works with
no ``--key`` at all: items are then identified only by ``ann_id``. This arm's aggregate
also carries ``n_adjudicated``, the count of non-unanimous items
(``len(items_needing_adjudication)``); the ``--key-arm on`` aggregate does not carry this
key, so the round-2 byte-identity regression is unaffected.

``--key-arm on`` reproduces `aggregate_annotations_v2.py`'s behaviour byte for byte, given the
same inputs: ``needs_adjudication`` is ``(not unanimous) or (majority outside the
expected-label set)``, and the key's ``item_id``, ``construction_category`` and
``expected_label`` columns are required. This setting exists only for the byte-identity
regression test in evaluation/tests/test_claims_annotation_v3.py; the HSS test set and the
real-claims set do not use it for real, because ``--altered-expected strict`` (see
build_hss_set.py) removed the disjunctive expected labels that made that check harmless.

Inputs, all paths given on the command line:
  --annotator-a/-b/-c   the three completed sheets (ann_id, claim_text, cited_paper_doi,
                        annotator_label, annotator_confidence, annotator_note)
  --sheet               the blind sheet, for cited_paper_passage -> evidence_mode
  --key                 optional withheld key; with (ann_id, item_id) only, or with
                        (ann_id, item_id, construction_category, expected_label[, built_at])

Outputs, written under --out-dir as ``{out-prefix}_completed_{suffix}.csv``,
``{out-prefix}_aggregate_{suffix}.json`` and ``{out-prefix}_aggregate_{suffix}.md``.

Deterministic: no randomness, no network calls, no LLM/paid API calls. Needs pandas and
scikit-learn (available to the evaluation suite's interpreter and to
"evaluation/.venv/Scripts/python").
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATORS = ["A", "B", "C"]

WITHHELD_MARKER = "Evidence withheld: no passage or paper text is available for this item"
NO_MATCH_MARKER = "No matching passage found in the cited paper (best overlap below threshold)"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", required=True)
    parser.add_argument("--annotator-b", required=True)
    parser.add_argument("--annotator-c", required=True)
    parser.add_argument("--sheet", required=True, help="the blind sheet, for evidence_mode")
    parser.add_argument("--key", default=None, help="optional withheld key csv")
    parser.add_argument("--key-arm", choices=["on", "off"], default="off")
    parser.add_argument("--out-dir", default=BASE_DIR)
    parser.add_argument("--out-prefix", default="hss_annotation")
    parser.add_argument("--suffix", required=True)
    parser.add_argument(
        "--title",
        default="HSS annotation aggregate",
        help="heading of the markdown report; change it when the set is not the HSS one",
    )
    return parser.parse_args(argv)


def load_annotator(path: str, letter: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(
        columns={
            "annotator_label": f"label_{letter}",
            "annotator_confidence": f"conf_{letter}",
            "annotator_note": f"note_{letter}",
        }
    )
    keep = [
        "ann_id",
        "claim_text",
        "cited_paper_doi",
        f"label_{letter}",
        f"conf_{letter}",
        f"note_{letter}",
    ]
    return df[keep]


def load_key(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    key_cols = ("ann_id", "item_id", "construction_category", "expected_label")
    cols = [c for c in key_cols if c in df.columns]
    return df[cols]


def load_evidence_mode(path: str) -> pd.DataFrame:
    """Derive evidence_mode per ann_id from the (unblinded) sheet's passage column."""
    df = pd.read_csv(path, encoding="utf-8-sig")

    def classify(passage: str) -> str:
        text = str(passage).strip()
        if text == WITHHELD_MARKER:
            return "withheld"
        if text == NO_MATCH_MARKER:
            return "no_match"
        return "passage"

    df["evidence_mode"] = df["cited_paper_passage"].apply(classify)
    return df[["ann_id", "evidence_mode"]]


def majority_label(labels: list[str]) -> str | None:
    """Return the label held by a strict majority of raters, else None.

    With 3 raters this means: all three raters agree (unanimous), or exactly two agree. A
    three-way split (all three different) returns None.
    """
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    best_count = max(counts.values())
    winners = [lab for lab, c in counts.items() if c == best_count]
    if best_count >= 2 and len(winners) == 1:
        return winners[0]
    return None


def fleiss_kappa(per_item_counts: list[dict[str, int]], categories: list[str]) -> float:
    """Standard Fleiss' kappa for n raters, N items, k categories.

    per_item_counts[i][cat] = number of raters who assigned `cat` to item i. All rows must
    sum to the same n (number of raters per item).
    """
    n_items = len(per_item_counts)
    n_raters = sum(per_item_counts[0].values())

    p_i_list = []
    category_totals = {c: 0 for c in categories}
    for row in per_item_counts:
        sum_sq = sum(row.get(c, 0) ** 2 for c in categories)
        p_i = (sum_sq - n_raters) / (n_raters * (n_raters - 1))
        p_i_list.append(p_i)
        for c in categories:
            category_totals[c] += row.get(c, 0)

    p_bar = sum(p_i_list) / n_items
    p_j = {c: category_totals[c] / (n_items * n_raters) for c in categories}
    p_e_bar = sum(v ** 2 for v in p_j.values())

    if p_e_bar == 1:
        return 1.0
    return (p_bar - p_e_bar) / (1 - p_e_bar)


def expected_label_set(expected_label: str) -> set[str]:
    """Split a possibly pipe-separated expected_label into a set of labels."""
    if pd.isna(expected_label):
        return set()
    return {part.strip() for part in str(expected_label).split("|")}


def per_item_counts_for(merged: pd.DataFrame, categories: list[str]) -> list[dict[str, int]]:
    counts_list: list[dict[str, int]] = []
    for _, row in merged.iterrows():
        labels = [row["label_A"], row["label_B"], row["label_C"]]
        counts = {cat: 0 for cat in categories}
        for lab in labels:
            counts[lab] += 1
        counts_list.append(counts)
    return counts_list


def pairwise_stats(merged: pd.DataFrame, categories: list[str]) -> dict[str, dict[str, float]]:
    pairwise: dict[str, dict[str, float]] = {}
    for x, y in combinations(ANNOTATORS, 2):
        la = merged[f"label_{x}"]
        lb = merged[f"label_{y}"]
        percent_agreement = float((la == lb).mean())
        kappa = float(cohen_kappa_score(la, lb, labels=categories))
        pairwise[f"{x}{y}"] = {
            "percent_agreement": percent_agreement,
            "cohen_kappa": kappa,
        }
    return pairwise


def build_merged(args) -> tuple[pd.DataFrame, bool, bool]:
    a = load_annotator(args.annotator_a, "A")
    b = load_annotator(args.annotator_b, "B").drop(columns=["claim_text", "cited_paper_doi"])
    c = load_annotator(args.annotator_c, "C").drop(columns=["claim_text", "cited_paper_doi"])
    merged = a.merge(b, on="ann_id", how="inner").merge(c, on="ann_id", how="inner")

    has_item_id = False
    has_expected = False
    if args.key:
        key = load_key(args.key)
        merged = merged.merge(key, on="ann_id", how="inner")
        has_item_id = "item_id" in key.columns
        has_expected = "construction_category" in key.columns and "expected_label" in key.columns

    evidence = load_evidence_mode(args.sheet)
    merged = merged.merge(evidence, on="ann_id", how="inner")

    if len(merged) != len(a) or merged.isna().any().any():
        raise ValueError(
            "Merge produced missing rows or NaNs; annotator/key/sheet ann_id sets or row "
            "counts do not line up as expected."
        )

    merged["ann_num"] = merged["ann_id"].str.rsplit("-", n=1).str[-1].astype(int)
    merged = merged.sort_values("ann_num").reset_index(drop=True)
    merged = merged.drop(columns=["ann_num"])
    return merged, has_item_id, has_expected


def compute_common(merged: pd.DataFrame) -> dict:
    categories = sorted(
        set(merged["label_A"]) | set(merged["label_B"]) | set(merged["label_C"])
    )

    majority_labels: list[str | None] = []
    unanimous_flags: list[bool] = []
    for _, row in merged.iterrows():
        labels = [row["label_A"], row["label_B"], row["label_C"]]
        majority_labels.append(majority_label(labels))
        unanimous_flags.append(labels[0] == labels[1] == labels[2])

    merged = merged.copy()
    merged["majority_label"] = majority_labels
    merged["unanimous"] = unanimous_flags

    unanimity_count = int(sum(unanimous_flags))

    counts_all = per_item_counts_for(merged, categories)
    fk_all = fleiss_kappa(counts_all, categories)

    evidence_mask = merged["evidence_mode"].isin(["passage", "no_match"])
    merged_evidence = merged.loc[evidence_mask].reset_index(drop=True)
    counts_evidence = per_item_counts_for(merged_evidence, categories)
    fk_evidence = fleiss_kappa(counts_evidence, categories) if len(merged_evidence) else None

    pairwise_all = pairwise_stats(merged, categories)
    pairwise_evidence = pairwise_stats(merged_evidence, categories) if len(merged_evidence) else {}

    return {
        "merged": merged,
        "merged_evidence": merged_evidence,
        "categories": categories,
        "unanimity_count": unanimity_count,
        "fk_all": fk_all,
        "fk_evidence": fk_evidence,
        "pairwise_all": pairwise_all,
        "pairwise_evidence": pairwise_evidence,
    }


def run_key_arm_on(merged: pd.DataFrame, has_item_id: bool, has_expected: bool) -> dict:
    if not (has_item_id and has_expected):
        raise ValueError(
            "--key-arm on requires a --key file with item_id, construction_category and "
            "expected_label columns"
        )
    common = compute_common(merged)
    merged = common["merged"]
    merged_evidence = common["merged_evidence"]

    def agrees_with_expected(maj, expected_label) -> bool:
        if maj is None:
            return False
        return maj in expected_label_set(expected_label)

    merged["expected_agree"] = [
        agrees_with_expected(maj, exp)
        for maj, exp in zip(merged["majority_label"], merged["expected_label"])
    ]
    merged["needs_adjudication"] = (~merged["unanimous"]) | (~merged["expected_agree"])

    overall_agreement = float(merged["expected_agree"].mean())

    per_category: dict[str, dict[str, float]] = {}
    for cat, group in merged.groupby("construction_category"):
        per_category[cat] = {
            "n": int(len(group)),
            "agreement_rate": float(group["expected_agree"].mean()),
        }

    majority_display = merged["majority_label"].fillna("no_majority")
    confusion = pd.crosstab(majority_display, merged["expected_label"])
    confusion_dict = {
        str(row_label): {str(col_label): int(v) for col_label, v in row.items()}
        for row_label, row in confusion.iterrows()
    }

    disagreements = merged.loc[merged["needs_adjudication"], "item_id"].tolist()
    non_unanimous_items = merged.loc[~merged["unanimous"], "item_id"].tolist()

    items = []
    for _, row in merged.iterrows():
        items.append({
            "ann_id": row["ann_id"],
            "item_id": row["item_id"],
            "evidence_mode": row["evidence_mode"],
            "label_A": row["label_A"],
            "label_B": row["label_B"],
            "label_C": row["label_C"],
            "majority_label": row["majority_label"],
            "unanimous": bool(row["unanimous"]),
            "construction_category": row["construction_category"],
            "expected_label": row["expected_label"],
            "needs_adjudication": bool(row["needs_adjudication"]),
        })

    out_cols = [
        "ann_id", "item_id", "claim_text", "cited_paper_doi", "evidence_mode",
        "label_A", "conf_A", "note_A", "label_B", "conf_B", "note_B",
        "label_C", "conf_C", "note_C", "majority_label", "unanimous",
        "construction_category", "expected_label", "needs_adjudication",
    ]
    completed = merged[out_cols].copy()
    completed["majority_label"] = completed["majority_label"].fillna("")

    n_items = int(len(merged))
    n_evidence = int(len(merged_evidence))
    aggregate = {
        "n_items": n_items,
        "n_items_evidence_subset": n_evidence,
        "annotators": ANNOTATORS,
        "categories": common["categories"],
        "unanimity_count": common["unanimity_count"],
        "non_unanimous_items": non_unanimous_items,
        f"fleiss_kappa_all{n_items}": common["fk_all"],
        f"fleiss_kappa_evidence{n_evidence}": common["fk_evidence"],
        f"pairwise_agreement_all{n_items}": common["pairwise_all"],
        f"pairwise_agreement_evidence{n_evidence}": common["pairwise_evidence"],
        "overall_majority_vs_expected_agreement": overall_agreement,
        "per_construction_category_agreement": per_category,
        "confusion_majority_vs_expected": confusion_dict,
        "items_needing_adjudication": disagreements,
        "items": items,
    }
    return {"aggregate": aggregate, "completed": completed, "merged": merged}


def run_key_arm_off(merged: pd.DataFrame, has_item_id: bool, has_expected: bool) -> dict:
    common = compute_common(merged)
    merged = common["merged"]
    merged_evidence = common["merged_evidence"]

    id_col = "item_id" if has_item_id else "ann_id"

    if has_expected:
        def agrees_with_expected(maj, expected_label) -> bool:
            if maj is None:
                return False
            return maj in expected_label_set(expected_label)

        merged["expected_agree"] = [
            agrees_with_expected(maj, exp)
            for maj, exp in zip(merged["majority_label"], merged["expected_label"])
        ]
        overall_agreement = float(merged["expected_agree"].mean())
        per_category: dict[str, dict[str, float]] | None = {}
        for cat, group in merged.groupby("construction_category"):
            per_category[cat] = {
                "n": int(len(group)),
                "agreement_rate": float(group["expected_agree"].mean()),
            }
        majority_display = merged["majority_label"].fillna("no_majority")
        confusion = pd.crosstab(majority_display, merged["expected_label"])
        confusion_dict: dict | None = {
            str(row_label): {str(col_label): int(v) for col_label, v in row.items()}
            for row_label, row in confusion.iterrows()
        }
    else:
        overall_agreement = None
        per_category = None
        confusion_dict = None

    merged["needs_adjudication"] = ~merged["unanimous"]
    disagreements = merged.loc[merged["needs_adjudication"], id_col].tolist()
    non_unanimous_items = merged.loc[~merged["unanimous"], id_col].tolist()

    items = []
    for _, row in merged.iterrows():
        item = {"ann_id": row["ann_id"]}
        if has_item_id:
            item["item_id"] = row["item_id"]
        item["evidence_mode"] = row["evidence_mode"]
        item["label_A"] = row["label_A"]
        item["label_B"] = row["label_B"]
        item["label_C"] = row["label_C"]
        item["majority_label"] = row["majority_label"]
        item["unanimous"] = bool(row["unanimous"])
        if has_expected:
            item["construction_category"] = row["construction_category"]
            item["expected_label"] = row["expected_label"]
            item["expected_agree"] = bool(row["expected_agree"])
        item["needs_adjudication"] = bool(row["needs_adjudication"])
        items.append(item)

    out_cols = ["ann_id"]
    if has_item_id:
        out_cols.append("item_id")
    out_cols += ["claim_text", "cited_paper_doi", "evidence_mode",
                 "label_A", "conf_A", "note_A", "label_B", "conf_B", "note_B",
                 "label_C", "conf_C", "note_C", "majority_label", "unanimous"]
    if has_expected:
        out_cols += ["construction_category", "expected_label", "expected_agree"]
    out_cols.append("needs_adjudication")
    completed = merged[out_cols].copy()
    completed["majority_label"] = completed["majority_label"].fillna("")

    n_items = int(len(merged))
    n_evidence = int(len(merged_evidence))
    aggregate = {
        "key_arm": "off",
        "has_key": has_item_id or has_expected,
        "has_expected_labels": has_expected,
        "n_items": n_items,
        "n_items_evidence_subset": n_evidence,
        "annotators": ANNOTATORS,
        "categories": common["categories"],
        "unanimity_count": common["unanimity_count"],
        "non_unanimous_items": non_unanimous_items,
        f"fleiss_kappa_all{n_items}": common["fk_all"],
        f"fleiss_kappa_evidence{n_evidence}": common["fk_evidence"],
        f"pairwise_agreement_all{n_items}": common["pairwise_all"],
        f"pairwise_agreement_evidence{n_evidence}": common["pairwise_evidence"],
        "items_needing_adjudication": disagreements,
        "n_adjudicated": len(disagreements),
        "items": items,
    }
    if has_expected:
        aggregate["overall_majority_vs_expected_agreement"] = overall_agreement
        aggregate["per_construction_category_agreement"] = per_category
        aggregate["confusion_majority_vs_expected"] = confusion_dict
    return {"aggregate": aggregate, "completed": completed, "merged": merged}


def write_markdown(aggregate: dict, args) -> str:
    key_arm = aggregate.get("key_arm", "on")
    n = aggregate["n_items"]
    n_ev = aggregate["n_items_evidence_subset"]
    lines: list[str] = []
    lines.append(f"# {args.title} ({args.suffix}, key-arm {key_arm})")
    lines.append("")
    lines.append(
        f"Aggregation of the round-3 blind triple annotation (annotators A, B, C) for "
        f"the {n}-item set, produced by aggregate_annotations_v3.py."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| items | {n} |")
    lines.append(f"| evidence-subset items (passage or no_match, excl. withheld) | {n_ev} |")
    lines.append(f"| unanimous items (A == B == C) | {aggregate['unanimity_count']} |")
    if "n_adjudicated" in aggregate:
        lines.append(f"| adjudicated (non-unanimous) items | {aggregate['n_adjudicated']} |")
    fk_all_key = next(k for k in aggregate if k.startswith("fleiss_kappa_all"))
    fk_ev_key = next(k for k in aggregate if k.startswith("fleiss_kappa_evidence"))
    fk_all_val = aggregate[fk_all_key]
    fk_ev_val = aggregate[fk_ev_key]
    lines.append(f"| Fleiss' kappa (all {n} items) | {fk_all_val:.3f} |")
    if fk_ev_val is not None:
        lines.append(f"| Fleiss' kappa (evidence subset, {n_ev} items) | {fk_ev_val:.3f} |")
    if "overall_majority_vs_expected_agreement" in aggregate:
        lines.append(
            "| majority vs expected agreement (overall, diagnostic only) | "
            f"{aggregate['overall_majority_vs_expected_agreement']:.3f} |"
        )
    lines.append("")

    pw_all_key = next(k for k in aggregate if k.startswith("pairwise_agreement_all"))
    pw_ev_key = next(k for k in aggregate if k.startswith("pairwise_agreement_evidence"))
    lines.append("## Pairwise agreement, all items")
    lines.append("")
    lines.append("| pair | percent agreement | Cohen's kappa |")
    lines.append("|---|---|---|")
    for pair_name in ["AB", "AC", "BC"]:
        vals = aggregate[pw_all_key][pair_name]
        lines.append(
            f"| {pair_name} | {vals['percent_agreement']:.3f} | {vals['cohen_kappa']:.3f} |"
        )
    lines.append("")

    if aggregate[pw_ev_key]:
        lines.append("## Pairwise agreement, evidence subset")
        lines.append("")
        lines.append("| pair | percent agreement | Cohen's kappa |")
        lines.append("|---|---|---|")
        for pair_name in ["AB", "AC", "BC"]:
            vals = aggregate[pw_ev_key][pair_name]
            lines.append(
                f"| {pair_name} | {vals['percent_agreement']:.3f} | {vals['cohen_kappa']:.3f} |"
            )
        lines.append("")

    lines.append("## Non-unanimous items (adjudication trigger)")
    lines.append("")
    if aggregate["items_needing_adjudication"]:
        for item_id in aggregate["items_needing_adjudication"]:
            lines.append(f"- {item_id}")
    else:
        lines.append("None: every item is unanimous.")
    lines.append("")

    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    merged, has_item_id, has_expected = build_merged(args)

    if args.key_arm == "on":
        result = run_key_arm_on(merged, has_item_id, has_expected)
    else:
        result = run_key_arm_off(merged, has_item_id, has_expected)

    os.makedirs(args.out_dir, exist_ok=True)

    completed_path = os.path.join(args.out_dir, f"{args.out_prefix}_completed_{args.suffix}.csv")
    result["completed"].to_csv(completed_path, index=False, encoding="utf-8-sig")

    json_path = os.path.join(args.out_dir, f"{args.out_prefix}_aggregate_{args.suffix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result["aggregate"], f, indent=2, ensure_ascii=False)
        f.write("\n")

    md_path = os.path.join(args.out_dir, f"{args.out_prefix}_aggregate_{args.suffix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(write_markdown(result["aggregate"], args))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {completed_path}")
    print(f"Unanimous: {result['aggregate']['unanimity_count']}/{result['aggregate']['n_items']}")
    print(f"Items needing adjudication: {result['aggregate']['items_needing_adjudication']}")
    return result


if __name__ == "__main__":
    main()

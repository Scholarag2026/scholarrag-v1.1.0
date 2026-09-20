"""Aggregate the blind triple annotation for the ScholarRAG HSS claim-verification set.

Reads the three independent annotator sheets, the construction
answer key, and the blinding sheet (used only to recover which items
showed a real passage, the no-match marker, or the withheld marker), then
computes:
  - per-item labels from annotators A/B/C
  - majority label per item (None when there is a three-way split)
  - unanimity count (items where A == B == C) and the list of non-unanimous
    items (the adjudication trigger)
  - Fleiss' kappa over all 30 items, and separately over the subset of items
    that show evidence (a real passage or the no-match marker; the withheld
    items are excluded because there is nothing for annotators to judge)
  - raw pairwise percent agreement and Cohen's kappa (via
    sklearn.metrics.cohen_kappa_score) for the AB, AC, BC pairs, for both the
    all-30 set and the evidence-only subset
  - agreement of the majority label with expected_label, overall and per
    construction_category (expected_label may be a pipe-separated set, e.g.
    "unsupported|needs_nuance"; a majority label inside that set counts as
    agreement)
  - a confusion table of majority label vs expected_label

Inputs (evaluation/claims/annotation/):
  annotator2_A.csv, annotator2_B.csv, annotator2_C.csv
  hss_annotation_key_v2.csv
  hss_annotation_sheet_v2.csv

Outputs (evaluation/claims/annotation/):
  hss_annotation_aggregate_v2.json
  hss_annotation_aggregate_v2.md
  hss_annotation_completed_v2.csv

Deterministic: no randomness, no network calls, no LLM/paid API calls.
Run with the evaluation virtualenv's Python (pandas + scikit-learn):
  "evaluation/.venv/Scripts/python" aggregate_annotations_v2.py
"""

from __future__ import annotations

import json
import os
from itertools import combinations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATORS = ["A", "B", "C"]

WITHHELD_MARKER = "Evidence withheld: no passage or paper text is available for this item"
NO_MATCH_MARKER = "No matching passage found in the cited paper (best overlap below threshold)"


def load_annotator(letter: str) -> pd.DataFrame:
    path = os.path.join(BASE_DIR, f"annotator2_{letter}.csv")
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


def load_key() -> pd.DataFrame:
    path = os.path.join(BASE_DIR, "hss_annotation_key_v2.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df[["ann_id", "item_id", "construction_category", "expected_label"]]


def load_evidence_mode() -> pd.DataFrame:
    """Derive evidence_mode per ann_id from the (unblinded) sheet's passage column."""
    path = os.path.join(BASE_DIR, "hss_annotation_sheet_v2.csv")
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

    With 3 raters this means: all three raters agree (unanimous), or exactly
    two agree. A three-way split (all three different) returns None.
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

    per_item_counts[i][cat] = number of raters who assigned `cat` to item i.
    All rows must sum to the same n (number of raters per item).
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


def build_merged() -> pd.DataFrame:
    a = load_annotator("A")
    b = load_annotator("B").drop(columns=["claim_text", "cited_paper_doi"])
    c = load_annotator("C").drop(columns=["claim_text", "cited_paper_doi"])
    key = load_key()
    evidence = load_evidence_mode()

    merged = a.merge(b, on="ann_id", how="inner").merge(c, on="ann_id", how="inner")
    merged = merged.merge(key, on="ann_id", how="inner")
    merged = merged.merge(evidence, on="ann_id", how="inner")

    if len(merged) != len(a) or merged.isna().any().any():
        raise ValueError(
            "Merge produced missing rows or NaNs; annotator/key/sheet ann_id "
            "sets or row counts do not line up as expected."
        )

    merged["ann_num"] = merged["ann_id"].str.replace("ann-", "", regex=False).astype(int)
    merged = merged.sort_values("ann_num").reset_index(drop=True)
    merged = merged.drop(columns=["ann_num"])
    return merged


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


def main() -> None:
    merged = build_merged()

    categories = sorted(
        set(merged["label_A"]) | set(merged["label_B"]) | set(merged["label_C"])
    )

    majority_labels: list[str | None] = []
    unanimous_flags: list[bool] = []
    for _, row in merged.iterrows():
        labels = [row["label_A"], row["label_B"], row["label_C"]]
        majority_labels.append(majority_label(labels))
        unanimous_flags.append(labels[0] == labels[1] == labels[2])

    merged["majority_label"] = majority_labels
    merged["unanimous"] = unanimous_flags

    unanimity_count = int(sum(unanimous_flags))
    non_unanimous_mask = ~merged["unanimous"]
    non_unanimous_items = merged.loc[non_unanimous_mask, "item_id"].tolist()

    # ---- Fleiss' kappa: all 30 items ----
    counts_all = per_item_counts_for(merged, categories)
    fk_all30 = fleiss_kappa(counts_all, categories)

    # ---- Fleiss' kappa: evidence-only subset (passage or no_match, excl. withheld) ----
    evidence_mask = merged["evidence_mode"].isin(["passage", "no_match"])
    merged_evidence = merged.loc[evidence_mask].reset_index(drop=True)
    counts_evidence = per_item_counts_for(merged_evidence, categories)
    fk_evidence25 = fleiss_kappa(counts_evidence, categories)

    # ---- pairwise agreement + Cohen's kappa, both subsets ----
    pairwise_all30 = pairwise_stats(merged, categories)
    pairwise_evidence25 = pairwise_stats(merged_evidence, categories)

    def agrees_with_expected(maj: str | None, expected_label: str) -> bool:
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

    items = []
    for _, row in merged.iterrows():
        items.append(
            {
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
            }
        )

    # ---- hss_annotation_completed_v2.csv ----
    out_cols = [
        "ann_id",
        "item_id",
        "claim_text",
        "cited_paper_doi",
        "evidence_mode",
        "label_A",
        "conf_A",
        "note_A",
        "label_B",
        "conf_B",
        "note_B",
        "label_C",
        "conf_C",
        "note_C",
        "majority_label",
        "unanimous",
        "construction_category",
        "expected_label",
        "needs_adjudication",
    ]
    completed = merged[out_cols].copy()
    completed["majority_label"] = completed["majority_label"].fillna("")
    completed_path = os.path.join(BASE_DIR, "hss_annotation_completed_v2.csv")
    completed.to_csv(completed_path, index=False, encoding="utf-8-sig")

    # ---- hss_annotation_aggregate_v2.json ----
    aggregate = {
        "n_items": int(len(merged)),
        "n_items_evidence_subset": int(len(merged_evidence)),
        "annotators": ANNOTATORS,
        "categories": categories,
        "unanimity_count": unanimity_count,
        "non_unanimous_items": non_unanimous_items,
        "fleiss_kappa_all30": fk_all30,
        "fleiss_kappa_evidence25": fk_evidence25,
        "pairwise_agreement_all30": pairwise_all30,
        "pairwise_agreement_evidence25": pairwise_evidence25,
        "overall_majority_vs_expected_agreement": overall_agreement,
        "per_construction_category_agreement": per_category,
        "confusion_majority_vs_expected": confusion_dict,
        "items_needing_adjudication": disagreements,
        "items": items,
    }
    json_path = os.path.join(BASE_DIR, "hss_annotation_aggregate_v2.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # ---- hss_annotation_aggregate_v2.md ----
    lines: list[str] = []
    lines.append("# HSS annotation aggregate (round 2)")
    lines.append("")
    lines.append(
        "Aggregation of the round 2 blind triple annotation (annotators A, B, "
        "C) for the 30-item HSS claim-verification set, produced by "
        "aggregate_annotations_v2.py."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| items | {len(merged)} |")
    lines.append(
        f"| evidence-subset items (passage or no_match, excl. withheld) | {len(merged_evidence)} |"
    )
    lines.append(f"| unanimous items (A == B == C) | {unanimity_count} |")
    lines.append(f"| Fleiss' kappa (all {len(merged)} items) | {fk_all30:.3f} |")
    lines.append(
        f"| Fleiss' kappa (evidence subset, {len(merged_evidence)} items) | {fk_evidence25:.3f} |"
    )
    lines.append(
        f"| majority vs expected agreement (overall) | {overall_agreement:.3f} |"
    )
    lines.append("")

    lines.append("## Pairwise agreement, all items")
    lines.append("")
    lines.append("| pair | percent agreement | Cohen's kappa |")
    lines.append("|---|---|---|")
    for pair_name in ["AB", "AC", "BC"]:
        vals = pairwise_all30[pair_name]
        lines.append(
            f"| {pair_name} | {vals['percent_agreement']:.3f} | "
            f"{vals['cohen_kappa']:.3f} |"
        )
    lines.append("")

    lines.append("## Pairwise agreement, evidence subset")
    lines.append("")
    lines.append("| pair | percent agreement | Cohen's kappa |")
    lines.append("|---|---|---|")
    for pair_name in ["AB", "AC", "BC"]:
        vals = pairwise_evidence25[pair_name]
        lines.append(
            f"| {pair_name} | {vals['percent_agreement']:.3f} | "
            f"{vals['cohen_kappa']:.3f} |"
        )
    lines.append("")

    lines.append("## Majority vs expected agreement by construction category")
    lines.append("")
    lines.append("| construction_category | n | agreement rate |")
    lines.append("|---|---|---|")
    for cat in sorted(per_category):
        vals = per_category[cat]
        lines.append(f"| {cat} | {vals['n']} | {vals['agreement_rate']:.3f} |")
    lines.append("")

    lines.append("## Confusion table: majority label vs expected_label")
    lines.append("")

    def md_escape(value: str) -> str:
        # Escape literal "|" (e.g. in "unsupported|needs_nuance") so it is
        # not read as a markdown table cell separator.
        return str(value).replace("|", "\\|")

    expected_cols = list(confusion.columns)
    header = "| majority \\ expected | " + " | ".join(
        md_escape(c) for c in expected_cols
    ) + " |"
    sep = "|---|" + "|".join(["---"] * len(expected_cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for row_label in confusion.index:
        row_vals = [str(confusion.loc[row_label, col]) for col in expected_cols]
        lines.append(f"| {md_escape(row_label)} | " + " | ".join(row_vals) + " |")
    lines.append("")

    lines.append("## Items needing adjudication (not unanimous OR majority outside expected set)")
    lines.append("")
    if disagreements:
        for item_id in disagreements:
            lines.append(f"- {item_id}")
    else:
        lines.append(
            "None: every item is unanimous and its majority label agrees "
            "with expected_label."
        )
    lines.append("")

    md_path = os.path.join(BASE_DIR, "hss_annotation_aggregate_v2.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {completed_path}")
    print(f"Fleiss kappa (all 30): {fk_all30:.3f}")
    print(f"Fleiss kappa (evidence {len(merged_evidence)}): {fk_evidence25:.3f}")
    print(f"Unanimous: {unanimity_count}/{len(merged)}")
    print(f"Non-unanimous items: {non_unanimous_items}")
    print(f"Overall majority-vs-expected agreement: {overall_agreement:.3f}")
    print(f"Items needing adjudication: {disagreements}")


if __name__ == "__main__":
    main()

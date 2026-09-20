"""Majority label of record over two or more reviewers' filled Delivered workbooks, for one
promoted demo run's delivered claims.

Interpreter: the system python (``openpyxl``). No backend import, no network, no LLM call.
Importers: ``tests/test_claims_delivered_human_scores.py``.

Every reviewer workbook is opened read-only and never written back to. Reads and validates
each workbook exactly as ``score_review_sheet.py`` does for one workbook (same required
fields, same refusal messages), then aligns the rows across all of them by
``(draft_id, row, sentence, claim_checked, paper_doi)`` and refuses unless every workbook
carries exactly the same set of rows.

Majority rule
-------------
For each row and each yes/no question, the majority answer is yes when at least half plus
one of the reviewers answered yes, that is ``n // 2 + 1`` out of ``n`` reviewers. With three
reviewers this is two of three. With two reviewers this means both must say yes; one yes and
one no is a no. This rule never ties: every row gets a definite yes or no majority answer.

Output
------
``--out``: a json file with, at the top level, ``demo_run_id``, ``export_time``,
``workbook_id``, ``n_reviewers``, ``reviewers`` (the ordinal labels ``reviewer_1`` and so on,
in the order the ``--workbook`` flags were given; carries no reviewer identity), ``unit``
(``"verified claim row"``), ``majority_rule``, ``n_rows``, ``per_reviewer`` (one
``score_review_sheet.score_delivered`` block per reviewer, scored on that reviewer's own
answers only), ``majority`` (the same headline shape, computed over the majority answer of
each row, with a per-section and a per-sentence block), ``agreement`` (pairwise raw agreement
and Cohen's kappa between every pair of reviewers, plus the mean of each, for both
questions), and ``rows_judged_no`` (every row where at least one reviewer answered no to
either question, with each reviewer's own answers and note, no names). The json is written
with sorted keys and no non-deterministic content, so running the same inputs twice writes
the same bytes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import score_review_sheet as srs  # noqa: E402

from common import cohens_kappa  # noqa: E402

#: Re-exported so callers and tests can catch one exception type without importing
#: ``score_review_sheet`` themselves. Raised on any structural problem: a blank required
#: human field, a wrong run id or export time, a missing Delivered sheet, or a row-set
#: mismatch between reviewers.
ReviewSheetError = srs.ReviewSheetError

#: The join key used to line up one logical row across every reviewer's own copy of the
#: workbook. ``row`` restarts at 1 in every section, so it is combined with the other four
#: fields rather than used alone.
ROW_KEY_FIELDS: tuple[str, ...] = ("draft_id", "row", "sentence", "claim_checked", "paper_doi")

#: The two yes/no questions every Delivered row carries.
YES_NO_COLUMNS: tuple[str, ...] = ("human_sentence_correct", "human_quote_supports")

#: What one aligned, judged Delivered row counts as in the manuscript.
UNIT = "verified claim row"

MAJORITY_RULE = (
    "yes when at least half plus one of the reviewers answer yes (n // 2 + 1 out of n "
    "reviewers). With three reviewers this is two of three. With two reviewers both must "
    "say yes. This rule never ties."
)


# --------------------------------------------------------------------------------------
# Reading and validating one reviewer's workbook (delegates to score_review_sheet.py)
# --------------------------------------------------------------------------------------


def load_reviewer_workbook(
    path: Path, *, demo_run_id: str | None, export_time: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
    """Reads and validates one reviewer's Delivered workbook, the same way
    ``score_review_sheet.py`` validates a single workbook: the Provenance ``demo_run_id`` and
    ``export_time`` must match when given, the workbook must carry a Delivered sheet, and
    every row must pass :func:`score_review_sheet.validate_delivered_rows`. Returns
    ``(rows, provenance, workbook_id)``. Raises :class:`ReviewSheetError` on any failure,
    reusing that function's own messages."""
    provenance = srs.read_provenance_fields(path)
    srs.validate_demo_run_id((provenance or {}).get("demo_run_id"), demo_run_id)
    srs.validate_export_time((provenance or {}).get("export_time"), export_time)
    rows = srs.read_delivered_sheet(path)
    if rows is None:
        raise ReviewSheetError(f"{path}: no {srs.DELIVERED_SHEET_NAME!r} sheet")
    srs.validate_delivered_rows(rows)
    workbook_id = srs.read_workbook_id(path)
    return rows, provenance, workbook_id


# --------------------------------------------------------------------------------------
# Aligning rows across reviewers
# --------------------------------------------------------------------------------------


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(f) for f in ROW_KEY_FIELDS)


def align_rows(
    rows_by_reviewer: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Mapping[str, Any]]]:
    """Lines up one logical Delivered row across every reviewer, keyed by
    :data:`ROW_KEY_FIELDS`. Returns a list, in the first reviewer's own row order, of
    ``{reviewer_label: row}`` dicts, one per aligned row. Raises :class:`ReviewSheetError`
    when a reviewer's own sheet has a duplicated row key, or when two reviewers do not carry
    exactly the same set of row keys."""
    labels = list(rows_by_reviewer)
    keyed: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {}
    for label in labels:
        this_keyed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        duplicated: list[tuple[Any, ...]] = []
        for row in rows_by_reviewer[label]:
            key = _row_key(row)
            if key in this_keyed:
                duplicated.append(key)
            this_keyed[key] = row
        if duplicated:
            raise ReviewSheetError(
                f"{label}: duplicated Delivered row key(s) on its own sheet: {duplicated}"
            )
        keyed[label] = this_keyed

    base_label = labels[0]
    base_keys = list(keyed[base_label])
    base_key_set = set(base_keys)
    for label in labels[1:]:
        other_set = set(keyed[label])
        if other_set != base_key_set:
            missing = sorted(
                (str(k) for k in base_key_set - other_set),
            )
            extra = sorted(
                (str(k) for k in other_set - base_key_set),
            )
            raise ReviewSheetError(
                f"{label} does not carry the same Delivered rows as {base_label}; "
                f"{len(missing)} row(s) present in {base_label} but not in {label}: "
                f"{missing}; {len(extra)} row(s) present in {label} but not in "
                f"{base_label}: {extra}"
            )

    return [{label: keyed[label][key] for label in labels} for key in base_keys]


def _canonical_row(aligned_row: Mapping[str, Mapping[str, Any]], labels: Sequence[str]) -> Mapping[str, Any]:
    """The row content fields (everything but the human answers) are the exporter's own and
    do not vary by reviewer, so the first reviewer's own copy stands in for all of them."""
    return aligned_row[labels[0]]


# --------------------------------------------------------------------------------------
# Majority rows and the majority headline
# --------------------------------------------------------------------------------------


def _majority_threshold(n_reviewers: int) -> int:
    return n_reviewers // 2 + 1


def row_majority(
    aligned_row: Mapping[str, Mapping[str, Any]], labels: Sequence[str], column: str,
) -> str:
    """The majority yes/no answer for *column* on one aligned row, under
    :data:`MAJORITY_RULE`."""
    n_yes = sum(
        1 for label in labels
        if srs.normalise_yes_no(aligned_row[label].get(column)) == "yes"
    )
    return "yes" if n_yes >= _majority_threshold(len(labels)) else "no"


def build_majority_rows(
    aligned: Sequence[Mapping[str, Mapping[str, Any]]], labels: Sequence[str],
) -> list[dict[str, Any]]:
    """One synthetic Delivered-shaped row per aligned row: the exporter's own content fields,
    plus the majority yes/no answer for each of :data:`YES_NO_COLUMNS`, in the same field
    names ``score_review_sheet.py``'s own scoring helpers expect."""
    out = []
    for entry in aligned:
        canon = _canonical_row(entry, labels)
        row: dict[str, Any] = {
            "row": canon.get("row"),
            "section_title": canon.get("section_title"),
            "draft_id": canon.get("draft_id"),
            "sentence": canon.get("sentence"),
            "claim_checked": canon.get("claim_checked"),
            "source_located": canon.get("source_located"),
            "passage_located": canon.get("passage_located"),
            "unlocated_quotes": canon.get("unlocated_quotes"),
            "sentence_in_draft": canon.get("sentence_in_draft"),
            "location_section_mismatch": canon.get("location_section_mismatch"),
        }
        for column in YES_NO_COLUMNS:
            row[column] = row_majority(entry, labels, column)
        out.append(row)
    return out


def score_majority_delivered(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The same headline shape ``score_review_sheet.score_delivered`` reports, computed over
    the majority answer of each row rather than one reviewer's own answers. Rows
    :func:`score_review_sheet._delivered_fidelity_failed` flags are excluded from the
    headline, the per-section blocks and the per-sentence block, exactly as they are for one
    reviewer."""
    n_rows = len(rows)
    excluded_rows = srs.delivered_fidelity_excluded_rows(rows)
    headline_rows = [r for r in rows if not srs._delivered_fidelity_failed(r)]  # noqa: SLF001
    sentence_block = srs._delivered_yes_no_block(  # noqa: SLF001
        headline_rows, "human_sentence_correct",
    )
    quote_block = srs._delivered_yes_no_block(headline_rows, "human_quote_supports")  # noqa: SLF001
    section_labels = srs.delivered_section_order(rows)
    by_section = {
        label: srs._score_delivered_section(  # noqa: SLF001
            [r for r in rows if srs._delivered_section_label(r) == label]  # noqa: SLF001
        )
        for label in section_labels
    }
    return {
        "n_rows": n_rows,
        "n_judged": len(headline_rows),
        "n_fidelity_excluded": len(excluded_rows),
        "sentence_correct_count": sentence_block["n_yes"],
        "sentence_correct_rate": sentence_block["rate"],
        "sentence_correct_wilson_95": sentence_block["wilson_95"],
        "quote_supports_count": quote_block["n_yes"],
        "quote_supports_rate": quote_block["rate"],
        "quote_supports_wilson_95": quote_block["wilson_95"],
        "n_sections": len(section_labels),
        "by_section": by_section,
        "by_sentence": srs.score_delivered_by_sentence(rows),
    }


# --------------------------------------------------------------------------------------
# Pairwise agreement between reviewers
# --------------------------------------------------------------------------------------


def pairwise_agreement_block(
    aligned: Sequence[Mapping[str, Mapping[str, Any]]], labels: Sequence[str], column: str,
) -> dict[str, Any]:
    """Raw percent agreement and Cohen's kappa between every pair of reviewers on *column*,
    over every aligned row, plus the mean of each pair value."""
    pairs: dict[str, Any] = {}
    percents: list[float] = []
    kappas: list[float] = []
    for a, b in itertools.combinations(labels, 2):
        answers_a = [srs.normalise_yes_no(entry[a].get(column)) for entry in aligned]
        answers_b = [srs.normalise_yes_no(entry[b].get(column)) for entry in aligned]
        n = len(answers_a)
        n_agree = sum(1 for x, y in zip(answers_a, answers_b, strict=True) if x == y)
        percent = (n_agree / n) if n else None
        kappa = cohens_kappa(answers_a, answers_b)
        pairs[f"{a}_{b}"] = {
            "n": n, "n_agree": n_agree, "percent_agreement": percent, "kappa": kappa,
        }
        if percent is not None:
            percents.append(percent)
        if kappa is not None:
            kappas.append(kappa)
    return {
        "pairs": pairs,
        "mean_percent_agreement": (sum(percents) / len(percents)) if percents else None,
        "mean_kappa": (sum(kappas) / len(kappas)) if kappas else None,
    }


# --------------------------------------------------------------------------------------
# Rows judged no by at least one reviewer
# --------------------------------------------------------------------------------------


def build_rows_judged_no(
    aligned: Sequence[Mapping[str, Mapping[str, Any]]], labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Every aligned row where at least one reviewer answered no to either question: the
    unanimous nos and the disagreements both worth a second look. Carries each reviewer's own
    two answers and note (never a name)."""
    out = []
    for entry in aligned:
        any_no = any(
            srs.normalise_yes_no(entry[label].get(column)) == "no"
            for label in labels for column in YES_NO_COLUMNS
        )
        if not any_no:
            continue
        canon = _canonical_row(entry, labels)
        reviewers = {
            label: {
                "human_sentence_correct": entry[label].get("human_sentence_correct"),
                "human_quote_supports": entry[label].get("human_quote_supports"),
                "human_note": entry[label].get("human_note") or "",
            }
            for label in labels
        }
        out.append({
            "section": srs._delivered_section_label(canon),  # noqa: SLF001
            "row": canon.get("row"),
            "claim_checked": canon.get("claim_checked") or "",
            "reviewers": reviewers,
        })
    return out


# --------------------------------------------------------------------------------------
# Top-level scoring
# --------------------------------------------------------------------------------------


def score(
    rows_by_reviewer: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    demo_run_id: str | None,
    export_time: str | None,
    workbook_id: str | None,
) -> dict[str, Any]:
    """Builds the full delivered-human-scores result from each reviewer's own already-read
    and already-validated Delivered rows. Raises :class:`ReviewSheetError` when fewer than two
    reviewers are given, or when :func:`align_rows` refuses."""
    labels = list(rows_by_reviewer)
    if len(labels) < 2:
        raise ReviewSheetError("at least two reviewer workbooks are required")
    aligned = align_rows(rows_by_reviewer)
    per_reviewer = {label: srs.score_delivered(list(rows_by_reviewer[label])) for label in labels}
    majority_rows = build_majority_rows(aligned, labels)
    majority = score_majority_delivered(majority_rows)
    agreement = {
        column: pairwise_agreement_block(aligned, labels, column) for column in YES_NO_COLUMNS
    }
    rows_judged_no = build_rows_judged_no(aligned, labels)
    return {
        "demo_run_id": demo_run_id,
        "export_time": export_time,
        "workbook_id": workbook_id,
        "n_reviewers": len(labels),
        "reviewers": labels,
        "unit": UNIT,
        "majority_rule": MAJORITY_RULE,
        "n_rows": len(aligned),
        "per_reviewer": per_reviewer,
        "majority": majority,
        "agreement": agreement,
        "rows_judged_no": rows_judged_no,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--workbook", action="append", required=True, type=Path, dest="workbooks",
        help="a reviewer's filled Delivered workbook. Repeat once per reviewer (two or more).",
    )
    ap.add_argument(
        "--demo-run-id", required=True,
        help="every workbook's own Provenance demo_run_id must equal this exactly.",
    )
    ap.add_argument(
        "--export-time", required=True,
        help="every workbook's own Provenance export_time must equal this exactly.",
    )
    ap.add_argument("--out", required=True, type=Path)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.workbooks) < 2:
        print("ERROR: at least two --workbook paths are required")
        return 2

    labels = [f"reviewer_{i}" for i in range(1, len(args.workbooks) + 1)]
    rows_by_reviewer: dict[str, list[dict[str, Any]]] = {}
    workbook_ids: dict[str, str | None] = {}
    try:
        for label, path in zip(labels, args.workbooks, strict=True):
            rows, _provenance, workbook_id = load_reviewer_workbook(
                path, demo_run_id=args.demo_run_id, export_time=args.export_time,
            )
            rows_by_reviewer[label] = rows
            workbook_ids[label] = workbook_id
        distinct_ids = sorted({str(v) for v in workbook_ids.values()})
        if len(distinct_ids) != 1:
            print(f"ERROR: reviewer workbooks do not share one workbook id: {workbook_ids}")
            return 2
        result = score(
            rows_by_reviewer, demo_run_id=args.demo_run_id, export_time=args.export_time,
            workbook_id=distinct_ids[0],
        )
    except ReviewSheetError as exc:
        print(f"ERROR: {exc}")
        return 2

    text = json.dumps(result, indent=2, sort_keys=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

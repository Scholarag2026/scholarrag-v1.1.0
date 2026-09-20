"""Score a filled Excel review workbook against the key, the model labels and the tool.

Interpreter: the evaluation venv's python (``openpyxl``) or plain system python -- no backend
import, no network, no LLM. Importers: ``tests/test_claims_score_review_sheet*.py``.

Reads back the workbook ``export_review_sheet.py`` wrote once a reviewer has filled it in, and
computes agreement with the
verifier per set and per status, agreement with the model labels, Cohen's kappa, the
disagreement_axis counts, and, when the workbook has a Screening sheet, the screening agreement
per status (all statuses, including ``unscreened``) and per criterion (restricted to actual
``exclude`` decisions -- a ``needs_review`` row demoted onto the same criterion id is a
different tool decision and would otherwise be pooled in with it), the headline overall
accuracy and kappa (``unscreened`` rows excluded from both -- there is no human-comparable tool
decision for one to agree or disagree with -- and a row
whose ``needs_review_reason`` is ``unanchored_exclude`` excluded the same way and reported on
its own, both at the headline and inside ``needs_review``'s own ``per_status`` block, since the
Guide dictates that row's ``human_status`` rather than leaving it to judgement), the
``quote_is_verbatim`` / ``criterion_is_right`` rates (the former excluding, and separately
reporting, any row a guard demoted for an unanchored quote),
and the screen-in share of the sampled exclusions, reported separately for the
numbered-criterion and off-topic strata by reading the sheet's own hidden ``stratum`` column.
Refuses (:class:`ReviewSheetError`) a workbook
with a blank required human field, naming every row and column at once rather than stopping at
the first.

Review sheet
------------
Required, non-blank on every row: ``human_verdict`` (one of the four canonical statuses) and
``human_agrees_with_verifier`` (``yes``/``no``). When ``human_agrees_with_verifier`` is ``yes``,
``human_verdict`` must equal ``verifier_status`` (a "yes" with a different verdict is a
contradiction, not a disagreement recorded elsewhere). When it is ``no``, ``human_verdict`` must
differ from ``verifier_status``, and ``disagreement_axis`` (one of the closed vocabulary),
``human_quote`` and ``human_justification`` are all required. ``countersigned_by`` and
``countersigned_at`` are never required. ``--key`` (the exporter's key csv, joined by the
Review sheet's own opaque ``item_id`` column) is required whenever the workbook has a Review
sheet; :func:`validate_workbook_binding` checks the workbook's ``Workbook id:`` Guide-sheet
stamp against the key's ``workbook_id`` column first, and :func:`validate_key_membership`
refuses unless the sheet's ``item_id`` set and the key's ``opaque_id`` set are exactly equal in
both directions -- a Review row deleted or blanked in Excel is not read at all by
:func:`_read_sheet_rows`, so without this check the scored denominator would shrink silently.

Screening sheet
----------------
Required, non-blank on every row: ``human_status`` (one of ``include``/``needs_review``/
``exclude``) and ``human_agrees`` (``yes``/``no``); ``human_note`` is required only when
``human_agrees`` is ``no``; ``quote_is_verbatim``/``criterion_is_right`` (``yes``/``no``) are
required whenever the row carries a non-blank ``quote``/``criterion`` respectively (a census row
with neither has nothing to confirm). No key file needed -- the Screening sheet never hides the
tool's own ``status``.

Output
------
``--out``: a json file with the full score block described above. A markdown summary is written
alongside it (the same path with a ``.md`` suffix) unless ``--no-markdown`` is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import openpyxl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import export_review_sheet as ex  # noqa: E402

from common import cohens_kappa, fmt, markdown_table, wilson_interval  # noqa: E402

REVIEW_SHEET_NAME = "Review"
MODEL_LABELS_SHEET_NAME = "Model_labels"
SCREENING_SHEET_NAME = "Screening"
PROVENANCE_SHEET_NAME = "Provenance"
#: The final product a user receives, judged sentence by sentence.
DELIVERED_SHEET_NAME = "Delivered"
YES_NO = frozenset({"yes", "no"})
#: Mirrors ``export_review_sheet.py``'s own ``needs_review_reason`` value for a guard-demoted
#: exclusion -- kept as a literal constant rather than
#: imported, since it names a data value, not a shared function or class the two modules must
#: agree on the identity of.
UNANCHORED_EXCLUDE_REASON = "unanchored_exclude"


class ReviewSheetError(ValueError):
    """A filled Review or Screening sheet fails a structural check: a blank required human
    field, an out-of-vocabulary value, a yes/no answer that contradicts its own verdict, an
    ``item_id`` absent from the key, or a workbook/key id mismatch."""


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def _read_sheet_rows(wb: Any, sheet_name: str) -> list[dict[str, Any]]:
    ws = wb[sheet_name]
    rows_iter = ws.iter_rows(values_only=True)
    header = list(next(rows_iter))
    out: list[dict[str, Any]] = []
    for row_number, values in enumerate(rows_iter, start=2):
        if all(v is None for v in values):
            continue
        row = {header[i]: values[i] for i in range(len(header))}
        row["_row_number"] = row_number
        out.append(row)
    return out


def read_review_sheet(path: Path) -> list[dict[str, Any]]:
    wb = openpyxl.load_workbook(path, data_only=True)
    if REVIEW_SHEET_NAME not in wb.sheetnames:
        raise ReviewSheetError(f"{path}: no {REVIEW_SHEET_NAME!r} sheet")
    return _read_sheet_rows(wb, REVIEW_SHEET_NAME)


def read_model_labels_sheet(path: Path) -> dict[str, dict[str, Any]] | None:
    """``{item_id: row}`` from the ``Model_labels`` sheet, or ``None`` when the workbook has
    none (exported without ``--annotation``)."""
    wb = openpyxl.load_workbook(path, data_only=True)
    if MODEL_LABELS_SHEET_NAME not in wb.sheetnames:
        return None
    rows = _read_sheet_rows(wb, MODEL_LABELS_SHEET_NAME)
    return {str(r.get("item_id")): r for r in rows}


def read_screening_sheet(path: Path) -> list[dict[str, Any]] | None:
    wb = openpyxl.load_workbook(path, data_only=True)
    if SCREENING_SHEET_NAME not in wb.sheetnames:
        return None
    return _read_sheet_rows(wb, SCREENING_SHEET_NAME)


def read_delivered_sheet(path: Path) -> list[dict[str, Any]] | None:
    """Rows from the ``Delivered`` sheet, or ``None`` when the workbook has none (exported
    without ``--delivered-evidence``)."""
    wb = openpyxl.load_workbook(path, data_only=True)
    if DELIVERED_SHEET_NAME not in wb.sheetnames:
        return None
    return _read_sheet_rows(wb, DELIVERED_SHEET_NAME)


def read_workbook_id(path: Path) -> str | None:
    wb = openpyxl.load_workbook(path, data_only=True)
    if "Guide" not in wb.sheetnames:
        return None
    prefix = ex.WORKBOOK_ID_PREFIX
    for row in wb["Guide"].iter_rows(values_only=True):
        for value in row:
            if isinstance(value, str) and value.startswith(prefix):
                return value[len(prefix):].split()[0].strip()
    return None


def read_provenance_fields(path: Path) -> dict[str, Any] | None:
    """``{field: value}`` from the ``Provenance`` sheet, or ``None`` when the workbook has none
    (every workbook ``export_review_sheet.py`` writes has one, but a hand-built or damaged
    workbook might not). This is how the scorer reads back the
    ``demo_run_id`` and ``export_time`` an export stamped on the sheet, so a returned copy can
    be checked against the run the caller expects rather than trusted on its file name alone."""
    wb = openpyxl.load_workbook(path, data_only=True)
    if PROVENANCE_SHEET_NAME not in wb.sheetnames:
        return None
    out: dict[str, Any] = {}
    for row in wb[PROVENANCE_SHEET_NAME].iter_rows(min_row=2, values_only=True):
        if row and row[0]:
            out[str(row[0])] = row[1] if len(row) > 1 else None
    return out


def load_key(path: Path) -> dict[str, dict[str, Any]]:
    """``{opaque_id: {real_item_id, set, construction_category, expected_label}}`` from the
    exporter's key csv."""
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["opaque_id"]] = {
                "real_item_id": row.get("real_item_id") or "",
                "set": row.get("set") or "",
                "construction_category": row.get("construction_category") or "",
                "expected_label": [
                    s for s in (row.get("expected_label") or "").split(";") if s
                ],
            }
    return out


def load_key_workbook_id(path: Path) -> str | None:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            return row.get("workbook_id") or None
    return None


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------


def normalise_status(value: str | None) -> str:
    return (value or "").strip().lower().replace(" ", "_")


def normalise_yes_no(value: str | None) -> str:
    return (value or "").strip().lower()


# --------------------------------------------------------------------------------------
# Validation -- Review sheet
# --------------------------------------------------------------------------------------


def _fmt_rows(row_numbers: Sequence[int]) -> str:
    return ", ".join(str(n) for n in row_numbers)


def find_review_blanks(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every problem found in *rows*' human columns, one message per problem class (naming
    every offending row number at once) -- never raises; :func:`validate_review_rows` raises
    with this list joined when it is non-empty."""
    problems: list[str] = []
    empty_verdict, bad_verdict = [], []
    empty_agrees, bad_agrees = [], []
    contradictions = []
    empty_axis, bad_axis = [], []
    empty_quote, empty_justification = [], []

    for r in rows:
        row_no = r["_row_number"]
        verdict = normalise_status(r.get("human_verdict"))
        agrees = normalise_yes_no(r.get("human_agrees_with_verifier"))
        verifier_status = normalise_status(r.get("verifier_status"))

        if not verdict:
            empty_verdict.append(row_no)
        elif verdict not in ex.HUMAN_VERDICT_CHOICES:
            bad_verdict.append(row_no)

        if not agrees:
            empty_agrees.append(row_no)
        elif agrees not in YES_NO:
            bad_agrees.append(row_no)

        if verdict and agrees in YES_NO:
            if agrees == "yes" and verdict != verifier_status:
                contradictions.append(row_no)
            elif agrees == "no" and verdict == verifier_status:
                contradictions.append(row_no)

        if agrees == "no":
            axis = str(r.get("disagreement_axis") or "").strip()
            if not axis:
                empty_axis.append(row_no)
            elif axis not in ex.DISAGREEMENT_AXIS_CHOICES:
                bad_axis.append(row_no)
            if not str(r.get("human_quote") or "").strip():
                empty_quote.append(row_no)
            if not str(r.get("human_justification") or "").strip():
                empty_justification.append(row_no)

    if empty_verdict:
        problems.append(f"empty human_verdict at Review row(s): {_fmt_rows(empty_verdict)}")
    if bad_verdict:
        problems.append(
            f"human_verdict must be one of {ex.HUMAN_VERDICT_CHOICES} at Review row(s): "
            f"{_fmt_rows(bad_verdict)}"
        )
    if empty_agrees:
        problems.append(
            f"empty human_agrees_with_verifier at Review row(s): {_fmt_rows(empty_agrees)}"
        )
    if bad_agrees:
        problems.append(
            f"human_agrees_with_verifier must be yes or no at Review row(s): "
            f"{_fmt_rows(bad_agrees)}"
        )
    if contradictions:
        problems.append(
            "human_agrees_with_verifier contradicts human_verdict versus verifier_status at "
            f"Review row(s): {_fmt_rows(contradictions)}"
        )
    if empty_axis:
        problems.append(
            f"empty disagreement_axis on a disagreeing row at Review row(s): "
            f"{_fmt_rows(empty_axis)}"
        )
    if bad_axis:
        problems.append(
            f"disagreement_axis must be one of {ex.DISAGREEMENT_AXIS_CHOICES} at Review "
            f"row(s): {_fmt_rows(bad_axis)}"
        )
    if empty_quote:
        problems.append(
            f"empty human_quote on a disagreeing row at Review row(s): {_fmt_rows(empty_quote)}"
        )
    if empty_justification:
        problems.append(
            "empty human_justification on a disagreeing row at Review row(s): "
            f"{_fmt_rows(empty_justification)}"
        )
    return problems


def validate_review_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    problems = find_review_blanks(rows)
    if problems:
        raise ReviewSheetError("; ".join(problems))


def validate_key_membership(
    rows: Sequence[Mapping[str, Any]], key: Mapping[str, Mapping[str, Any]],
) -> None:
    """Refuses unless the set of ``item_id`` values on the Review sheet is exactly the set of
    ``opaque_id`` values in the key -- checked in both directions, and named separately, so a
    row a reviewer added or renamed (unknown to the key) and a row a reviewer deleted or
    blanked (a key item with nothing on the sheet) are never mistaken for each other, and
    neither can silently shrink the scored denominator (a deleted/blanked Review row is
    skipped, not read, by :func:`_read_sheet_rows`, so its absence would otherwise pass
    unnoticed). A duplicated ``item_id`` is refused too, since it can mask a missing one
    behind an equal-sized set without changing the set itself."""
    row_ids = [str(r.get("item_id")) for r in rows]
    row_id_set = set(row_ids)
    key_id_set = set(key.keys())

    unknown_rows = [r["_row_number"] for r in rows if r.get("item_id") not in key]
    missing_from_sheet = sorted(key_id_set - row_id_set)
    duplicated = sorted(k for k, n in Counter(row_ids).items() if n > 1)

    problems: list[str] = []
    if unknown_rows:
        problems.append(
            f"item_id not present in the key file at Review row(s): {_fmt_rows(unknown_rows)} "
            "(this usually means the wrong --key file was given for this workbook)"
        )
    if missing_from_sheet:
        problems.append(
            f"the key file holds {len(key_id_set)} item(s) but the Review sheet has "
            f"{len(row_id_set)} distinct item_id(s); key item(s) with no row on the sheet "
            f"(deleted or blanked?): {', '.join(missing_from_sheet)}"
        )
    if duplicated:
        problems.append(
            f"item_id appears on more than one Review row: {', '.join(duplicated)}"
        )
    if problems:
        raise ReviewSheetError("; ".join(problems))


def validate_workbook_binding(
    review_workbook_id: str | None, key_workbook_id: str | None,
) -> None:
    if review_workbook_id is None and key_workbook_id is None:
        return
    if review_workbook_id != key_workbook_id:
        raise ReviewSheetError(
            f"the key file does not match this workbook (workbook id {review_workbook_id!r}, "
            f"key file {key_workbook_id!r}); re-export or use the matching --key file"
        )


def validate_demo_run_id(
    provenance_demo_run_id: str | None, expected: str | None,
) -> None:
    """Refuses unless *provenance_demo_run_id* (the
    workbook's own ``Provenance!demo_run_id``) equals *expected* (``--demo-run-id``) exactly,
    whenever *expected* is given -- a superseded copy of an earlier same-named export otherwise
    scores silently against the wrong run's Screening sample. A no-op when *expected* is
    ``None`` (the flag was not given): the scorer has no opinion on which run a workbook came
    from unless the caller names one.

    This check alone cannot tell two exports of the *same*
    demo run apart -- both carry the same ``demo_run_id`` and the same workbook id, since
    both depend only on the kept item order and the seed, not on when the export ran. See
    :func:`validate_export_time` for the check that does."""
    if expected is None:
        return
    if provenance_demo_run_id != expected:
        raise ReviewSheetError(
            f"this workbook's Provenance demo_run_id is {provenance_demo_run_id!r}, expected "
            f"{expected!r} (--demo-run-id); refusing to score a workbook exported against a "
            "different demo run than the one you named -- this usually means a superseded "
            "copy of an earlier same-named export was sent back instead of this one"
        )


def validate_export_time(
    provenance_export_time: str | None, expected: str | None,
) -> None:
    """Refuses unless *provenance_export_time* (the
    workbook's own ``Provenance!export_time``) equals *expected* (``--export-time``) exactly,
    whenever *expected* is given. Two exports of the same demo run share the same
    ``demo_run_id`` and the same workbook id, so :func:`validate_demo_run_id` alone cannot
    tell a superseded copy of an earlier export of that same run apart from the current one;
    ``export_time`` is the one Provenance value that does. A no-op when *expected* is
    ``None``: the scorer has no opinion on when a workbook was exported unless the caller
    names a time."""
    if expected is None:
        return
    if provenance_export_time != expected:
        raise ReviewSheetError(
            f"this workbook's Provenance export_time is {provenance_export_time!r}, expected "
            f"{expected!r} (--export-time); refusing to score a workbook exported at a "
            "different time than the one you named -- this usually means a superseded copy "
            "of an earlier export of the same demo run was sent back instead of this one"
        )


# --------------------------------------------------------------------------------------
# Validation -- Screening sheet
# --------------------------------------------------------------------------------------


def find_screening_blanks(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    empty_status, bad_status = [], []
    empty_agrees, bad_agrees = [], []
    empty_note = []
    empty_quote_verbatim, bad_quote_verbatim = [], []
    empty_criterion_right, bad_criterion_right = [], []
    for r in rows:
        row_no = r["_row_number"]
        status = normalise_status(r.get("human_status"))
        agrees = normalise_yes_no(r.get("human_agrees"))
        if not status:
            empty_status.append(row_no)
        elif status not in ex.SCREENING_HUMAN_STATUS_CHOICES:
            bad_status.append(row_no)
        if not agrees:
            empty_agrees.append(row_no)
        elif agrees not in YES_NO:
            bad_agrees.append(row_no)
        if agrees == "no" and not str(r.get("human_note") or "").strip():
            empty_note.append(row_no)

        # Required whenever the row actually carries a quote/criterion for
        # the human to check -- not on every row, since a census row with neither (e.g. the
        # excluded_missing_anchor stratum's own point) has nothing to confirm as verbatim or
        # on point.
        if str(r.get("quote") or "").strip():
            qv = normalise_yes_no(r.get("quote_is_verbatim"))
            if not qv:
                empty_quote_verbatim.append(row_no)
            elif qv not in YES_NO:
                bad_quote_verbatim.append(row_no)
        if str(r.get("criterion") or "").strip():
            cr = normalise_yes_no(r.get("criterion_is_right"))
            if not cr:
                empty_criterion_right.append(row_no)
            elif cr not in YES_NO:
                bad_criterion_right.append(row_no)
    if empty_status:
        problems.append(f"empty human_status at Screening row(s): {_fmt_rows(empty_status)}")
    if bad_status:
        problems.append(
            f"human_status must be one of {ex.SCREENING_HUMAN_STATUS_CHOICES} at Screening "
            f"row(s): {_fmt_rows(bad_status)}"
        )
    if empty_agrees:
        problems.append(f"empty human_agrees at Screening row(s): {_fmt_rows(empty_agrees)}")
    if bad_agrees:
        problems.append(
            f"human_agrees must be yes or no at Screening row(s): {_fmt_rows(bad_agrees)}"
        )
    if empty_note:
        problems.append(
            f"empty human_note on a disagreeing row at Screening row(s): {_fmt_rows(empty_note)}"
        )
    if empty_quote_verbatim:
        problems.append(
            f"empty quote_is_verbatim on a row with a quote at Screening row(s): "
            f"{_fmt_rows(empty_quote_verbatim)}"
        )
    if bad_quote_verbatim:
        problems.append(
            f"quote_is_verbatim must be yes or no at Screening row(s): "
            f"{_fmt_rows(bad_quote_verbatim)}"
        )
    if empty_criterion_right:
        problems.append(
            f"empty criterion_is_right on a row with a criterion at Screening row(s): "
            f"{_fmt_rows(empty_criterion_right)}"
        )
    if bad_criterion_right:
        problems.append(
            f"criterion_is_right must be yes or no at Screening row(s): "
            f"{_fmt_rows(bad_criterion_right)}"
        )
    return problems


def validate_screening_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    problems = find_screening_blanks(rows)
    if problems:
        raise ReviewSheetError("; ".join(problems))


#: The Screening sheet's human-filled columns (mirrors ``export_review_sheet.py``'s own
#: ``screening_header`` human block) -- the columns :func:`screening_sheet_is_blank` checks to
#: tell an entirely unfilled sheet apart from a partly filled one: the author may
#: return a workbook whose Screening sheet was never touched, and before this the only
#: response was the same refusal a half-filled sheet gets, naming every row as if the human had
#: started and left gaps rather than never started at all.
SCREENING_HUMAN_COLUMNS = (
    "human_status", "human_agrees", "quote_is_verbatim", "criterion_is_right", "human_note",
)


def screening_sheet_is_blank(rows: Sequence[Mapping[str, Any]]) -> bool:
    """True when *rows* is empty, or every :data:`SCREENING_HUMAN_COLUMNS` cell is blank on
    every row -- the sheet was exported but never filled in. A sheet with even one non-blank
    human cell is not blank by this test, so it still goes through :func:`validate_screening_rows`
    and refuses in the usual way if any *other* row is left incomplete."""
    if not rows:
        return True
    return all(
        not str(r.get(col) or "").strip()
        for r in rows
        for col in SCREENING_HUMAN_COLUMNS
    )


# --------------------------------------------------------------------------------------
# Validation -- Delivered sheet
# --------------------------------------------------------------------------------------


def find_delivered_blanks(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every problem found in *rows*' human columns, one message per problem class -- never
    raises; :func:`validate_delivered_rows` raises with this list joined when it is non-empty.

    ``human_sentence_correct`` and ``human_quote_supports`` are required (yes/no) on every
    row: unlike the Review sheet's confirm-or-override shape, every Delivered row is an
    independent judgement of the final product, with no "agrees" branch to make either one
    conditional on. ``human_note`` is required only when either of those two answers is
    ``no`` -- mirroring the Review sheet's own ``human_justification`` and the Screening
    sheet's own ``human_note``, both required only on a disagreeing row."""
    problems: list[str] = []
    empty_sentence_correct, bad_sentence_correct = [], []
    empty_quote_supports, bad_quote_supports = [], []
    empty_note = []

    for r in rows:
        row_no = r["_row_number"]
        sentence_correct = normalise_yes_no(r.get("human_sentence_correct"))
        quote_supports = normalise_yes_no(r.get("human_quote_supports"))

        if not sentence_correct:
            empty_sentence_correct.append(row_no)
        elif sentence_correct not in YES_NO:
            bad_sentence_correct.append(row_no)

        if not quote_supports:
            empty_quote_supports.append(row_no)
        elif quote_supports not in YES_NO:
            bad_quote_supports.append(row_no)

        if (sentence_correct == "no" or quote_supports == "no") and not str(
            r.get("human_note") or ""
        ).strip():
            empty_note.append(row_no)

    if empty_sentence_correct:
        problems.append(
            f"empty human_sentence_correct at Delivered row(s): "
            f"{_fmt_rows(empty_sentence_correct)}"
        )
    if bad_sentence_correct:
        problems.append(
            f"human_sentence_correct must be yes or no at Delivered row(s): "
            f"{_fmt_rows(bad_sentence_correct)}"
        )
    if empty_quote_supports:
        problems.append(
            f"empty human_quote_supports at Delivered row(s): "
            f"{_fmt_rows(empty_quote_supports)}"
        )
    if bad_quote_supports:
        problems.append(
            f"human_quote_supports must be yes or no at Delivered row(s): "
            f"{_fmt_rows(bad_quote_supports)}"
        )
    if empty_note:
        problems.append(
            f"empty human_note on a row judged no at Delivered row(s): {_fmt_rows(empty_note)}"
        )
    return problems


def validate_delivered_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    problems = find_delivered_blanks(rows)
    if problems:
        raise ReviewSheetError("; ".join(problems))


# --------------------------------------------------------------------------------------
# Scoring -- Review sheet
# --------------------------------------------------------------------------------------


def _accuracy_block(gold: Sequence[str], pred: Sequence[str]) -> dict[str, Any]:
    n = len(gold)
    correct = sum(1 for g, p in zip(gold, pred, strict=True) if g == p)
    lo, hi = wilson_interval(correct, n) if n else (None, None)
    return {
        "n": n, "n_correct": correct, "accuracy": (correct / n) if n else None,
        "wilson_95": {"lo": lo, "hi": hi},
    }


def _enrich_review(
    rows: Sequence[Mapping[str, Any]], key: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        k = key.get(r.get("item_id"), {})
        out.append({
            **r,
            "real_item_id": k.get("real_item_id") or "",
            "set": k.get("set") or "unknown",
            "verifier_status_n": normalise_status(r.get("verifier_status")),
            "human_verdict_n": normalise_status(r.get("human_verdict")),
            "agrees_n": normalise_yes_no(r.get("human_agrees_with_verifier")),
        })
    return out


def _withhold_no_full_text(
    subset: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int]:
    """*subset* filtered to rows whose ``verifier_status_n`` is not ``no_full_text``, plus how
    many were dropped -- the one withheld-row rule :func:`agreement_with_verifier` and
    :func:`agreement_with_model_labels` both apply, pulled out
    once so ``overall``, ``per_set`` and the model-labels block cannot silently drift apart on
    which rows they call withheld."""
    judged = [e for e in subset if e["verifier_status_n"] != "no_full_text"]
    return judged, len(subset) - len(judged)


def agreement_with_verifier(enriched: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Accuracy of ``human_verdict`` against ``verifier_status``, overall, per ``set``
    (constructed/real) and per ``verifier_status`` value.

    The headline ``overall`` and ``kappa``, and each ``per_set`` block, are computed over the
    judged rows only. A row whose ``verifier_status`` is ``no_full_text`` is withheld by
    instruction: the Guide tells the reviewer to answer ``no_full_text`` on that row without
    looking further, so ``human_verdict`` there is a guaranteed agreement rather than a
    judgement, and folding it into either number would inflate it by construction rather than
    by any human agreement. This mirrors ``score_screening``'s exclusion of an unscreened row
    from its own headline. ``n_items``, ``n_judged`` and ``n_withheld`` make the exclusion
    visible at the top level and on every ``per_set``
    block too (a per-set ``"n"`` is therefore the judged count for that set, matching the
    headline's own convention, not the set's total row count). ``per_status`` gives its
    ``no_full_text`` bucket the same treatment
    ``score_screening``'s own ``per_status`` gives a guard-demoted ``needs_review`` bucket: the
    withheld rows are excluded from that status's own accuracy block (every row of the
    ``no_full_text`` bucket is withheld by definition, so its main block's ``"n"`` is 0) and
    reported nested under a ``"withheld"`` key, alongside an ``"n_withheld"`` count on every
    status block (0 on the three judged statuses). Before this, the whole ``no_full_text``
    bucket surfaced as a trivial 1.000 accuracy row -- correct, since a dictated answer always
    matches, but indistinguishable in the json from a status where every answer was a genuine
    human judgement."""
    judged, n_withheld = _withhold_no_full_text(enriched)
    overall = _accuracy_block(
        [e["verifier_status_n"] for e in judged], [e["human_verdict_n"] for e in judged],
    )
    per_set = {}
    for s in sorted({e["set"] for e in enriched}):
        subset = [e for e in enriched if e["set"] == s]
        subset_judged, subset_withheld = _withhold_no_full_text(subset)
        block = _accuracy_block(
            [e["verifier_status_n"] for e in subset_judged],
            [e["human_verdict_n"] for e in subset_judged],
        )
        block["n_items"] = len(subset)
        block["n_judged"] = len(subset_judged)
        block["n_withheld"] = subset_withheld
        per_set[s] = block
    per_status = {}
    for status in sorted({e["verifier_status_n"] for e in enriched if e["verifier_status_n"]}):
        subset = [e for e in enriched if e["verifier_status_n"] == status]
        subset_judged, subset_withheld = _withhold_no_full_text(subset)
        block = _accuracy_block(
            [e["verifier_status_n"] for e in subset_judged],
            [e["human_verdict_n"] for e in subset_judged],
        )
        block["n_withheld"] = subset_withheld
        if subset_withheld:
            # subset is homogeneous by status (the loop groups by it), and the only status
            # _withhold_no_full_text ever drops a row from is "no_full_text" itself, so a
            # non-zero count here means every row of subset was withheld -- the withheld rows
            # are simply subset unfiltered, not a partial slice of it.
            block["withheld"] = _accuracy_block(
                [e["verifier_status_n"] for e in subset], [e["human_verdict_n"] for e in subset],
            )
        per_status[status] = block
    kappa = cohens_kappa(
        [e["verifier_status_n"] for e in judged], [e["human_verdict_n"] for e in judged],
    )
    return {
        "n_items": len(enriched), "n_judged": len(judged), "n_withheld": n_withheld,
        "overall": overall, "per_set": per_set, "per_status": per_status, "kappa": kappa,
    }


def agreement_with_model_labels(
    enriched: Sequence[Mapping[str, Any]], model_labels: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    """Accuracy and kappa of ``human_verdict`` against the ``Model_labels`` sheet's
    ``model_annotation_label``, restricted to rows the sheet actually covers. ``None`` when the
    workbook has no ``Model_labels`` sheet.

    Mirrors :func:`agreement_with_verifier`'s withheld-row exclusion: a row whose
    ``verifier_status`` is ``no_full_text`` is dropped from the
    accuracy/kappa computation the same way, since the Guide dictates that row's
    ``human_verdict`` too, and every withheld row's own ``model_annotation_label`` is also
    ``no_full_text`` (the Model_labels sheet was built from the same instruction), so leaving
    them in would inflate this number by construction as well. ``n_items``, ``n_judged`` and
    ``n_withheld`` make the exclusion visible the same way."""
    if model_labels is None:
        return None
    paired = [
        (e, model_labels[e["item_id"]]) for e in enriched if e.get("item_id") in model_labels
    ]
    paired_judged = [
        (e, m) for e, m in paired if e["verifier_status_n"] != "no_full_text"
    ]
    gold = [normalise_status(m.get("model_annotation_label")) for _, m in paired_judged]
    pred = [e["human_verdict_n"] for e, _ in paired_judged]
    block = _accuracy_block(gold, pred)
    block["kappa"] = cohens_kappa(gold, pred)
    block["n_items"] = len(paired)
    block["n_judged"] = len(paired_judged)
    block["n_withheld"] = len(paired) - len(paired_judged)
    return block


def disagreement_axis_counts(enriched: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(
        str(e.get("disagreement_axis") or "").strip()
        for e in enriched if e["agrees_n"] == "no"
    )
    counter.pop("", None)
    return dict(sorted(counter.items()))


def overridden_rows(enriched: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "real_item_id": e.get("real_item_id"), "verifier_status": e.get("verifier_status"),
            "human_verdict": e.get("human_verdict"),
            "disagreement_axis": e.get("disagreement_axis") or "",
            "human_justification": e.get("human_justification") or "",
        }
        for e in enriched if e["agrees_n"] == "no"
    ]


def score(
    rows: Sequence[Mapping[str, Any]],
    key: Mapping[str, Mapping[str, Any]],
    *,
    model_labels: Mapping[str, Mapping[str, Any]] | None = None,
    review_workbook_id: str | None = None,
    key_workbook_id: str | None = None,
) -> dict[str, Any]:
    validate_review_rows(rows)
    validate_key_membership(rows, key)
    validate_workbook_binding(review_workbook_id, key_workbook_id)
    enriched = _enrich_review(rows, key)
    return {
        "n_items": len(enriched),
        "agreement_with_verifier": agreement_with_verifier(enriched),
        "agreement_with_model_labels": agreement_with_model_labels(enriched, model_labels),
        "disagreement_axis_counts": disagreement_axis_counts(enriched),
        "overridden_rows": overridden_rows(enriched),
    }


# --------------------------------------------------------------------------------------
# Scoring -- Delivered sheet
# --------------------------------------------------------------------------------------


def _delivered_yes_no_block(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, Any]:
    """``{n, n_yes, rate, wilson_95}`` over every row of *rows* -- unlike
    :func:`_yes_no_rate` (the Screening sheet's own helper, which drops a row where the
    question does not apply), every Delivered row always carries both yes/no columns once
    :func:`validate_delivered_rows` has passed, so *n* here is simply ``len(rows)``."""
    n = len(rows)
    n_yes = sum(1 for r in rows if normalise_yes_no(r.get(column)) == "yes")
    lo, hi = wilson_interval(n_yes, n) if n else (None, None)
    return {"n": n, "n_yes": n_yes, "rate": (n_yes / n) if n else None,
            "wilson_95": {"lo": lo, "hi": hi}}


#: A single demo run can deliver rows from more than one generated section into one
#: Delivered sheet. A row with no ``section_title`` at all (blank, or an earlier
#: ``delivered_evidence.json`` that never carried the field) is grouped under this label rather
#: than under an empty string, so the per-section table always shows a named section.
UNRECORDED_SECTION_LABEL = "(section not recorded)"


def _delivered_section_label(row: Mapping[str, Any]) -> str:
    return str(row.get("section_title") or "").strip() or UNRECORDED_SECTION_LABEL


def delivered_section_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Distinct ``section_title`` values across *rows*, in first-seen order (a row with no
    section_title is grouped under :data:`UNRECORDED_SECTION_LABEL`) -- the order
    :func:`score_delivered`'s own ``by_section`` table is built and reported in, so a
    one-section file reads as exactly one named section rather than an unordered set."""
    order: list[str] = []
    seen: set[str] = set()
    for r in rows:
        label = _delivered_section_label(r)
        if label not in seen:
            seen.add(label)
            order.append(label)
    return order


def delivered_rows_judged_no(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every row where the human judged either question ``no`` -- the sentences and quotes
    worth a second look, in the same "list the disagreements on their own" shape
    :func:`overridden_rows` and :func:`score_screening`'s own note-taking already use.

    Carries its own ``section`` (:func:`_delivered_section_label`):
    ``row`` restarts at 1 in every section, so a multi-section sheet's own ``row`` value
    alone does not identify a row across the whole sheet -- the section label does."""
    return [
        {
            "row": r.get("row"), "section": _delivered_section_label(r),
            "sentence": r.get("sentence"),
            "claim_checked": r.get("claim_checked") or "",
            "human_sentence_correct": r.get("human_sentence_correct"),
            "human_quote_supports": r.get("human_quote_supports"),
            "human_note": r.get("human_note") or "",
        }
        for r in rows
        if normalise_yes_no(r.get("human_sentence_correct")) == "no"
        or normalise_yes_no(r.get("human_quote_supports")) == "no"
    ]


def _delivered_fidelity_failed(row: Mapping[str, Any]) -> bool:
    """A row's own ``source_located``, ``passage_located`` or ``sentence_in_draft`` flags
    say no -- an explicit no only; a blank cell (the field never recorded, e.g. a
    ``delivered_evidence.json`` written before the field existed) is not treated as a
    failure. ``passage_located`` is the flag that actually
    records whether a located source passage is shown: ``source_located`` alone is true
    whenever the named chunk was fetched, even on a row whose evidence quote could not be
    found anywhere in it (``source_passage`` null or only partly covering
    ``evidence_quotes``), which is exactly the row this check must not let through into
    the headline."""
    return (
        normalise_yes_no(row.get("source_located")) == "no"
        or normalise_yes_no(row.get("passage_located")) == "no"
        or normalise_yes_no(row.get("sentence_in_draft")) == "no"
    )


def delivered_fidelity_excluded_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Every row :func:`_delivered_fidelity_failed` flags -- not a full, source-verified
    reading of the sentence the user actually received, so listed here rather than
    silently folded into the headline rate either way.

    Carries its own ``section`` (:func:`_delivered_section_label`), for the same reason
    :func:`delivered_rows_judged_no` does: ``row`` restarts at 1 in
    every section, so it alone does not identify a row across a multi-section sheet."""
    return [
        {
            "row": r.get("row"), "section": _delivered_section_label(r),
            "sentence": r.get("sentence"),
            "claim_checked": r.get("claim_checked") or "",
            "source_located": r.get("source_located"),
            "passage_located": r.get("passage_located"),
            "unlocated_quotes": r.get("unlocated_quotes"),
            "sentence_in_draft": r.get("sentence_in_draft"),
        }
        for r in rows
        if _delivered_fidelity_failed(r)
    ]


def delivered_location_mismatch_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Every row whose own ``location_section_mismatch`` is yes -- the fetched chunk's own
    section disagreed with the section ``evidence_location`` recorded, the cheapest
    available sign that the passage shown may not be from the chunk the claim was
    verified against. Listed beside
    ``fidelity_excluded_rows`` rather than folded into the same exclusion: this is new
    surface, not an unclosed requirement, so a flagged row still counts toward the
    headline unless one of :func:`_delivered_fidelity_failed`'s own flags also excludes
    it.

    Carries its own ``section`` (:func:`_delivered_section_label`),
    for the same reason :func:`delivered_rows_judged_no` and
    :func:`delivered_fidelity_excluded_rows` already do: ``row`` restarts at 1 in every
    section, so it alone does not identify a row across a multi-section sheet."""
    return [
        {"row": r.get("row"), "section": _delivered_section_label(r), "sentence": r.get("sentence")}
        for r in rows
        if normalise_yes_no(r.get("location_section_mismatch")) == "yes"
    ]


def _score_delivered_section(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The same headline shape :func:`score_delivered` reports overall, computed over one
    section's own rows only (already-validated rows: the caller runs
    :func:`validate_delivered_rows` once, over every row, before splitting by section)."""
    n_rows = len(rows)
    excluded_rows = delivered_fidelity_excluded_rows(rows)
    headline_rows = [r for r in rows if not _delivered_fidelity_failed(r)]
    sentence_block = _delivered_yes_no_block(headline_rows, "human_sentence_correct")
    quote_block = _delivered_yes_no_block(headline_rows, "human_quote_supports")
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
    }


def delivered_sentence_groups(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[tuple[Any, Any], list[Mapping[str, Any]]]]:
    """The judged rows (those :func:`_delivered_fidelity_failed` does not flag), grouped by
    ``(draft_id, sentence)``, in first-seen order. A sentence whose every row is
    fidelity-excluded has no group at all here, so it is not counted in either direction of
    the per-sentence block."""
    judged = [r for r in rows if not _delivered_fidelity_failed(r)]
    order: list[tuple[Any, Any]] = []
    groups: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
    for r in judged:
        key = (r.get("draft_id"), r.get("sentence"))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    return [(key, groups[key]) for key in order]


def _delivered_sentence_yes_no_block(
    groups: Sequence[tuple[Any, list[Mapping[str, Any]]]], column: str,
) -> dict[str, Any]:
    """``{n, n_yes, rate, wilson_95}`` over *groups* -- a group counts yes only when every row
    in it is yes for *column* (one no row makes the whole sentence no), in the same shape
    :func:`_delivered_yes_no_block` returns for rows."""
    n = len(groups)
    n_yes = sum(
        1 for _key, group_rows in groups
        if all(normalise_yes_no(r.get(column)) == "yes" for r in group_rows)
    )
    lo, hi = wilson_interval(n_yes, n) if n else (None, None)
    return {"n": n, "n_yes": n_yes, "rate": (n_yes / n) if n else None,
            "wilson_95": {"lo": lo, "hi": hi}}


def score_delivered_by_sentence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The secondary, per-sentence view of the Delivered sheet: a sentence counts as faithful
    only when every one of its judged rows is yes. Returns
    ``{"n_sentences": n, "sentence_correct": {...}, "quote_supports": {...}}``, reported under
    the headline (:func:`score_delivered`'s own row-level keys), not in place of it."""
    groups = delivered_sentence_groups(rows)
    return {
        "n_sentences": len(groups),
        "sentence_correct": _delivered_sentence_yes_no_block(groups, "human_sentence_correct"),
        "quote_supports": _delivered_sentence_yes_no_block(groups, "human_quote_supports"),
    }


def score_delivered(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The product-level headline: how often the final delivered sentence states what its
    source states, and how often the quote the verifier found actually supports it, over
    every row of the Delivered sheet -- the sheet carries no construction category or key, so
    there is nothing to join and nothing to blind. Refuses (:class:`ReviewSheetError`) on any
    blank required human cell (:func:`validate_delivered_rows`); every row is still judged,
    but a row :func:`_delivered_fidelity_failed` flags is not a full, source-verified reading
    of the delivered sentence, and is excluded from the headline rates and from
    ``rows_judged_no``, reported instead in ``fidelity_excluded_rows``. ``n_judged`` is the
    count the headline rates are computed over (``n_rows`` minus ``n_fidelity_excluded``), not
    simply ``n_rows`` as it was before that fix. Headline rows are selected by testing each
    row object itself against :func:`_delivered_fidelity_failed`, not by round-tripping
    through the sheet's own ``row`` cell: a workbook filled
    in and returned by a human has no protected cell, so a duplicated or blank ``row`` value
    must not silently drop an unrelated, unflagged row from the headline. A row whose
    ``location_section_mismatch`` is yes is listed in ``location_mismatch_rows`` but not
    excluded on that basis alone.

    A single demo run can deliver rows from more than one generated section into one
    Delivered sheet. The headline above stays over every row regardless of section; ``n_sections``
    and ``by_section`` (:func:`delivered_section_order`'s own labels, each mapped to the same
    headline shape computed over that section's rows alone, via :func:`_score_delivered_section`)
    report the same numbers broken down by section, so a low or high overall rate cannot hide one
    section performing very differently from the others."""
    validate_delivered_rows(rows)
    n_rows = len(rows)
    excluded_rows = delivered_fidelity_excluded_rows(rows)
    headline_rows = [r for r in rows if not _delivered_fidelity_failed(r)]
    sentence_block = _delivered_yes_no_block(headline_rows, "human_sentence_correct")
    quote_block = _delivered_yes_no_block(headline_rows, "human_quote_supports")
    section_labels = delivered_section_order(rows)
    by_section = {
        label: _score_delivered_section(
            [r for r in rows if _delivered_section_label(r) == label]
        )
        for label in section_labels
    }
    return {
        "n_rows": n_rows,
        "n_judged": len(headline_rows),
        "n_fidelity_excluded": len(excluded_rows),
        "fidelity_excluded_rows": excluded_rows,
        "location_mismatch_rows": delivered_location_mismatch_rows(rows),
        "sentence_correct_count": sentence_block["n_yes"],
        "sentence_correct_rate": sentence_block["rate"],
        "sentence_correct_wilson_95": sentence_block["wilson_95"],
        "quote_supports_count": quote_block["n_yes"],
        "quote_supports_rate": quote_block["rate"],
        "quote_supports_wilson_95": quote_block["wilson_95"],
        "rows_judged_no": delivered_rows_judged_no(headline_rows),
        "n_sections": len(section_labels),
        "by_section": by_section,
        "by_sentence": score_delivered_by_sentence(rows),
    }


# --------------------------------------------------------------------------------------
# Scoring -- Screening sheet
# --------------------------------------------------------------------------------------


def _enrich_screening(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **r,
            "status_n": normalise_status(r.get("status")),
            "human_status_n": normalise_status(r.get("human_status")),
        }
        for r in rows
    ]


def _yes_no_rate(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, Any]:
    """Share of *rows* with a non-blank *column* answered ``yes`` -- a row where the question
    does not apply (the cell is blank because the row carries no quote/criterion to check) is
    excluded from the denominator, not counted against the rate."""
    answered = [
        normalise_yes_no(r.get(column)) for r in rows if str(r.get(column) or "").strip()
    ]
    n = len(answered)
    n_yes = sum(1 for a in answered if a == "yes")
    return {"n": n, "n_yes": n_yes, "rate": (n_yes / n) if n else None}


def _split_unanchored_exclude(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Splits *rows* into ``(main, unanchored_exclude)`` by ``needs_review_reason`` -- the
    guard-demoted exclusions the Guide sheet's own ``UNANCHORED_EXCLUDE_NOTE`` paragraph tells
    the reviewer to record with ``human_status`` ``exclude`` and ``human_agrees`` ``no`` even
    though the tool's own ``status`` is ``needs_review``.
    Folding these rows into :func:`score_screening`'s headline ``overall``/``kappa``, or into
    ``per_status``'s own ``needs_review`` block, would count that dictated answer as a human
    disagreement and
    deflate both by construction -- the same defect m4 already fixed for
    :func:`_quote_is_verbatim_rate` and m7 fixed for the withheld ``no_full_text`` rows in
    :func:`agreement_with_verifier`. These rows are never dropped, only reported separately, in
    the same accuracy-block shape as the numbers they are excluded from."""
    unanchored = [
        r for r in rows
        if normalise_status(r.get("needs_review_reason")) == UNANCHORED_EXCLUDE_REASON
    ]
    unanchored_ids = {id(r) for r in unanchored}
    main = [r for r in rows if id(r) not in unanchored_ids]
    return main, unanchored


def _quote_is_verbatim_rate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``quote_is_verbatim_rate``, excluding any row whose
    ``needs_review_reason`` is ``unanchored_exclude`` from the rate's own denominator:
    a guard demoted that row's status to ``needs_review`` precisely
    because its quote was not anchored, verbatim, in the shown text, so a human who correctly
    answers "no" on it is scoring the guard's own catch, not a quote-fidelity miss, and pooling
    the two caps the rate below what shipped quote fidelity alone could ever achieve. Those
    rows are not dropped -- they are reported on their own, under the ``"unanchored_exclude"``
    key, in the same ``{n, n_yes, rate}`` shape."""
    main_rows = [
        r for r in rows
        if normalise_status(r.get("needs_review_reason")) != UNANCHORED_EXCLUDE_REASON
    ]
    unanchored_rows = [
        r for r in rows
        if normalise_status(r.get("needs_review_reason")) == UNANCHORED_EXCLUDE_REASON
    ]
    return {
        **_yes_no_rate(main_rows, "quote_is_verbatim"),
        "unanchored_exclude": _yes_no_rate(unanchored_rows, "quote_is_verbatim"),
    }


#: The two exclusion strata for the "share of sampled exclusions the
#: human would have screened in" number, read off the exporter's own hidden ``stratum`` column
#: -- not the tool's ``status``/``criterion``, which cannot distinguish a numbered-criterion
#: exclusion from an off-topic one once both are folded to plain "exclude".
SCREEN_IN_SHARE_STRATA = ("excluded_numbered", "excluded_off_topic")
#: A human ``human_status`` of ``include`` or ``needs_review`` both mean the record was not
#: screened out -- the same "screened in" convention this evaluation already uses elsewhere
#: (e.g. recall counting INCLUDE plus NEEDS_REVIEW as screened in): a NEEDS_REVIEW record still
#: proceeds to full-text screening rather than being dropped, so a human who would have called
#: an excluded record NEEDS_REVIEW is disagreeing with the exclusion just as much as one who
#: would have called it INCLUDE.
SCREENED_IN_HUMAN_STATUSES = frozenset({"include", "needs_review"})


def _screen_in_share(enriched: Sequence[Mapping[str, Any]], stratum: str) -> dict[str, Any]:
    """The "share of sampled exclusions the human would have screened in",
    computed separately per *stratum* (one of :data:`SCREEN_IN_SHARE_STRATA`) by reading the
    row's own ``stratum`` column -- the only way to tell a numbered-criterion exclusion from an
    off-topic one apart once scoring, since both carry ``status_n == "exclude"``."""
    subset = [e for e in enriched if str(e.get("stratum") or "") == stratum]
    n = len(subset)
    n_in = sum(1 for e in subset if e["human_status_n"] in SCREENED_IN_HUMAN_STATUSES)
    return {"n": n, "n_screened_in": n_in, "rate": (n_in / n) if n else None}


def score_screening(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Scores the Screening sheet, or reports it as skipped without raising
    when :func:`screening_sheet_is_blank` says every human cell on it is blank -- the sheet was
    exported but the reviewer never touched it, which is not the same failure as a sheet filled
    in on some rows and left incomplete on others (that still raises, below, exactly as
    before). The skip is reported as ``{"screening_skipped": True, "screening_skip_reason":
    ...}``; the caller's other sheets (Review, Delivered) are scored normally either way."""
    if screening_sheet_is_blank(rows):
        return {
            "screening_skipped": True,
            "screening_skip_reason": (
                "every human column on the Screening sheet (human_status, human_agrees, "
                "quote_is_verbatim, criterion_is_right, human_note) was blank on every row; "
                "treated as not provided and not scored, rather than refused"
            ),
        }
    validate_screening_rows(rows)
    enriched = _enrich_screening(rows)
    # The "unscreened" census rows have no human-comparable tool decision at all
    # (the tool never reached one); folding them into the headline accuracy/kappa drags both
    # down by construction, not by any human disagreement.
    # per_status below still reports them on their own row, unaffected.
    not_unscreened = [e for e in enriched if e["status_n"] != "unscreened"]
    n_unscreened = len(enriched) - len(not_unscreened)
    # A row a guard demoted onto needs_review for an unanchored quote/criterion has its own
    # human_status dictated by the Guide (UNANCHORED_EXCLUDE_NOTE), not judged -- the same
    # withheld-row reasoning as the unscreened rows above, so it is excluded from the headline
    # and from needs_review's own per_status block and reported on its own instead.
    scored, unanchored_exclude_rows = _split_unanchored_exclude(not_unscreened)
    n_unanchored_exclude = len(unanchored_exclude_rows)
    overall = _accuracy_block(
        [e["status_n"] for e in scored], [e["human_status_n"] for e in scored],
    )
    overall["unanchored_exclude"] = _accuracy_block(
        [e["status_n"] for e in unanchored_exclude_rows],
        [e["human_status_n"] for e in unanchored_exclude_rows],
    )
    per_status = {}
    for status in sorted({e["status_n"] for e in enriched if e["status_n"]}):
        subset = [e for e in enriched if e["status_n"] == status]
        subset_main, subset_unanchored = _split_unanchored_exclude(subset)
        block = _accuracy_block(
            [e["status_n"] for e in subset_main], [e["human_status_n"] for e in subset_main],
        )
        if subset_unanchored:
            block["unanchored_exclude"] = _accuracy_block(
                [e["status_n"] for e in subset_unanchored],
                [e["human_status_n"] for e in subset_unanchored],
            )
        per_status[status] = block
    # Restricted to actual EXCLUDE decisions: a demoted
    # NEEDS_REVIEW row can also carry a non-blank criterion, and pooling it with the real
    # exclusions on that criterion silently mixed two different tool decisions into one bucket.
    per_criterion = {}
    for criterion in sorted({
        str(e.get("criterion") or "") for e in enriched if e["status_n"] == "exclude"
    }):
        if not criterion:
            continue
        subset = [
            e for e in enriched if e["status_n"] == "exclude"
            and str(e.get("criterion") or "") == criterion
        ]
        per_criterion[criterion] = _accuracy_block(
            [e["status_n"] for e in subset], [e["human_status_n"] for e in subset],
        )
    kappa = cohens_kappa(
        [e["status_n"] for e in scored], [e["human_status_n"] for e in scored],
    )
    return {
        "n_items": len(enriched), "n_scored": len(scored), "n_unscreened": n_unscreened,
        "n_unanchored_exclude": n_unanchored_exclude,
        "overall": overall, "per_status": per_status, "per_criterion": per_criterion,
        "kappa": kappa,
        # "The share of quotes the human confirms as verbatim and on point"
        # is one of only three screening numbers the paper may report from the workbook --
        # reported as two rates rather than one combined figure, since a quote can be
        # verbatim on a mismatched criterion or vice versa.
        "quote_is_verbatim_rate": _quote_is_verbatim_rate(rows),
        "criterion_is_right_rate": _yes_no_rate(rows, "criterion_is_right"),
        # The other of the three: reported per stratum, numbered-criterion and off-topic
        # separately.
        "screen_in_share": {
            stratum: _screen_in_share(enriched, stratum) for stratum in SCREEN_IN_SHARE_STRATA
        },
        "overridden_records": [
            {
                "record_id": e.get("record_id"), "status": e.get("status"),
                "human_status": e.get("human_status"), "human_note": e.get("human_note") or "",
            }
            for e in enriched if normalise_yes_no(e.get("human_agrees")) == "no"
        ],
    }


# --------------------------------------------------------------------------------------
# Markdown summary
# --------------------------------------------------------------------------------------


def render_markdown_summary(result: Mapping[str, Any]) -> str:
    lines = ["# Review workbook score summary", ""]
    # The Delivered sheet judges the final product a user actually receives, so its
    # own numbers are the headline, printed above the Review/Screening component numbers
    # rather than folded in with them.
    if "delivered" in result:
        d = result["delivered"]
        lines.append("## Delivered (the final product a user receives)")
        lines.append(
            f"n rows: {d['n_rows']} (n judged: {d['n_judged']}, "
            f"{d.get('n_fidelity_excluded', 0)} excluded: source not located, passage not "
            "located, or sentence not in draft)"
        )
        lines.append(
            f"sentence_correct: {fmt(d['sentence_correct_rate'])} "
            f"({d['sentence_correct_count']} of {d['n_judged']}), 95% Wilson "
            f"[{fmt(d['sentence_correct_wilson_95']['lo'])}, "
            f"{fmt(d['sentence_correct_wilson_95']['hi'])}]"
        )
        lines.append(
            f"quote_supports: {fmt(d['quote_supports_rate'])} "
            f"({d['quote_supports_count']} of {d['n_judged']}), 95% Wilson "
            f"[{fmt(d['quote_supports_wilson_95']['lo'])}, "
            f"{fmt(d['quote_supports_wilson_95']['hi'])}]"
        )
        lines.append("")
        # A sentence-level secondary block, reported under the row-level headline above, not
        # in place of it: `.get` so an older cached result json written before this key
        # existed still renders, without the block.
        by_sentence = d.get("by_sentence")
        if by_sentence:
            sc = by_sentence["sentence_correct"]
            qs = by_sentence["quote_supports"]
            lines.append("### Per delivered sentence (secondary)")
            lines.append(
                f"n sentences: {by_sentence['n_sentences']} (a sentence counts as faithful "
                "only when every one of its judged rows is yes)"
            )
            lines.append(
                f"sentence_correct: {fmt(sc['rate'])} ({sc['n_yes']} of {sc['n']}), 95% "
                f"Wilson [{fmt(sc['wilson_95']['lo'])}, {fmt(sc['wilson_95']['hi'])}]"
            )
            lines.append(
                f"quote_supports: {fmt(qs['rate'])} ({qs['n_yes']} of {qs['n']}), 95% "
                f"Wilson [{fmt(qs['wilson_95']['lo'])}, {fmt(qs['wilson_95']['hi'])}]"
            )
            lines.append(
                "Claim granularity is the tool's own: the same content can be one claim in "
                "one sentence and two in another, so the row denominator follows the tool's "
                "split rather than a fixed rule."
            )
            lines.append("")
        # A single demo run can deliver rows from more than one generated section;
        # the headline above stays over every row, this table breaks the same numbers down by
        # section. `.get` throughout: an older result dict (or one re-rendered
        # from a cached json written before this field existed) carries neither key.
        by_section = d.get("by_section") or {}
        if by_section:
            n_sections = d.get("n_sections", len(by_section))
            section_word = "section" if n_sections == 1 else "sections"
            lines.append(f"### Per section ({n_sections} generated {section_word})")
            lines.append(markdown_table(
                ["section", "n_rows", "n_judged", "sentence_correct", "quote_supports"],
                [
                    [label, b["n_rows"], b["n_judged"], fmt(b["sentence_correct_rate"]),
                     fmt(b["quote_supports_rate"])]
                    for label, b in by_section.items()
                ],
            ))
            lines.append("")
        fidelity_excluded_rows = d.get("fidelity_excluded_rows") or []
        if fidelity_excluded_rows:
            lines.append(
                f"{len(fidelity_excluded_rows)} row(s) excluded from the headline above "
                "(source not located, passage not located, or sentence not in draft):"
            )
            lines.append(markdown_table(
                ["row", "section", "sentence", "claim_checked", "source_located",
                 "passage_located", "unlocated_quotes", "sentence_in_draft"],
                [
                    [r.get("row"), r.get("section"), r.get("sentence"),
                     r.get("claim_checked", ""), r.get("source_located"),
                     r.get("passage_located"), r.get("unlocated_quotes"),
                     r.get("sentence_in_draft")]
                    for r in fidelity_excluded_rows
                ],
            ))
        else:
            lines.append(
                "(no rows excluded for source_located, passage_located or sentence_in_draft)"
            )
        lines.append("")
        location_mismatch_rows = d.get("location_mismatch_rows") or []
        if location_mismatch_rows:
            lines.append(
                f"{len(location_mismatch_rows)} row(s) flagged location_section_mismatch "
                "(listed here, not excluded from the headline above):"
            )
            lines.append(markdown_table(
                ["row", "section", "sentence"],
                [
                    [r.get("row"), r.get("section"), r.get("sentence")]
                    for r in location_mismatch_rows
                ],
            ))
        else:
            lines.append("(no rows flagged location_section_mismatch)")
        lines.append("")
        rows_judged_no = d.get("rows_judged_no") or []
        if rows_judged_no:
            lines.append(
                f"{len(rows_judged_no)} row(s) judged no on at least one question:"
            )
            lines.append(markdown_table(
                ["row", "section", "sentence", "claim_checked", "human_sentence_correct",
                 "human_quote_supports", "human_note"],
                [
                    [r.get("row"), r.get("section"), r.get("sentence"),
                     r.get("claim_checked", ""), r.get("human_sentence_correct"),
                     r.get("human_quote_supports"), r.get("human_note")]
                    for r in rows_judged_no
                ],
            ))
        else:
            lines.append("(no rows judged no)")
        lines.append("")
    if "agreement_with_verifier" in result:
        av = result["agreement_with_verifier"]
        lines.append(f"n items: {result.get('n_items')}")
        lines.append("")
        lines.append("## Agreement with the verifier")
        lines.append(
            f"overall accuracy: {fmt(av['overall']['accuracy'])} "
            f"({av['overall']['n_correct']} of {av['overall']['n']}), kappa "
            f"{fmt(av['kappa'])} -- excludes {av.get('n_withheld', 0)} withheld "
            f"no_full_text row(s) (n_items = {av.get('n_items')}, n_judged = "
            f"{av.get('n_judged')}); a withheld row's human_verdict is dictated by the "
            "Guide, not a judgement to score"
        )
        lines.append("")
        lines.append(
            "### Per set (n_judged/accuracy exclude each set's own withheld no_full_text rows)"
        )
        lines.append(markdown_table(
            ["set", "n_items", "n_judged", "n_withheld", "accuracy"],
            [
                [s, b.get("n_items", b["n"]), b["n"], b.get("n_withheld", 0), fmt(b["accuracy"])]
                for s, b in sorted(av["per_set"].items())
            ],
        ))
        lines.append("")
        lines.append(
            "### Per verifier_status (no_full_text excludes its own withheld row(s), "
            "reported separately below the table)"
            if av.get("n_withheld", 0)
            else "### Per verifier_status"
        )
        lines.append(markdown_table(
            ["status", "n", "accuracy"],
            [[s, b["n"], fmt(b["accuracy"])] for s, b in sorted(av["per_status"].items())],
        ))
        no_full_text_withheld = (av["per_status"].get("no_full_text") or {}).get("withheld")
        if no_full_text_withheld:
            lines.append(
                f"no_full_text's own {no_full_text_withheld['n']} withheld row(s), scored on "
                f"their own: {fmt(no_full_text_withheld['accuracy'])} "
                f"({no_full_text_withheld['n_correct']} of {no_full_text_withheld['n']})"
            )
        lines.append("")
        aml = result.get("agreement_with_model_labels")
        if aml is not None:
            lines.append("## Agreement with the model labels")
            lines.append(
                f"accuracy: {fmt(aml['accuracy'])} ({aml['n_correct']} of {aml['n']}), "
                f"kappa {fmt(aml['kappa'])} -- excludes {aml.get('n_withheld', 0)} withheld "
                f"no_full_text row(s) (n_items = {aml.get('n_items')}, n_judged = "
                f"{aml.get('n_judged')}); mirrors the same withheld-row exclusion as the "
                "agreement-with-verifier headline"
            )
            lines.append("")
        lines.append("## Disagreement axis counts")
        counts = result.get("disagreement_axis_counts") or {}
        if counts:
            lines.append(markdown_table(
                ["axis", "n"], [[k, v] for k, v in counts.items()],
            ))
        else:
            lines.append("(no disagreements recorded)")
        lines.append("")
    if "screening" in result:
        sc = result["screening"]
        lines.append("## Screening sheet")
        if sc.get("screening_skipped"):
            lines.append("The Screening sheet was not filled in; screening is not scored.")
            lines.append("")
        else:
            lines.append(
                f"overall accuracy: {fmt(sc['overall']['accuracy'])} "
                f"({sc['overall']['n_correct']} of {sc['overall']['n']}), kappa {fmt(sc['kappa'])} "
                f"-- excludes {sc.get('n_unscreened', 0)} unscreened row(s) and "
                f"{sc.get('n_unanchored_exclude', 0)} guard-demoted unanchored_exclude row(s) "
                f"(n_items = {sc.get('n_items')}, n_scored = {sc.get('n_scored')}); an unscreened "
                "row has no tool decision to agree or disagree with, and an unanchored_exclude "
                "row's human_status is dictated by the Guide, not a judgement to score"
            )
            lines.append("")
            has_unanchored_exclude = bool(sc.get("n_unanchored_exclude", 0))
            lines.append(
                "### Per status (needs_review excludes its own guard-demoted unanchored_exclude "
                "row(s), reported separately below the table)"
                if has_unanchored_exclude
                else "### Per status"
            )
            lines.append(markdown_table(
                ["status", "n", "accuracy"],
                [[s, b["n"], fmt(b["accuracy"])] for s, b in sorted(sc["per_status"].items())],
            ))
            needs_review_unanchored = (
                (sc["per_status"].get("needs_review") or {}).get("unanchored_exclude")
            )
            if needs_review_unanchored:
                lines.append(
                    f"needs_review's own {needs_review_unanchored['n']} guard-demoted "
                    f"unanchored_exclude row(s), scored on their own: "
                    f"{fmt(needs_review_unanchored['accuracy'])} "
                    f"({needs_review_unanchored['n_correct']} of {needs_review_unanchored['n']})"
                )
            lines.append("")
            lines.append("### Per criterion (exclusion rows only)")
            if sc["per_criterion"]:
                lines.append(markdown_table(
                    ["criterion", "n", "accuracy"],
                    [
                        [c, b["n"], fmt(b["accuracy"])]
                        for c, b in sorted(sc["per_criterion"].items())
                    ],
                ))
            else:
                lines.append("(no criterion-bearing exclusion rows sampled)")
            lines.append("")
            qv = sc.get("quote_is_verbatim_rate") or {}
            qv_unanchored = qv.get("unanchored_exclude") or {}
            cr = sc.get("criterion_is_right_rate") or {}
            if has_unanchored_exclude:
                lines.append(
                    f"quote_is_verbatim rate: {fmt(qv.get('rate'))} "
                    f"({qv.get('n_yes', 0)} of {qv.get('n', 0)}) -- excludes "
                    f"{qv_unanchored.get('n', 0)} guard-demoted unanchored_exclude row(s), "
                    f"reported on their own: {fmt(qv_unanchored.get('rate'))} "
                    f"({qv_unanchored.get('n_yes', 0)} of {qv_unanchored.get('n', 0)})"
                )
            else:
                lines.append(
                    f"quote_is_verbatim rate: {fmt(qv.get('rate'))} "
                    f"({qv.get('n_yes', 0)} of {qv.get('n', 0)})"
                )
            lines.append(
                f"criterion_is_right rate: {fmt(cr.get('rate'))} "
                f"({cr.get('n_yes', 0)} of {cr.get('n', 0)})"
            )
            lines.append("")
            lines.append("### Screen-in share of sampled exclusions")
            sis = sc.get("screen_in_share") or {}
            lines.append(markdown_table(
                ["stratum", "n", "n_screened_in", "rate"],
                [
                    [s, v["n"], v["n_screened_in"], fmt(v.get("rate"))]
                    for s, v in sorted(sis.items())
                ],
            ))
            lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--workbook", required=True, type=Path)
    ap.add_argument(
        "--key", type=Path, default=None,
        help="the exporter's key csv; required whenever the workbook has a Review sheet",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-markdown", action="store_true")
    ap.add_argument(
        "--demo-run-id", default=None,
        help="refuses (exit 2, writes no output) unless this workbook's own "
             "Provenance!demo_run_id equals this exactly -- catches a superseded copy of an "
             "earlier same-named export sent back instead of the current one",
    )
    ap.add_argument(
        "--export-time", default=None,
        help="refuses (exit 2, writes no output) unless this workbook's own "
             "Provenance!export_time equals this exactly -- catches a superseded copy of an "
             "earlier export of the same demo run, which --demo-run-id alone cannot "
             "distinguish since both share the same demo_run_id and workbook id",
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    wb = openpyxl.load_workbook(args.workbook, data_only=True)
    has_review = REVIEW_SHEET_NAME in wb.sheetnames
    has_screening = SCREENING_SHEET_NAME in wb.sheetnames
    has_delivered = DELIVERED_SHEET_NAME in wb.sheetnames
    if not has_review and not has_screening and not has_delivered:
        print(
            f"ERROR: {args.workbook}: no {REVIEW_SHEET_NAME!r}, {SCREENING_SHEET_NAME!r} or "
            f"{DELIVERED_SHEET_NAME!r} sheet"
        )
        return 2
    if has_review and args.key is None:
        print(f"ERROR: {args.workbook} has a {REVIEW_SHEET_NAME!r} sheet; --key is required")
        return 2

    provenance = read_provenance_fields(args.workbook)
    try:
        validate_demo_run_id(
            (provenance or {}).get("demo_run_id"), args.demo_run_id,
        )
        validate_export_time(
            (provenance or {}).get("export_time"), args.export_time,
        )
    except ReviewSheetError as exc:
        print(f"ERROR: {exc}")
        return 2

    result: dict[str, Any] = {}
    if provenance is not None:
        result["demo_run_id"] = provenance.get("demo_run_id")
        result["export_time"] = provenance.get("export_time")
    try:
        if has_review:
            key = load_key(args.key)
            rows = read_review_sheet(args.workbook)
            model_labels = read_model_labels_sheet(args.workbook)
            review_workbook_id = read_workbook_id(args.workbook)
            key_workbook_id = load_key_workbook_id(args.key)
            result.update(score(
                rows, key, model_labels=model_labels,
                review_workbook_id=review_workbook_id, key_workbook_id=key_workbook_id,
            ))
        if has_screening:
            screening_rows = read_screening_sheet(args.workbook) or []
            result["screening"] = score_screening(screening_rows)
        if has_delivered:
            delivered_rows = read_delivered_sheet(args.workbook) or []
            result["delivered"] = score_delivered(delivered_rows)
    except ReviewSheetError as exc:
        print(f"ERROR: {exc}")
        return 2

    text = json.dumps(result, indent=2, sort_keys=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    if not args.no_markdown:
        md_path = args.out.with_suffix(".md")
        md_path.write_text(render_markdown_summary(result), encoding="utf-8")
        print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

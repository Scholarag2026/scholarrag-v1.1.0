"""Recompute agreement and Cohen's kappa for the independent blind check of the
false-negative categorisation.

Interpreter: ``evaluation/.venv/Scripts/python`` or plain ``python`` (stdlib only:
csv, pathlib, typing).

Provenance. This directory ships the blind check the categorisation procedure names in
``evaluation/README.md`` and ``screening/results/fn_taxonomy.md``, so that "kappa 1.0" is
regenerable from the deposit rather than only asserted in prose. ``sample_blind.tsv`` is 73
rows (one row per ``<dataset>|<run>|<record_id>``, tab-separated: index, key, the screener's
stated reason text), with the category withheld, drawn stratified by dataset (30 of 446
Nagtegaal_2019 rows, 30 of 464 van_de_Schoot_2017 rows and all 13 Smid_2020 rows, a census);
29 of the 73 were among the 372
rows the first agent had already read during the categorisation pass, and 0 of the 73 were
among the 20 overrides, so the two reads are mostly, but not entirely, independent samples
of the row population. The original draw's random seed was not recorded, so this committed
row list is the frozen sample, not a seed that could regenerate an equivalent draw.
``labels.tsv`` (same key, same row order) is the label a second, independent Claude Code
agent session assigned after reading the reason text alone, blind to the committed category.
Both files were produced on 2026-09-05 by Claude Code agent sessions, not by the authors of
this software; see ``evaluation/README.md`` for the full disclosure, including the rule
engine, the 20 overrides and the seed-20260905 audit that this check does not cover.

:func:`compute` reads the *currently committed* ``results/<dataset>_fn_categories.csv``
files (read-only) and compares them against ``labels.tsv``, so the printed numbers track the
committed category column rather than a frozen snapshot; ``summarize.py`` imports and calls
it to render the false-negative-categories caption. This module never reads or writes
anything under ``evaluation/**/data/`` or any ``*_run?.jsonl`` / ``*.meta.json`` file.

Run directly to print the agreement count and Cohen's kappa::

    evaluation/.venv/Scripts/python evaluation/screening/results/fn_blind_check/fn_blind_check.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent  # .../screening/results/fn_blind_check
RESULTS_DIR = HERE.parent  # .../screening/results
EVAL_ROOT = HERE.parents[2]  # .../evaluation
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from common import cohens_kappa  # noqa: E402

SAMPLE_PATH = HERE / "sample_blind.tsv"
LABELS_PATH = HERE / "labels.tsv"


class BlindCheckResult(NamedTuple):
    n: int
    n_agree: int
    kappa: float | None
    mismatches: tuple[tuple[str, str, str], ...]  # (key, committed_category, independent_label)


def _read_keyed_tsv(path: Path) -> dict[str, str]:
    """Parse a ``<row index>\t<dataset>|<run>|<record_id>\t<value>`` file into
    ``{key: value}``; ``value`` may be empty (a false negative with no reason recorded)."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            _row_index, key, value = line.split("\t", 2)
            out[key] = value
    return out


def committed_categories(results_dir: Path = RESULTS_DIR) -> dict[str, str]:
    """``{"<dataset>|<run>|<record_id>": category}`` from the three committed, already-coded
    ``results/<dataset>_fn_categories.csv`` files (read-only; the ``category`` column is
    never written by this module)."""
    categories: dict[str, str] = {}
    for csv_path in sorted(results_dir.glob("*_fn_categories.csv")):
        with csv_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                key = f"{row['dataset']}|{row['run']}|{row['record_id']}"
                categories[key] = row["category"]
    return categories


def compute(
    sample_path: Path = SAMPLE_PATH,
    labels_path: Path = LABELS_PATH,
    results_dir: Path = RESULTS_DIR,
) -> BlindCheckResult:
    """Recompute agreement between the committed ``category`` column and the independently
    assigned blind label for every row of the blind check."""
    sample_keys = list(_read_keyed_tsv(sample_path))
    labels = _read_keyed_tsv(labels_path)
    if list(labels) != sample_keys:
        raise ValueError("labels.tsv keys/order do not match sample_blind.tsv")
    committed = committed_categories(results_dir)
    missing = [key for key in sample_keys if key not in committed]
    if missing:
        raise KeyError(
            f"{len(missing)} blind-check key(s) not found in the committed FN CSVs "
            f"(first few: {missing[:5]})"
        )
    n = len(sample_keys)
    n_agree = sum(1 for key in sample_keys if committed[key] == labels[key])
    mismatches = tuple(
        (key, committed[key], labels[key]) for key in sample_keys if committed[key] != labels[key]
    )
    kappa = cohens_kappa(
        [committed[key] for key in sample_keys], [labels[key] for key in sample_keys]
    )
    return BlindCheckResult(n=n, n_agree=n_agree, kappa=kappa, mismatches=mismatches)


def main() -> int:
    result = compute()
    print(f"blind check: {result.n_agree}/{result.n} rows agree with the committed category")
    print(f"Cohen's kappa: {result.kappa}")
    if result.mismatches:
        print(f"{len(result.mismatches)} mismatch(es):")
        for key, committed_cat, my_label in result.mismatches:
            print(f"  {key}: committed={committed_cat} independent={my_label}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

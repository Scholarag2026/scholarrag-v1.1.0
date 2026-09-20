"""WoS venue-gate effect for one screening dataset (evaluation E1).

Interpreter: ``evaluation/.venv/Scripts/python`` or the system ``python`` (stdlib only).

ScholarRAG can optionally restrict Stage-1 retrieval to journals listed in the Web of Science
Core Collection. This script measures the recall ceiling such a gate imposes: the share of
*human-included* records (and of all records) whose journal ISSN / eISSN appears in the four
licensed WoS collection CSVs that exist only on the authors' machine at the repository root::

    Science Citation Index Expanded (SCIE).csv
    Social Sciences Citation Index (SSCI).csv
    Arts & Humanities Citation Index (AHCI).csv
    Emerging Sources Citation Index (ESCI).csv

(columns ``Journal title, ISSN, eISSN, Publisher name, Publisher address, Languages,
Web of Science Categories``). The CSVs are read in place and never copied into
``evaluation/``; only aggregate counts are written to ``results/<dataset>_wos_gate.json``.
If none of the CSVs is present the script prints a message and exits with status 0.

The dataset JSONL must carry ``issn`` lists (route ``v2`` of ``fetch_synergy.py`` or route
``v1`` with ``--hydrate``); records without an ISSN are reported as "unresolved" and are
counted in the denominators so the ceiling is not overstated.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    REPO_ROOT,
    load_protocol,
    now_iso,
    read_jsonl,
    resolve_path_args,
    write_json,
)

DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results" / "v3"
PROTOCOLS_DIR = HERE / "protocols"

WOS_FILES: dict[str, str] = {
    "SCIE": "Science Citation Index Expanded (SCIE).csv",
    "SSCI": "Social Sciences Citation Index (SSCI).csv",
    "AHCI": "Arts & Humanities Citation Index (AHCI).csv",
    "ESCI": "Emerging Sources Citation Index (ESCI).csv",
}

_ISSN_RE = re.compile(r"^\d{4}-\d{3}[\dX]$")


def normalise_issn(value: str | None) -> str | None:
    """``'1234567x'`` / ``' 1234-567X '`` -> ``'1234-567X'``; invalid -> ``None``."""
    if not value:
        return None
    s = re.sub(r"[\s-]", "", str(value)).upper()
    if len(s) != 8:
        return None
    s = f"{s[:4]}-{s[4:]}"
    return s if _ISSN_RE.match(s) else None


def parse_wos_issns(csv_text: str) -> set[str]:
    """All normalised ISSN and eISSN values in one WoS collection CSV."""
    issns: set[str] = set()
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        for column in ("ISSN", "eISSN"):
            n = normalise_issn(row.get(column))
            if n:
                issns.add(n)
    return issns


def load_wos_collections(root: Path) -> tuple[dict[str, set[str]], list[str]]:
    """Read whichever collection CSVs exist under ``root``. Returns (issns, missing)."""
    loaded: dict[str, set[str]] = {}
    missing: list[str] = []
    for code, filename in WOS_FILES.items():
        path = root / filename
        if not path.exists():
            missing.append(code)
            continue
        loaded[code] = parse_wos_issns(path.read_text(encoding="utf-8-sig", errors="replace"))
    return loaded, missing


def _record_issns(record: Mapping[str, Any]) -> set[str]:
    return {n for n in (normalise_issn(x) for x in record.get("issn") or []) if n}


def _subset_stats(
    records: Iterable[Mapping[str, Any]], collections: Mapping[str, set[str]]
) -> dict[str, Any]:
    recs = list(records)
    n = len(recs)
    n_with_issn = 0
    any_hits = 0
    per_collection = {code: 0 for code in collections}
    for rec in recs:
        issns = _record_issns(rec)
        if not issns:
            continue
        n_with_issn += 1
        hit_any = False
        for code, table in collections.items():
            if issns & table:
                per_collection[code] += 1
                hit_any = True
        if hit_any:
            any_hits += 1
    return {
        "n": n,
        "n_with_issn": n_with_issn,
        "n_unresolved": n - n_with_issn,
        "n_in_wos_any": any_hits,
        "share_in_wos_any": (any_hits / n) if n else None,
        "share_in_wos_any_of_resolved": (any_hits / n_with_issn) if n_with_issn else None,
        "per_collection": {
            code: {"n": count, "share": (count / n) if n else None}
            for code, count in per_collection.items()
        },
    }


def gate_effect(
    records: Sequence[Mapping[str, Any]],
    collections: Mapping[str, set[str]],
    label_field: str,
) -> dict[str, Any]:
    """Share of all / human-included records whose venue is in the WoS tables."""
    included = [r for r in records if r.get(label_field) == 1]
    return {
        "label_field": label_field,
        "all_records": _subset_stats(records, collections),
        "included_records": _subset_stats(included, collections),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--label-field", default=None, help="default: the protocol's label_field")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--protocols-dir", type=Path, default=PROTOCOLS_DIR)
    ap.add_argument("--wos-dir", type=Path, default=REPO_ROOT, help="folder holding the WoS CSVs")
    return resolve_path_args(ap.parse_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    collections, missing = load_wos_collections(args.wos_dir)
    if not collections:
        print(
            "WoS collection CSVs not found in "
            f"{args.wos_dir} (expected: {', '.join(WOS_FILES.values())}). "
            "These licensed files are not distributed; skipping the gate-effect analysis."
        )
        return 0
    if missing:
        print(f"WARNING: collections missing: {', '.join(missing)}", file=sys.stderr)

    label_field = args.label_field
    if label_field is None:
        label_field = load_protocol(args.protocols_dir / f"{args.dataset}.json").label_field
    records = read_jsonl(args.data_dir / f"{args.dataset}.jsonl")
    if not records:
        raise SystemExit(f"no records for {args.dataset}; run fetch_synergy.py first")

    result = {
        "dataset": args.dataset,
        "generated": now_iso(),
        "collections_loaded": {code: len(table) for code, table in collections.items()},
        "missing_collections": missing,
        **gate_effect(records, collections, label_field),
    }
    out = args.results_dir / f"{args.dataset}_wos_gate.json"
    write_json(out, result)
    inc = result["included_records"]
    print(
        f"wrote {out}: included records n={inc['n']} with ISSN={inc['n_with_issn']} "
        f"in WoS(any)={inc['n_in_wos_any']} share={inc['share_in_wos_any']}"
    )
    if inc["n_with_issn"] == 0:
        print(
            "  no ISSNs in this dataset file; re-fetch with --route v2 or --hydrate included",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

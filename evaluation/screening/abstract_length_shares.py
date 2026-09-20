"""Abstract-length shares for one dataset's fetched jsonl.

Interpreter: any Python (stdlib only -- no network, no model call, no LLM).

Reads ``data/<dataset>.jsonl`` (written by ``fetch_synergy.py``) and reports, over the
non-empty abstracts, the count, mean, median and the share of abstracts that would be
shown *whole* at 2,000 / 3,000 / 4,000 characters -- the same six aggregate fields
reported for the three legacy corpora
(``Nagtegaal_2019``, ``van_de_Schoot_2017``, ``Smid_2020``), so the abstract cap decision
(3,000 characters unless any of the six datasets fails to clear 0.95 at that cap, in which
case 4,000) can be re-derived for the three new SYNERGY+ corpora without a network call or
a model call.

Writes ``data/<dataset>.abstract_lengths.json`` and prints only that path (nothing else),
so the script composes cleanly in a shell pipeline.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import read_jsonl, resolve_path_args, write_json  # noqa: E402

DATA_DIR = HERE / "data"

#: Character caps the brief measured against (2,000 as a rejected low anchor, 3,000 the
#: chosen default if all six datasets clear 0.95, 4,000 the fallback otherwise).
CAP_THRESHOLDS: tuple[int, ...] = (2000, 3000, 4000)


def abstract_lengths(records: Sequence[dict[str, Any]]) -> list[int]:
    """Character length of every non-empty, whitespace-stripped abstract in ``records``."""
    lengths = []
    for r in records:
        text = (r.get("abstract") or "").strip()
        if text:
            lengths.append(len(text))
    return lengths


def share_at(lengths: Sequence[int], limit: int) -> float | None:
    """Share of ``lengths`` at or under ``limit`` (shown whole at that cap); ``None`` if empty."""
    if not lengths:
        return None
    return sum(1 for n in lengths if n <= limit) / len(lengths)


def length_shares(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The six aggregate fields of brief section 2.2's cap-choice table for one corpus."""
    lengths = abstract_lengths(records)
    out: dict[str, Any] = {
        "n_records": len(records),
        "n_nonempty_abstracts": len(lengths),
        "mean": statistics.fmean(lengths) if lengths else None,
        "median": statistics.median(lengths) if lengths else None,
    }
    for limit in CAP_THRESHOLDS:
        out[f"share_at_{limit}"] = share_at(lengths, limit)
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, help="SYNERGY dataset id, e.g. Fong_2021")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return resolve_path_args(ap.parse_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = args.data_dir / f"{args.dataset}.jsonl"
    records = read_jsonl(path)
    if not records:
        raise SystemExit(f"no records in {path}; run fetch_synergy.py first")
    result = {"dataset": args.dataset, **length_shares(records)}
    out_path = args.data_dir / f"{args.dataset}.abstract_lengths.json"
    write_json(out_path, result)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

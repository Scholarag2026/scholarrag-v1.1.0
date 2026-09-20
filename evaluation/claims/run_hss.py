"""Run the production claim-verification agent over the HSS applied-linguistics set (E2).

Interpreter: the **system** ``python`` (backend dependencies). ``--dry-run`` works anywhere.
Importers: ``tests/test_claims_hss.py``.

Inputs: ``claims/hss_claims.jsonl`` (committed; ``build_hss_set.py``) and the full-text cache
``claims/data/hss_fulltext/<doi-slug>.json``. Each item hands the agent *all* chunks of
``chunk_doi`` in the order ``chunk_text`` produced them (as production does), or no chunks
for the ``no_full_text`` rule, which takes production's deterministic path without a model
call. Rows additionally carry ``rule``, ``expected`` and ``correct`` (predicted status in
``expected``; labels are by construction). Output: ``results/hss_run<A|B>.jsonl`` + meta.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from build_hss_set import DEFAULT_CACHE_DIR, DEFAULT_OUT, load_cache  # noqa: E402
from common import read_jsonl  # noqa: E402
from verify_common import add_common_args, execute, parse_common  # noqa: E402

ITEM_KEYS = ("item_id", "rule", "expected", "claim", "source_doi", "chunk_doi", "chunk_index",
             "alteration")


def load_items(claims_path: Path, cache_dir: Path) -> list[dict[str, Any]]:
    claims = read_jsonl(claims_path)
    if not claims:
        raise SystemExit(f"no claims in {claims_path}; run build_hss_set.py first")
    cache = load_cache(cache_dir)
    items: list[dict[str, Any]] = []
    for row in claims:
        chunk_doi = row.get("chunk_doi")
        if chunk_doi:
            paper = cache.get(chunk_doi)
            if paper is None:
                raise SystemExit(f"{row['item_id']}: {chunk_doi} missing from cache {cache_dir}")
            chunks = [str(c["text"]) for c in paper["chunks"]]
        else:
            chunks = []
        item = {k: row.get(k) for k in ITEM_KEYS}
        item["title"] = row.get("title") or ""
        item["authors"] = row.get("authors")
        item["chunks"] = chunks
        items.append(item)
    return items


def mark_correct(row: dict[str, Any]) -> None:
    row["correct"] = row.get("predicted_status") in (row.get("expected") or [])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap)
    ap.add_argument("--claims", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument(
        "--name", default="hss",
        help="result-file name prefix (default 'hss'); a second HSS-shaped dev set sharing a "
        "results directory (e.g. band 2) must use a different name, or its "
        "<name>_run<X>.jsonl would collide with -- and silently append to -- the first set's",
    )
    return parse_common(ap, argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    items = load_items(args.claims, args.cache_dir)
    if args.limit:
        items = items[: args.limit]
    counts = {}
    for it in items:
        counts[it["rule"]] = counts.get(it["rule"], 0) + 1
    print(f"HSS set: {len(items)} items by rule {counts}")
    return asyncio.run(
        execute(
            name=args.name,
            items=items,
            chunks_of=lambda it: list(it["chunks"]),
            args=args,
            row_hook=mark_correct,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the production claim-verification agent over the SciFact dev (or train) set (E2).

Interpreter: the **system** ``python`` (backend dependencies). ``--dry-run`` works anywhere.

Data (``fetch_scifact.py`` or the SciFact release tarball extracted under ``claims/data/``)::

    claims_dev.jsonl    {id, claim, evidence: {doc_id: [{sentences: [...], label}]},
                         cited_doc_ids}
    claims_train.jsonl  same shape as claims_dev.jsonl (809 claims)
    corpus.jsonl        {doc_id, title, abstract: [sentence, ...], structured} -- shared by
                        both splits

For each claim and each of its ``cited_doc_ids`` one item is built: the abstract
sentences joined by a single space form the *one* full-text chunk handed to
``format_verification_prompt(claim, [chunk], title, None)``. Gold per (claim, doc):
SUPPORT / CONTRADICT when the claim's evidence for that document carries that label,
otherwise NOT_ENOUGH_INFO.

``--split {dev,train}`` (default ``dev``) selects the claims file; ``--split train`` writes
``scifact-train_run<A|B>.jsonl`` instead of ``scifact_run<A|B>.jsonl``, so a dev-tuning
results directory never collides with a SciFact **dev**-split test run in the same folder
``--sample-n``/``--sample-seed`` (train split only) draw the
120-pair stratified development sample (40 items per gold bucket, sorted
by ``(claim_id, doc_id)``, ``random.Random(seed).sample`` per bucket in SUPPORT / CONTRADICT /
NOT_ENOUGH_INFO order -- see :func:`stratified_train_sample`) and, when ``--sample-file`` is
also given and does not yet exist, write the resulting ``item_id`` list there so the sample is
frozen independently of the RNG from then on. ``--sample-file PATH`` alone (file already
present) loads that frozen list instead of resampling, which is how every run after the first
reproduces the exact same 120 pairs.

Output: ``results/scifact_run<A|B>.jsonl`` (or ``scifact-train_run<A|B>.jsonl``) plus
``.meta.json``; see ``verify_common.py`` for the row schema (claim_id, doc_id, gold,
predicted_status, evidence_quote, quote_is_verbatim, explanation, model_reported,
system_fingerprint, latency_s, tokens). ``--limit N`` keeps the first N claims (smoke tests,
applied before any ``--sample-file``/``--sample-n`` filtering). Resumable.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    SCIFACT_LABELS,
    read_json,
    read_jsonl,
    scifact_gold_for_doc,
    write_json,
)
from verify_common import ITEM_ID, add_common_args, execute, parse_common  # noqa: E402

DEFAULT_DATA_DIR = HERE / "data" / "data"
FALLBACK_DATA_DIR = HERE / "data"
CLAIMS_FILENAME: dict[str, str] = {"dev": "claims_dev.jsonl", "train": "claims_train.jsonl"}


def resolve_data_dir(data_dir: Path, claims_filename: str = "claims_dev.jsonl") -> Path:
    for candidate in (data_dir, FALLBACK_DATA_DIR, DEFAULT_DATA_DIR):
        if (candidate / claims_filename).exists() and (candidate / "corpus.jsonl").exists():
            return candidate
    raise SystemExit(
        f"{claims_filename} / corpus.jsonl not found under {data_dir}; run fetch_scifact.py first"
    )


def load_scifact(
    data_dir: Path, split: str = "dev"
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    filename = CLAIMS_FILENAME[split]
    folder = resolve_data_dir(data_dir, filename)
    claims = read_jsonl(folder / filename)
    corpus = {int(doc["doc_id"]): doc for doc in read_jsonl(folder / "corpus.jsonl")}
    return claims, corpus


def stratified_train_sample(
    items: Sequence[Mapping[str, Any]], *, n_per_bucket: int, seed: int
) -> list[str]:
    """The exact procedure for the frozen ``scifact-train-120`` sample.

    ``items`` is the list :func:`build_items` produced from the train split (already in claim
    file order / ``cited_doc_ids`` list order). Each item is bucketed by its ``gold`` label;
    each bucket is sorted by ``(claim_id, doc_id)``; a fresh ``random.Random(seed)`` samples
    ``n_per_bucket`` items from each bucket, in the order SUPPORT, CONTRADICT,
    NOT_ENOUGH_INFO (``common.SCIFACT_LABELS``). Returns the concatenated ``item_id`` list
    (120 entries for ``n_per_bucket=40``). Raises ``ValueError`` if a bucket is smaller than
    ``n_per_bucket`` (``random.Random.sample``'s own error).
    """
    by_gold: dict[str, list[Mapping[str, Any]]] = {g: [] for g in SCIFACT_LABELS}
    for it in items:
        by_gold[str(it["gold"])].append(it)
    out: list[str] = []
    for gold in SCIFACT_LABELS:
        bucket = sorted(by_gold[gold], key=lambda it: (it["claim_id"], it["doc_id"]))
        chosen = random.Random(seed).sample(bucket, n_per_bucket)
        out.extend(str(it[ITEM_ID]) for it in chosen)
    return out


def build_items(
    claims: Sequence[Mapping[str, Any]],
    corpus: Mapping[int, Mapping[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """One item per (claim, cited document); abstract sentences joined into one chunk."""
    items: list[dict[str, Any]] = []
    missing = 0
    for claim in claims[:limit] if limit else claims:
        for doc_id in claim.get("cited_doc_ids") or []:
            doc = corpus.get(int(doc_id))
            if doc is None:
                missing += 1
                continue
            chunk = " ".join(s.strip() for s in doc.get("abstract") or [] if s and s.strip())
            items.append(
                {
                    ITEM_ID: f"{claim['id']}:{doc_id}",
                    "claim_id": claim["id"],
                    "doc_id": int(doc_id),
                    "gold": scifact_gold_for_doc(claim, doc_id),
                    "claim": claim["claim"],
                    "title": doc.get("title") or "",
                    "authors": None,
                    "chunks": [chunk] if chunk else [],
                }
            )
    if missing:
        print(f"WARNING: {missing} cited documents missing from corpus.jsonl", file=sys.stderr)
    return items


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--split", choices=("dev", "train"), default="dev",
                    help="claims_dev.jsonl (default) or claims_train.jsonl")
    ap.add_argument("--sample-n", type=int, default=None,
                    help="train split only: items per gold bucket (brief 5.1: 40 -> 120 pairs)")
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="required with --sample-n; random.Random seed for the stratified draw")
    ap.add_argument("--sample-file", type=Path, default=None,
                    help="frozen item_id list (JSON array): loaded when it already exists, "
                    "else built from --sample-n/--sample-seed and written there")
    return parse_common(ap, argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    claims, corpus = load_scifact(args.data_dir, args.split)
    items = build_items(claims, corpus, args.limit)
    gold_counts = {g: sum(1 for it in items if it["gold"] == g) for g in
                   ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")}
    label = "SciFact dev" if args.split == "dev" else "SciFact train"
    print(f"{label}: {len(claims)} claims -> {len(items)} claim-document pairs {gold_counts}")
    sample_ids: list[str] | None = None
    if args.split == "train" and args.sample_file is not None and args.sample_file.exists():
        sample_ids = [str(x) for x in read_json(args.sample_file, [])]
        print(f"loaded frozen sample {args.sample_file} ({len(sample_ids)} item ids)")
    elif args.split == "train" and args.sample_n:
        if args.sample_seed is None:
            raise SystemExit("--sample-seed is required with --sample-n")
        sample_ids = stratified_train_sample(
            items, n_per_bucket=args.sample_n, seed=args.sample_seed
        )
        if args.sample_file is not None:
            write_json(args.sample_file, sample_ids)
            print(f"wrote frozen sample {args.sample_file} ({len(sample_ids)} item ids)")
    if sample_ids is not None:
        wanted = set(sample_ids)
        items = [it for it in items if it[ITEM_ID] in wanted]
        print(f"sampled down to {len(items)} claim-document pairs")
    name = "scifact" if args.split == "dev" else "scifact-train"
    return asyncio.run(
        execute(name=name, items=items, chunks_of=lambda it: list(it["chunks"]), args=args)
    )


if __name__ == "__main__":
    raise SystemExit(main())

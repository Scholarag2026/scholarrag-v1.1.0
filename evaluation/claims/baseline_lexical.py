"""Lexical-overlap baseline for the claim-verification evaluation (E2): SciFact and HSS.

Interpreter: ``evaluation/.venv/Scripts/python`` or the system ``python`` (stdlib only).

``--set scifact`` (default) scores the SciFact dev pairs; ``--set hss`` scores the HSS set
loaded exactly as ``run_hss.py`` does (``hss_claims.jsonl`` + the full-text cache), applying
the same rule per item -- items without chunks take the deterministic ``no_full_text`` path
-- and reports per-rule accuracy plus the verified-vs-not binary metrics
(``results/hss_baseline.json``). On the HSS set the baseline shows what lexical overlap alone
buys: verbatim items and single-word alterations both have Jaccard close to 1, so the
baseline marks altered items ``verified``; it cannot answer ``needs_nuance`` at all, so it
scores 0 on the ``over_specified`` stratum by construction.

For every (claim, cited document) pair of the SciFact dev set the baseline computes the
maximum token-Jaccard similarity between the claim and any sentence of the document's
abstract. ``>= --threshold`` (default 0.5) predicts ``verified``, otherwise ``unsupported``.
Gold labels are mapped exactly as for the LLM runs (SUPPORT -> verified; CONTRADICT and
NOT_ENOUGH_INFO -> unsupported). The baseline can never say ``needs_nuance``.

``--match-run <results/scifact_runA.jsonl>`` restricts the pairs to those present in an LLM
run file so that a ``--limit`` smoke run and its baseline are comparable.

Output: ``results/scifact_baseline.json`` with per-class precision/recall/F1, the confusion
matrix (raw gold label x prediction), accuracy, macro-F1 and the Jaccard distribution;
``results/hss_baseline.json`` with per-rule accuracy, predicted-status counts and the binary
verified metrics.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from build_hss_set import DEFAULT_CACHE_DIR, DEFAULT_OUT  # noqa: E402
from common import (  # noqa: E402
    SCIFACT_LABELS,
    SCIFACT_TO_STATUS,
    accuracy,
    binary_metrics,
    confusion_counts,
    confusion_matrix,
    macro_f1,
    max_sentence_jaccard,
    now_iso,
    per_class_prf,
    percentile,
    read_jsonl,
    resolve_path_args,
    split_sentences,
    write_json,
)
from run_scifact import DEFAULT_DATA_DIR, build_items, load_scifact  # noqa: E402

RESULTS_DIR = HERE / "results" / "v6"
BASELINE_CLASSES: tuple[str, ...] = ("verified", "unsupported")
# Sixth rule `over_specified`; the lexical baseline can never answer needs_nuance, so it scores
# 0 on this stratum by construction.
HSS_RULES: tuple[str, ...] = (
    "verbatim", "paraphrase", "altered", "over_specified", "wrong_paper", "no_full_text",
)


def predict_lexical(claim: str, chunks: Sequence[str], threshold: float) -> tuple[str, float]:
    """(status, max_jaccard) for one claim against the sentences of its chunks."""
    sentences = [s for c in chunks for s in split_sentences(c)]
    best, _ = max_sentence_jaccard(claim, sentences)
    return ("verified" if best >= threshold else "unsupported"), best


def evaluate_lexical(items: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    gold_raw: list[str] = []
    gold_mapped: list[str] = []
    preds: list[str] = []
    scores: list[float] = []
    for item in items:
        status, best = predict_lexical(item["claim"], item["chunks"], threshold)
        gold_raw.append(item["gold"])
        gold_mapped.append(SCIFACT_TO_STATUS[item["gold"]])
        preds.append(status)
        scores.append(best)
    prf = per_class_prf(gold_mapped, preds, BASELINE_CLASSES)
    return {
        "threshold": threshold,
        "n": len(items),
        "per_class": prf,
        "macro_f1": macro_f1(prf),
        "accuracy": accuracy(gold_mapped, preds),
        "confusion_gold_raw_x_predicted": confusion_matrix(
            gold_raw, preds, SCIFACT_LABELS, BASELINE_CLASSES
        ),
        "predicted_counts": {c: preds.count(c) for c in BASELINE_CLASSES},
        "jaccard": {
            "median": percentile(scores, 50),
            "p90": percentile(scores, 90),
            "max": max(scores) if scores else None,
        },
    }


def evaluate_lexical_hss(items: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    """Per-rule accuracy of the lexical rule on HSS items (``expected`` lists per item)."""
    scored: list[dict[str, Any]] = []
    for item in items:
        chunks = list(item.get("chunks") or [])
        if not chunks:
            status, best = "no_full_text", None
        else:
            status, best = predict_lexical(item["claim"], chunks, threshold)
        expected = list(item.get("expected") or [])
        scored.append({
            "item_id": item["item_id"],
            "rule": item.get("rule"),
            "expected": expected,
            "predicted_status": status,
            "max_jaccard": best,
            "correct": status in expected,
        })
    by_rule: dict[str, Any] = {}
    for rule in HSS_RULES:
        sub = [r for r in scored if r["rule"] == rule]
        if not sub:
            continue
        n_ok = sum(1 for r in sub if r["correct"])
        by_rule[rule] = {
            "n": len(sub),
            "correct": n_ok,
            "accuracy": n_ok / len(sub),
            "predicted_counts": {
                c: sum(1 for r in sub if r["predicted_status"] == c)
                for c in sorted({r["predicted_status"] for r in sub})
            },
        }
    # Binary view over items with chunks only: the no-chunk items are rejected by
    # construction (no_full_text) and are not true negatives, exactly as in
    # summarize.hss_run_summary for the LLM runs.
    answered = [r for r in scored if r["predicted_status"] != "no_full_text"]
    gold_pos = [1 if "verified" in r["expected"] else 0 for r in answered]
    pred_pos = [1 if r["predicted_status"] == "verified" else 0 for r in answered]
    verified_binary = binary_metrics(confusion_counts(gold_pos, pred_pos))
    verified_binary["n_excluded_deterministic"] = len(scored) - len(answered)
    n_ok = sum(1 for r in scored if r["correct"])
    return {
        "threshold": threshold,
        "n": len(scored),
        "correct": n_ok,
        "accuracy": (n_ok / len(scored)) if scored else None,
        "by_rule": by_rule,
        "verified_binary": verified_binary,
        "items": scored,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--set", choices=("scifact", "hss"), default="scifact")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="first N claims only")
    ap.add_argument("--match-run", type=Path, default=None, help="restrict to items of a run file")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--claims", type=Path, default=DEFAULT_OUT, help="--set hss: claims file")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="--set hss")
    return resolve_path_args(ap.parse_args(argv))


def main_hss(args: argparse.Namespace) -> int:
    from run_hss import load_items

    items = load_items(args.claims, args.cache_dir)
    result = {
        "name": "hss_baseline_lexical",
        "generated": now_iso(),
        "claims_file": args.claims.name,
        **evaluate_lexical_hss(items, args.threshold),
    }
    out = args.results_dir / "hss_baseline.json"
    write_json(out, result)
    per_rule = ", ".join(f"{r}={b['correct']}/{b['n']}" for r, b in result["by_rule"].items())
    print(f"wrote {out}: n={result['n']} accuracy={result['accuracy']} ({per_rule})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.set == "hss":
        return main_hss(args)
    claims, corpus = load_scifact(args.data_dir)
    items = build_items(claims, corpus, args.limit)
    if args.match_run is not None:
        wanted = {r["item_id"] for r in read_jsonl(args.match_run)}
        items = [it for it in items if it["item_id"] in wanted]
    result = {
        "name": "scifact_baseline_lexical",
        "generated": now_iso(),
        "matched_run": args.match_run.name if args.match_run else None,
        **evaluate_lexical(items, args.threshold),
    }
    out = args.results_dir / "scifact_baseline.json"
    write_json(out, result)
    v = result["per_class"]["verified"]
    print(
        f"wrote {out}: n={result['n']} accuracy={result['accuracy']} "
        f"verified P/R/F1={v['precision']}/{v['recall']}/{v['f1']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

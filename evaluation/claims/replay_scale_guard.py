"""Phase 2 replay of guard 7, scale-word fidelity: every stored v5 model answer and
every delivered row of the shipped demonstration baseline, re-scored through the real, shipped
``app.services.fulltext.verify_claim_with_policy`` / ``apply_verification_guards``, so the
guard exercised is the guard that ships, not a second implementation of the rule the
design's own feasibility probe used for phase 1.

Offline, no network, no model call, zero cost: for every model-answered row (not
``deterministic``, not ``error``) of the six ``results/v5/*.jsonl`` files, this rebuilds a
``ModelPassAnswer`` from that row's own stored ``model_status``, ``evidence_quote``,
``evidence_quotes``, ``assertions``, ``explanation``, ``suggested_revision`` and
``unstated_details`` (``replay_repair_turn_v5.py`` phase 1's own technique) and replays it
through ``verify_claim_with_policy`` with a ``call_model`` that returns that answer once and
raises a sentinel if asked a second time -- if the sentinel is ever raised, the repair turn
would have fired, which the design's own composition argument (section 6) says never
happens once the first pass's status is already ``verified``, since guard 7's gate closes
before the trigger's own reason-list check ever sees a difference.

``chunk_texts``/``chunks`` are the row's own evidence quotes, not the source paper: design
section 8 states chunk texts are not needed for either replay path, because guard 7 itself
reads only the claim and the quotes and only ever runs on a row that already reached
``verified``, i.e. a row every chunk-dependent guard already passed. This script's own
validation confirms this
empirically rather than assuming it: replaying every verified row of the six v5 files, every
verified `Review` item and all delivered rows with the quotes standing in for the chunks
reproduces the stored ``predicted_status``/``status`` exactly under the pre-guard-7 code, so
substituting them introduces no side effect through the five guards that do read chunk
texts (attribution, the two numeric guards) before guard 7's own, chunk-free comparison runs.
That guarantee is about the *verified* population only: one already-non-verified row
(``real-test-11``, stored ``needs_nuance``) replays as ``unsupported`` under the substitution,
because its short evidence quote alone no longer satisfies guard 1's coverage-or-located-quote
test the way the real chunk did. Guard 7 never sees this row either way (its gate closes
before guard 7 runs, on a status that already left ``verified``), and the row is already
scored incorrect against its adjudicated label under both statuses, so this substitution
artefact changes no reported number; it is surfaced in the per-row diff as
``n_non_verified_rows_changed`` rather than silently absorbed.

The delivered rows store no model pass at all (only ``claim_text``, ``status`` and
``evidence_quotes``: see ``demo/check_delivered.py``), so they are re-scored through
``apply_verification_guards`` directly, with ``status="verified"`` (every delivered row's own
stored status, confirmed at run time), exactly as the design's phase 2 specifies.

Usage (system python, from ``evaluation/claims/``)::

    python replay_scale_guard.py

Writes, outside the evaluation results tree (never under ``evaluation/**/results/**``, and
``results/v2`` through ``results/v5`` are never opened for writing):

- ``replay_scale_guard_per_row.json`` -- one entry per replayed v5 row: stored vs. replayed
  status and machine reasons, and whether the repair-turn sentinel fired.
- ``replay_scale_guard_delivered.json`` -- one entry per verified delivered row, same shape.
- ``replay_scale_guard_summary.json`` -- SciFact verified precision/recall, HSS/real accuracy
  and the quote-verbatim rate, each computed before and after the replay, beside the
  ``GUARD_DIGEST`` the replay ran under.

Phase 2 must reproduce phase 1's counts exactly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent  # evaluation/claims
EVAL_ROOT = HERE.parent  # evaluation
REPO_ROOT = EVAL_ROOT.parent  # repo root
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(HERE))

from common import SCIFACT_TO_STATUS, accuracy, add_backend_to_path, read_jsonl, write_json  # noqa: E402
from summarize import STATUS_TO_ANNOTATION_LABEL, load_annotation_labels  # noqa: E402

OUT_DIR = HERE / "results" / "scale_guard_replay"
V5_DIR = HERE / "results" / "v5"
DELIVERED_PATH = REPO_ROOT / "demo" / "expected" / "delivered_evidence.json"
REAL_LABELS_PATH = HERE / "annotation" / "real_annotation_adjudicated_v3.csv"

V5_FILES: list[str] = [
    "scifact_runA", "scifact_runB", "hss_runA", "hss_runB", "real_runA", "real_runB",
]
SCIFACT_FILES: list[str] = ["scifact_runA", "scifact_runB"]
HSS_FILES: list[str] = ["hss_runA", "hss_runB"]
REAL_FILES: list[str] = ["real_runA", "real_runB"]


class _WouldTriggerRepairTurn(Exception):
    """Sentinel: ``replay_repair_turn_v5.py`` phase 1's own technique. Raised instead of a
    real second model call so ``verify_claim_with_policy``'s own (unchanged) exception
    handling around the optional second call tells us, for free, whether its trigger check
    licensed one -- without a network call and without this script deciding the trigger
    condition itself."""


def _quotes_pool(row: dict[str, Any]) -> list[str]:
    quotes = row.get("evidence_quotes") or (
        [row["evidence_quote"]] if row.get("evidence_quote") else []
    )
    return [q for q in quotes if q]


async def _replay_v5_row(row: dict[str, Any], model_pass_answer: Any, verify: Any) -> dict[str, Any]:
    quotes = _quotes_pool(row)
    chunk_texts = list(quotes)
    chunks = [{"text": q} for q in quotes]

    async def call_model(repair_request: Any = None) -> Any:
        if repair_request is None:
            return model_pass_answer(
                status=row.get("model_status"),
                evidence_quote=row.get("evidence_quote"),
                evidence_quotes=list(row.get("evidence_quotes") or []),
                assertions=[dict(a) for a in (row.get("assertions") or [])],
                explanation=row.get("explanation"),
                suggested_revision=row.get("suggested_revision"),
                unstated_details=list(row.get("unstated_details") or []),
            )
        raise _WouldTriggerRepairTurn()

    entry: dict[str, Any] = {
        "item_id": row.get("item_id"),
        "stored_predicted_status": row.get("predicted_status"),
        "stored_machine_reasons": row.get("machine_reasons"),
        "quote_is_verbatim": row.get("quote_is_verbatim"),
    }
    try:
        result = await verify(row["claim"], chunk_texts, chunks, call_model)
    except _WouldTriggerRepairTurn:
        entry["repair_turn_would_have_fired"] = True
        return entry
    entry["repair_turn_would_have_fired"] = False
    entry["replayed_status"] = result.status
    entry["replayed_machine_reasons"] = result.machine_reasons
    return entry


def _replay_delivered_row(row: dict[str, Any], index: int, apply_guards: Any) -> dict[str, Any] | None:
    """*index* is the row's 1-based position in ``delivered_evidence.json``'s own ``rows``
    array -- the numbering the design (and the human-review workbooks) use throughout
    (rows 3, 7, 12, 17, 23, 26, 27, 31). The row's own ``row`` field is a *different*
    number (the sentence's position within its own paragraph, which restarts at 1 per
    paragraph) and must not be used here."""
    if row.get("status") != "verified":
        return None
    quotes = [q for q in (row.get("evidence_quotes") or []) if q]
    chunk_texts = list(quotes)
    chunks = [{"text": q} for q in quotes]
    status, reasons, _diagnostics = apply_guards(
        "verified",
        claim_text=row["claim_text"],
        evidence_quote=quotes[0] if quotes else None,
        evidence_quotes=quotes or None,
        assertions=row.get("assertions") or [],
        chunk_texts=chunk_texts,
        chunks=chunks,
    )
    return {
        "row": index,
        "stored_status": row.get("status"),
        "replayed_status": status,
        "replayed_machine_reasons": reasons,
    }


def _status_after(per_row_by_id: dict[str, dict[str, Any]], row: dict[str, Any]) -> str:
    entry = per_row_by_id.get(row.get("item_id"))
    if entry is None or entry.get("repair_turn_would_have_fired"):
        return str(row.get("predicted_status"))
    return str(entry["replayed_status"])


def _scifact_prf(gold_mapped: list[str], predicted: list[str]) -> dict[str, Any]:
    tp = sum(1 for g, p in zip(gold_mapped, predicted, strict=True) if g == "verified" and p == "verified")
    fp = sum(1 for g, p in zip(gold_mapped, predicted, strict=True) if g != "verified" and p == "verified")
    n_gold_support = sum(1 for g in gold_mapped if g == "verified")
    n_predicted_verified = tp + fp
    return {
        "n_gold_support": n_gold_support,
        "n_predicted_verified": n_predicted_verified,
        "true_positive": tp,
        "false_positive": fp,
        "precision": (tp / n_predicted_verified) if n_predicted_verified else None,
        "recall": (tp / n_gold_support) if n_gold_support else None,
    }


async def _run(out_dir: Path = OUT_DIR) -> int:
    add_backend_to_path()
    os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")
    from app.services.fulltext import (  # noqa: PLC0415
        GUARD_DIGEST,
        ModelPassAnswer,
        apply_verification_guards,
        verify_claim_with_policy,
    )

    per_row: dict[str, list[dict[str, Any]]] = {}
    per_row_by_id: dict[str, dict[str, dict[str, Any]]] = {}
    file_summaries: dict[str, Any] = {}
    for name in V5_FILES:
        rows = read_jsonl(V5_DIR / f"{name}.jsonl")
        candidates = [r for r in rows if not r.get("deterministic") and not r.get("error")]
        replayed = [
            await _replay_v5_row(row, ModelPassAnswer, verify_claim_with_policy)
            for row in candidates
        ]
        per_row[name] = replayed
        per_row_by_id[name] = {e["item_id"]: e for e in replayed}

        n_verified = sum(1 for r in candidates if r.get("predicted_status") == "verified")
        demoted = [
            r for r in replayed
            if r["stored_predicted_status"] == "verified"
            and not r["repair_turn_would_have_fired"]
            and r["replayed_status"] != "verified"
        ]
        promoted_or_redirected = [
            r for r in replayed
            if r["stored_predicted_status"] != "verified"
            and not r["repair_turn_would_have_fired"]
            and r["replayed_status"] != r["stored_predicted_status"]
        ]
        n_repair_triggered = sum(1 for r in replayed if r["repair_turn_would_have_fired"])
        file_summaries[name] = {
            "n_candidates": len(candidates),
            "n_verified": n_verified,
            "n_demoted_from_verified": len(demoted),
            "demoted_item_ids": [r["item_id"] for r in demoted],
            "n_non_verified_rows_changed": len(promoted_or_redirected),
            "n_repair_turn_would_have_fired": n_repair_triggered,
        }

    delivered_rows = json.loads(DELIVERED_PATH.read_text(encoding="utf-8"))["rows"]
    delivered_replay = [
        entry
        for index, row in enumerate(delivered_rows, start=1)
        if (entry := _replay_delivered_row(row, index, apply_verification_guards)) is not None
    ]
    delivered_demoted = [e for e in delivered_replay if e["replayed_status"] != "verified"]

    scifact_metrics: dict[str, Any] = {}
    for name in SCIFACT_FILES:
        rows = [
            r for r in read_jsonl(V5_DIR / f"{name}.jsonl")
            if not r.get("deterministic") and not r.get("error")
        ]
        gold_mapped = [SCIFACT_TO_STATUS[str(r["gold"])] for r in rows]
        before = [str(r.get("predicted_status")) for r in rows]
        after = [_status_after(per_row_by_id[name], r) for r in rows]
        scifact_metrics[name] = {
            "before": _scifact_prf(gold_mapped, before),
            "after": _scifact_prf(gold_mapped, after),
        }

    # HSS: `hss_run_summary`'s own "correct" definition -- `predicted_status in expected`,
    # a construction-rule list already stored on every row. This is the "0.917"/"0.900"
    # figure design section 9 states.
    accuracy_metrics: dict[str, Any] = {}
    for name in HSS_FILES:
        rows = read_jsonl(V5_DIR / f"{name}.jsonl")

        def _hss_accuracy(use_replayed: bool, rows=rows, name=name) -> dict[str, Any]:
            correct = 0
            for r in rows:
                if r.get("deterministic") or r.get("error") or not use_replayed:
                    status = str(r.get("predicted_status"))
                else:
                    status = _status_after(per_row_by_id[name], r)
                if status in (r.get("expected") or []):
                    correct += 1
            return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None}

        accuracy_metrics[name] = {"before": _hss_accuracy(False), "after": _hss_accuracy(True)}

    # Real set: no construction rule (`expected` is empty on every row), so `summarize.
    # hss_vs_annotation_block`'s own method applies instead -- `predicted_status` mapped
    # through `STATUS_TO_ANNOTATION_LABEL`, scored strictly against the blind triple-Claude-
    # Opus adjudication's own `final_label` (`real_annotation_adjudicated_v3.csv`). This is
    # the "0.423" figure design section 9 states (`results/v5/summary.md` Table E2-g).
    real_labels = load_annotation_labels(REAL_LABELS_PATH)
    for name in REAL_FILES:
        rows = read_jsonl(V5_DIR / f"{name}.jsonl")
        joined = [r for r in rows if r.get("item_id") in real_labels]

        def _real_accuracy(use_replayed: bool, joined=joined, name=name) -> dict[str, Any]:
            gold = [real_labels[r["item_id"]]["final_label"] for r in joined]
            pred = []
            for r in joined:
                status = (
                    _status_after(per_row_by_id[name], r) if use_replayed
                    else str(r.get("predicted_status"))
                )
                pred.append(STATUS_TO_ANNOTATION_LABEL.get(status, "unsupported"))
            return {"n": len(joined), "accuracy": accuracy(gold, pred)}

        accuracy_metrics[name] = {"before": _real_accuracy(False), "after": _real_accuracy(True)}

    quote_verbatim: dict[str, Any] = {}
    for name in V5_FILES:
        rows = read_jsonl(V5_DIR / f"{name}.jsonl")
        verified_before = [r for r in rows if r.get("predicted_status") == "verified"]
        still_verified = [
            r for r in verified_before if _status_after(per_row_by_id[name], r) == "verified"
        ]

        def _rate(population: list[dict[str, Any]]) -> float | None:
            return (
                sum(1 for r in population if r.get("quote_is_verbatim")) / len(population)
                if population else None
            )

        quote_verbatim[name] = {"before": _rate(verified_before), "after": _rate(still_verified)}

    summary = {
        "purpose": (
            "Phase 2 (design section 8): offline replay of guard 7 against every stored v5 "
            "model answer and every delivered row of the shipped demonstration baseline, "
            "through the real, shipped verify_claim_with_policy / "
            "apply_verification_guards -- no model call, no network. Must reproduce the "
            "prototype script's own counts exactly."
        ),
        "guard_digest": GUARD_DIGEST,
        "v5_file_summaries": file_summaries,
        "delivered": {
            "n_verified": len(delivered_replay),
            "n_demoted": len(delivered_demoted),
            "demoted_rows": [e["row"] for e in delivered_demoted],
        },
        "scifact_verified_precision_recall": scifact_metrics,
        "constructed_set_accuracy": accuracy_metrics,
        "quote_verbatim_rate": quote_verbatim,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "replay_scale_guard_per_row.json", per_row)
    write_json(out_dir / "replay_scale_guard_delivered.json", delivered_replay)
    write_json(out_dir / "replay_scale_guard_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out_dir / 'replay_scale_guard_summary.json'}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=OUT_DIR,
        help="directory the three replay_scale_guard_*.json files are written to "
        "(default: %(default)s, a results directory that ships with the release).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_run(args.out))


if __name__ == "__main__":
    raise SystemExit(main())

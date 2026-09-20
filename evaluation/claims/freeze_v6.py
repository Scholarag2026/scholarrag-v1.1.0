"""Freeze v6: deterministic re-scoring of every cached model answer under the guard set
shipped at HEAD. No model call and no network access anywhere in this script; every number
comes from replaying stored data through code.

The cached model answers under `evaluation/claims/results/v6/inputs/` were measured under
`GUARD_DIGEST 570a5b663e6140da` (`SOURCE_GUARD_DIGEST` below). Guard 7 (the
scale-word-fidelity guard, committed at `26dadc4`) moved it to `8a6c833ffc329f89` without
changing `VERIFICATION_POLICY_VERSION`, `QUOTE_RELOCATION_VERSION`,
`CLAIM_VERIFICATION_PROMPT_VERSION` or `QUOTE_REPAIR_PROMPT_VERSION`. Guard 7 is gated on
`status == "verified"` and can only ever cap a row to `needs_nuance`, so every cached answer
remains a valid observation of the frozen prompt at temperature 0, and a v6 result tree is
the deterministic replay of that answer through the guard set at HEAD rather than a fresh,
paid draw.

For each model-answered row of the six `results/v6/inputs/*.jsonl` files (`scifact`, `hss`,
`real`, runs A and B), this rebuilds a `ModelPassAnswer` from the row's own stored
`model_status`, `evidence_quote(s)`, `assertions`, `explanation`, `suggested_revision` and
`unstated_details`, and replays it through the real, shared `verify_claim_with_policy`
**twice**: once with guard 7 (`_guard_scale_fidelity`) monkeypatched to a no-op (the
pre-guard-7 baseline) and once with the real, shipped guard. The delta between the two
passes isolates guard 7's own effect from the quotes-as-chunks substitution's own effect --
neither the cached rows nor `delivered_evidence.json` store the source paper's full text, so
(as `replay_scale_guard.py` already established) the row's own evidence quotes stand in for
the source chunks. The substitution can shift an *earlier* guard's own verdict on a handful
of rows (documented per-row in
`FREEZE_DIGESTS_v6.md`), but because both the before-guard-7 and after-guard-7 passes use
the identical substituted chunks, any such shift is identical in both passes and cancels out
of the delta; the delta is guard 7's own contribution alone.

The delta is only ever applied to a row's *cached* status when that cached status was
itself `"verified"` -- the exact population guard 7's own gate targets in real production,
where the guard chain ran against the real source text. A row whose cached status already
left `verified` before guard 7 could run is left byte-for-byte as cached, regardless of what
the substitution-based replay's own before/after shows for it, because production's real
guard chain (against the real chunks) is authoritative for that row and guard 7 never had a
chance to examine it there.

Writes, under `evaluation/claims/results/v6/`, the same file layout as the cached inputs:
`PROMPT.txt` and the two baseline files (copied byte-for-byte -- the lexical baseline never
calls a guard), `{scifact,hss,real}_run{A,B}.jsonl` (the cached rows, each carrying an added
`v6_replay` provenance object; unchanged rows are unchanged byte-for-byte apart from that one
added key), `{scifact,hss,real}_run{A,B}.meta.json` (the cached session/model/cost fields,
unchanged, plus `guard_digest` updated to the new value and a `replay` block), and
`guard7_reach_diagnostic.json` (the ungated detector-reach count:
`scale_alignment_findings`/`_scale_negation_scope_shift` called directly on every candidate
row's own claim and quotes, regardless of status, so a reader can see the guard's reach on
this data even though the gate keeps every one of those firings off the scored population).
`summary.json`/`summary.md`/`figure4_claims.json` are written by a separate step
(`build_v6_summary`, invoked by `main`) that calls `summarize.py`'s own, unmodified
`summarise`/`hss_vs_annotation_block`/`render_markdown` against the v6 tree.

Usage (system python, from `evaluation/claims/`), matching `replay_scale_guard.py`::

    python freeze_v6.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent  # evaluation/claims
EVAL_ROOT = HERE.parent  # evaluation
REPO_ROOT = EVAL_ROOT.parent  # repo root
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(HERE))

from common import add_backend_to_path, now_iso, read_json, read_jsonl, write_json  # noqa: E402

V6_DIR = HERE / "results" / "v6"
INPUTS_DIR = V6_DIR / "inputs"
SOURCE_GUARD_DIGEST = "570a5b663e6140da"  # guard digest the cached answers were measured under

RUN_BLOCKS: tuple[str, ...] = ("scifact", "hss", "real")
RUNS: tuple[str, ...] = ("A", "B")
BASELINE_FILES: tuple[str, ...] = ("hss_baseline.json", "scifact_baseline.json")


class _WouldTriggerRepairTurnError(Exception):
    """Sentinel (the technique `replay_repair_turn_v5.py`/`replay_scale_guard.py` both use):
    raised by the replay's own `call_model` instead of making a real second model call, so
    `verify_claim_with_policy`'s own (unchanged) second-call logic tells us, for free,
    whether its trigger check licensed one -- without a network call and without this script
    deciding the trigger condition itself."""


def quotes_pool(row: dict[str, Any]) -> list[str]:
    quotes = row.get("evidence_quotes") or (
        [row["evidence_quote"]] if row.get("evidence_quote") else []
    )
    return [q for q in quotes if q]


def _fresh_call_model(row: dict[str, Any], model_pass_answer: Any) -> Any:
    """A `call_model` closure good for exactly one first-pass call; a second call (the
    repair turn) raises the sentinel. A fresh closure (fresh `used` counter) is required per
    replay pass, since the before-guard-7 and after-guard-7 passes are two independent calls
    to `verify_claim_with_policy`."""
    used = {"n": 0}

    async def call_model(repair_request: Any = None) -> Any:
        if repair_request is not None or used["n"] > 0:
            raise _WouldTriggerRepairTurnError()
        used["n"] += 1
        return model_pass_answer(
            status=row.get("model_status"),
            evidence_quote=row.get("evidence_quote"),
            evidence_quotes=list(row.get("evidence_quotes") or []),
            assertions=[dict(a) for a in (row.get("assertions") or [])],
            explanation=row.get("explanation"),
            suggested_revision=row.get("suggested_revision"),
            unstated_details=list(row.get("unstated_details") or []),
        )

    return call_model


async def replay_row_before_after(
    row: dict[str, Any], fulltext: Any, model_pass_answer: Any, verify: Any
) -> dict[str, Any]:
    """Replay one model-answered row twice (sequentially -- the guard-7 monkeypatch below is
    shared mutable module state, so the two passes for one row, and every row after it, must
    never run concurrently) and return the raw before/after facts. Interpretation (which
    facts drive the written status) is `apply_v6_status`'s job, not this function's."""
    quotes = quotes_pool(row)
    chunk_texts = list(quotes)
    chunks = [{"text": q} for q in quotes]
    original_guard = fulltext._guard_scale_fidelity

    async def _pass(disable_guard7: bool) -> dict[str, Any]:
        if disable_guard7:
            fulltext._guard_scale_fidelity = lambda status, **_kw: (status, [])
        try:
            result = await verify(
                row["claim"], chunk_texts, chunks, _fresh_call_model(row, model_pass_answer)
            )
        except _WouldTriggerRepairTurnError:
            return {"status": None, "machine_reasons": None, "repair_turn_would_fire": True}
        finally:
            fulltext._guard_scale_fidelity = original_guard
        return {
            "status": result.status,
            "machine_reasons": list(result.machine_reasons),
            "repair_turn_would_fire": False,
        }

    before = await _pass(True)
    after = await _pass(False)
    return {"before": before, "after": after}


def apply_v6_status(v5_row: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    """The written v6 `predicted_status`/`machine_reasons`, and the `v6_replay` provenance
    object, for one model-answered row. See the module docstring for why the delta is only
    ever applied when the row's own *stored* v5 status was `"verified"`."""
    v5_status = str(v5_row.get("predicted_status"))
    v5_reasons = list(v5_row.get("machine_reasons") or [])
    before, after = replay["before"], replay["after"]
    anomaly = bool(before["repair_turn_would_fire"] or after["repair_turn_would_fire"])

    guard7_new: list[str] = []
    status_changed = False
    if not anomaly and before["status"] is not None and after["status"] is not None:
        guard7_new = [r for r in after["machine_reasons"] if r not in before["machine_reasons"]]
        status_changed = after["status"] != before["status"]

    if v5_status == "verified" and status_changed and not anomaly:
        v6_status, v6_reasons = after["status"], v5_reasons + guard7_new
    else:
        v6_status, v6_reasons = v5_status, v5_reasons

    return {
        "status": v6_status,
        "machine_reasons": v6_reasons,
        "provenance": {
            "source_row_id": v5_row.get("item_id"),
            "guard_digest_before": SOURCE_GUARD_DIGEST,
            "guard_digest_after": None,  # filled in by the caller, which knows the live digest
            "status_before": v5_status,
            "status_after": v6_status,
            "status_changed": v6_status != v5_status,
            "guard7_examined": before["status"] == "verified" if not anomaly else None,
            "guard7_findings": guard7_new,
            "repair_turn_would_fire": anomaly,
        },
    }


async def replay_run_file(
    name: str, run: str, fulltext: Any, model_pass_answer: Any, verify: Any, guard_digest: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay one `<name>_run<run>.jsonl` file. Returns (v6 rows, per-file counters)."""
    rows = read_jsonl(INPUTS_DIR / f"{name}_run{run}.jsonl")
    source_file = f"results/v6/inputs/{name}_run{run}.jsonl"
    out_rows: list[dict[str, Any]] = []
    n_examined = n_changed = n_anomaly = 0
    for row in rows:
        v6_row = dict(row)
        if row.get("deterministic") or row.get("error"):
            provenance = {
                "source_row_id": row.get("item_id"),
                "guard_digest_before": SOURCE_GUARD_DIGEST,
                "guard_digest_after": guard_digest,
                "status_before": row.get("predicted_status"),
                "status_after": row.get("predicted_status"),
                "status_changed": False,
                "guard7_examined": False,
                "guard7_findings": [],
                "repair_turn_would_fire": False,
                "note": (
                    "deterministic no_full_text row" if row.get("deterministic") else "error row"
                ),
            }
        else:
            replay = await replay_row_before_after(row, fulltext, model_pass_answer, verify)
            applied = apply_v6_status(row, replay)
            v6_row["predicted_status"] = applied["status"]
            v6_row["machine_reasons"] = applied["machine_reasons"]
            provenance = applied["provenance"]
            provenance["guard_digest_after"] = guard_digest
            provenance["source_row_id"] = row.get("item_id")
            if provenance["guard7_examined"]:
                n_examined += 1
            if provenance["status_changed"]:
                n_changed += 1
            if provenance["repair_turn_would_fire"]:
                n_anomaly += 1
        provenance["source_file"] = source_file
        v6_row["v6_replay"] = provenance
        out_rows.append(v6_row)
    counters = {
        "n_rows": len(rows),
        "n_candidates": sum(1 for r in rows if not r.get("deterministic") and not r.get("error")),
        "n_examined_by_guard7": n_examined,
        "n_status_changed": n_changed,
        "n_repair_turn_anomalies": n_anomaly,
    }
    return out_rows, counters


def reach_diagnostic(
    all_rows_by_file: dict[str, list[dict[str, Any]]], fulltext: Any
) -> dict[str, Any]:
    """The guard's ungated reach: `scale_alignment_findings`/`_scale_negation_scope_shift`
    called directly on every candidate row's own claim and quotes, regardless of
    `predicted_status` -- i.e. bypassing guard 7's own gate entirely. Purely diagnostic:
    never drives `predicted_status`, reported so a reader can see the guard's reach on this
    data (eleven firings across all 852 v5 rows, every one already predicted
    `unsupported`)."""
    per_file: dict[str, Any] = {}
    total_hits: list[tuple[str, str]] = []
    for name, rows in all_rows_by_file.items():
        hits = []
        for row in rows:
            if row.get("deterministic") or row.get("error"):
                continue
            quotes = quotes_pool(row)
            findings = fulltext.scale_alignment_findings(row["claim"], quotes)
            negation = fulltext._scale_negation_scope_shift(row["claim"], quotes)
            if findings or negation:
                hits.append({
                    "item_id": row.get("item_id"),
                    "stored_predicted_status": row.get("predicted_status"),
                    "scale_word_mismatch": bool(findings),
                    "negation_scope_shift": bool(negation),
                })
        per_file[name] = hits
        total_hits.extend((name, h["item_id"]) for h in hits)
    return {
        "purpose": (
            "Ungated reach of guard 7's own comparison, regardless of status; never applied "
            "to predicted_status. Eleven such firings across all 852 v5 rows, every one "
            "already predicted unsupported."
        ),
        "per_file": per_file,
        "n_total": len(total_hits),
        "all_already_unsupported": all(
            h["stored_predicted_status"] == "unsupported"
            for hits in per_file.values() for h in hits
        ),
    }


async def build_v6_results() -> dict[str, Any]:
    """Replay every one of the six v5 run files and the two baselines, write
    `results/v6/`, and return the digests/counters needed for `FREEZE_DIGESTS_v6.md` and the
    v5-to-v6 comparison table."""
    add_backend_to_path()
    import os

    os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")
    from app.agents.claim_verification_agent import (  # noqa: PLC0415
        CLAIM_VERIFICATION_PROMPT_VERSION,
        QUOTE_REPAIR_PROMPT_VERSION,
    )
    from app.services import fulltext  # noqa: PLC0415
    from app.services.fulltext import (  # noqa: PLC0415
        GUARD_DIGEST,
        QUOTE_RELOCATION_VERSION,
        VERIFICATION_POLICY_VERSION,
        ModelPassAnswer,
        verify_claim_with_policy,
    )

    V6_DIR.mkdir(parents=True, exist_ok=True)

    file_summaries: dict[str, Any] = {}
    all_rows_by_file: dict[str, list[dict[str, Any]]] = {}
    for name in RUN_BLOCKS:
        for run in RUNS:
            key = f"{name}_run{run}"
            src_meta_path = INPUTS_DIR / f"{key}.meta.json"
            if not src_meta_path.exists():
                continue
            rows, counters = await replay_run_file(
                name, run, fulltext, ModelPassAnswer, verify_claim_with_policy, GUARD_DIGEST
            )
            all_rows_by_file[key] = rows
            out_path = V6_DIR / f"{key}.jsonl"
            with out_path.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

            src_meta = read_json(src_meta_path, {})
            v6_meta = dict(src_meta)
            v6_meta["guard_digest"] = GUARD_DIGEST
            v6_meta["status_counts"] = dict(Counter(str(r.get("predicted_status")) for r in rows))
            v6_meta["is_replay"] = True
            v6_meta["replayed_from"] = "results/v6/inputs"
            v6_meta["replay"] = {
                "source": f"results/v6/inputs/{key}.jsonl",
                "source_guard_digest": SOURCE_GUARD_DIGEST,
                "replayed_guard_digest": GUARD_DIGEST,
                "quote_relocation_version": QUOTE_RELOCATION_VERSION,
                "verification_policy_version": VERIFICATION_POLICY_VERSION,
                "claim_verification_prompt_version": CLAIM_VERIFICATION_PROMPT_VERSION,
                "quote_repair_prompt_version": QUOTE_REPAIR_PROMPT_VERSION,
                "method": (
                    "deterministic offline replay of the stored model answers through "
                    "apply_verification_guards/verify_claim_with_policy at HEAD; no model "
                    "call, no network"
                ),
                "generated_at": now_iso(),
                **counters,
            }
            write_json(V6_DIR / f"{key}.meta.json", v6_meta)
            file_summaries[key] = counters

    for filename in BASELINE_FILES:
        src = INPUTS_DIR / filename
        if src.exists():
            shutil.copyfile(src, V6_DIR / filename)

    prompt_src = INPUTS_DIR / "PROMPT.txt"
    if prompt_src.exists():
        (V6_DIR / "PROMPT.txt").write_bytes(prompt_src.read_bytes())

    diagnostic = reach_diagnostic(all_rows_by_file, fulltext)
    write_json(V6_DIR / "guard7_reach_diagnostic.json", diagnostic)

    return {
        "guard_digest": GUARD_DIGEST,
        "source_guard_digest": SOURCE_GUARD_DIGEST,
        "quote_relocation_version": QUOTE_RELOCATION_VERSION,
        "verification_policy_version": VERIFICATION_POLICY_VERSION,
        "claim_verification_prompt_version": CLAIM_VERIFICATION_PROMPT_VERSION,
        "quote_repair_prompt_version": QUOTE_REPAIR_PROMPT_VERSION,
        "file_summaries": file_summaries,
        "reach_diagnostic": diagnostic,
    }


def build_v6_summary(digests: dict[str, Any]) -> dict[str, Any]:
    """`summary.json`/`summary.md`/`figure4_claims.json` for `results/v6/`, using
    `summarize.py`'s own, unmodified functions (imported, never edited) against the v6
    tree."""
    import summarize as sm  # noqa: PLC0415

    labels_path = HERE / "annotation" / "hss_annotation_adjudicated_v3.csv"
    real_labels_path = HERE / "annotation" / "real_annotation_adjudicated_v3.csv"

    summary = sm.summarise(V6_DIR)
    summary["hss_vs_annotation"] = sm.hss_vs_annotation_block(V6_DIR, labels_path)
    summary["real_vs_annotation"] = sm.hss_vs_annotation_block(
        V6_DIR, real_labels_path, name="real"
    )

    def _relative(path_str: str) -> str:
        try:
            return str(Path(path_str).relative_to(REPO_ROOT)).replace("\\", "/")
        except ValueError:
            return path_str

    # `summarise`/`hss_vs_annotation_block` record the absolute path they were called with
    # (unmodified upstream behaviour, unchanged here); relativised to REPO_ROOT before this
    # committed file is written, so it names the same checkout's own path regardless of
    # which git worktree produced it, rather than one worktree's own directory name.
    summary["results_dir"] = _relative(summary["results_dir"])
    for _block in ("hss_vs_annotation", "real_vs_annotation"):
        summary[_block]["labels_source"] = _relative(summary[_block]["labels_source"])

    summary["v6_freeze"] = {
        "statement": (
            "v6 is a deterministic re-scoring of the cached model answers under "
            "results/v6/inputs/ through the shipped guard set (GUARD_DIGEST "
            "8a6c833ffc329f89, moved from the cached answers' own 570a5b663e6140da by "
            "guard 7); no model was called to produce it."
        ),
        "guard_digest": digests["guard_digest"],
        "source_guard_digest": digests["source_guard_digest"],
        "quote_relocation_version": digests["quote_relocation_version"],
        "verification_policy_version": digests["verification_policy_version"],
        "claim_verification_prompt_version": digests["claim_verification_prompt_version"],
        "quote_repair_prompt_version": digests["quote_repair_prompt_version"],
        "file_summaries": digests["file_summaries"],
    }
    write_json(V6_DIR / "summary.json", summary)
    write_json(V6_DIR / "figure4_claims.json", {
        "generated": summary["generated"], "metrics": list(sm.FIGURE_METRICS),
        "rows": sm.figure_rows(summary),
    })
    (V6_DIR / "summary.md").write_text(sm.render_markdown(summary), encoding="utf-8")
    return summary


async def _run() -> int:
    digests = await build_v6_results()
    build_v6_summary(digests)
    print(json.dumps(digests["file_summaries"], indent=2, default=str))
    print(f"wrote {V6_DIR}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())

"""Summarise the claim-verification evaluation (E2) from ``--results-dir`` (default
``claims/results/v6``, the result of record).

Interpreter: any Python 3.12+ (stdlib + ``common``). Importers: ``tests/test_claims_summarize.py``
(via ``conftest.load_script_module("claims", "summarize")`` -- this module shares its name with
``screening/summarize.py``; never ``import summarize``).

Inputs (all optional; missing runs are skipped): ``scifact_run{A,B}.jsonl`` (+ ``.meta.json``),
``scifact_baseline.json``, ``hss_run{A,B}.jsonl`` (+ ``.meta.json``), ``hss_baseline.json``,
``real_run{A,B}.jsonl`` (+ ``.meta.json``, the real-claims set; no baseline, no dev
counterpart, no gold label -- see the ``"real"`` block's own docstring in :func:`summarise`).
Every number is routed through ``common.fmt`` so a missing value prints as ``-`` (runs that
share no items, an HSS run with zero rows). On the HSS set ``needs_nuance`` answers are
reported per rule as their own column; for the ``paraphrase`` rule they are *not* counted as
correct (expected ``verified`` only, as the revision design fixes it).
Outputs: ``summary.json``, ``summary.md`` (Tables E2-a SciFact, E2-b HSS set, E2-c
provenance/cost/latency, E2-d when ``--labels`` is given, and E2-g when ``--real-labels`` is
given) and ``figure4_claims.json``.

``--labels <csv>`` (default: none) scores the committed ``hss_run<A|B>.jsonl`` against a
blind-annotation CSV (``item_id``, ``final_label``, ``expected_label``; see
``claims/annotation/hss_annotation_adjudicated_v2.csv`` and its ``PROVENANCE.md``) instead of
re-running the verifier: no LLM or network call is made. It maps ``predicted_status`` through
``STATUS_TO_ANNOTATION_LABEL`` (verified -> verified, needs_nuance -> needs_nuance,
unsupported/error -> unsupported, no_full_text -> no_full_text), joins on ``item_id``, and adds
a ``hss_vs_annotation`` block (strict ``accuracy`` against the single adjudicated label,
``accuracy_construction_tolerant`` -- Table E2-b's convention of scoring ``unsupported`` and
``needs_nuance`` as equally correct on the ten ``altered`` items, applied here too --
per-label P/R/F1, a 4x4 confusion table, ``dominant_miss_cell``, Cohen's kappa per run,
``n_adjudicated_needs_nuance``, and ``construction_vs_adjudicated`` -- the by-construction
label's own agreement with the adjudicated label) to ``summary.json`` and Table E2-d to
``summary.md``, without altering any other block.

``--real-labels <csv>`` (default: none) is the same scoring
(:func:`hss_vs_annotation_block`, ``name="real"``) against the committed
``real_run<A|B>.jsonl`` and a real-claims adjudication CSV (e.g.
``claims/annotation/real_annotation_adjudicated_v3.csv``), added as a ``real_vs_annotation``
block and Table E2-g, independently of (and in addition to) ``--labels``/``hss_vs_annotation``.
Every ``expected_label`` in the real-claims annotation CSV is empty (a real reviewer's claim
has no construction rule), so ``accuracy_construction_tolerant`` and
``construction_vs_adjudicated`` are still computed (for the JSON) but read as vacuous rather
than as a genuine 0 %; Table E2-g omits the corresponding prose and the closing
construction-vs-adjudicated line rather than report a misleading percentage.

Scoring (SciFact): gold SUPPORT -> ``verified``; CONTRADICT and NOT_ENOUGH_INFO ->
``unsupported`` (``common.SCIFACT_TO_STATUS``). *Strict* per-class P/R/F1 treats
``needs_nuance`` / ``no_full_text`` / ``error`` predictions as misses that are never a false
positive for a listed class (``common.per_class_prf``); the *lenient* binary view is
``verified`` vs not-verified, computed over model-answered rows only (``error`` rows are
excluded from the confusion counts and reported as ``n_errors``, so a failed call is never
a "correct rejection"). A/B agreement pairs only items that carry a model decision in *both*
runs: deterministic ``no_full_text`` rows and ``error`` rows are dropped from the pairs and
counted (``n_excluded_deterministic`` / ``n_excluded_error``); when both runs are constant
and identical kappa is undefined (``kappa_undefined``) and the tables print ``-`` with the
same footnote as the screening summary. Quote fidelity = share of ``verified`` rows whose
``evidence_quote`` is a verbatim substring of the chunk text (whitespace-normalised; a
case-insensitive variant and the share of null quotes are reported alongside).

``label_mapping_sensitivity`` (Table E2-f) answers the reviewer objection that collapsing
SciFact's three gold labels onto the verifier's four statuses is a choice: it rescoes the same
SciFact rows under a *strict* mapping (identical to the table above), a *lenient* mapping in
which a predicted needs_nuance counts as verified (everything else unchanged), and, only for
rows that carry the per-assertion ``verdict`` field the v2 prompt adds (absent from every
v1 row), a *three-way* mapping that splits a predicted unsupported row by whether any of its
own assertions was itself verdict ``contradicted``, scoring CONTRADICT against that split
bucket instead of the merged class it otherwise shares with NOT_ENOUGH_INFO. Each mapping
reports per-class P/R/F1, macro-F1, CONTRADICT recall (of gold CONTRADICT rows landing in the
mapping's own correct bucket) and the count of gold CONTRADICT rows that mapping's own
predictions call verified -- the number a reviewer reads as the safety check, since it is not
sensitive to which unsupported-flavoured class a miss falls into.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    SCIFACT_LABELS,
    SCIFACT_TO_STATUS,
    VERIFICATION_STATUSES,
    accuracy,
    binary_metrics,
    cohens_kappa,
    confusion_counts,
    confusion_matrix,
    fmt,
    kappa_undefined,
    macro_f1,
    markdown_table,
    now_iso,
    per_class_prf,
    percent_agreement,
    read_json,
    read_jsonl,
    resolve_path_args,
    wilson_interval,
    write_json,
)

RESULTS_DIR = HERE / "results" / "v6"
RUNS = ("A", "B")
STRICT_CLASSES: tuple[str, ...] = ("verified", "unsupported")
THREE_WAY_CLASSES: tuple[str, ...] = ("verified", "unsupported_contradicted", "unsupported_other")
GOLD_TO_THREE_WAY: dict[str, str] = {
    "SUPPORT": "verified",
    "CONTRADICT": "unsupported_contradicted",
    "NOT_ENOUGH_INFO": "unsupported_other",
}
MAPPING_LABELS: dict[str, str] = {
    "strict": "strict (needs_nuance is a miss)",
    "lenient_needs_nuance_as_verified": "lenient (needs_nuance -> verified)",
}
# The sixth rule `over_specified` is included, so `by_rule` no longer
# silently drops it (`hss_run_summary`'s `if not sub: continue` skips a rule with no rows,
# never a rule missing from this tuple).
HSS_RULES: tuple[str, ...] = (
    "verbatim", "paraphrase", "altered", "over_specified", "wrong_paper", "no_full_text",
)
ANNOTATION_LABELS: tuple[str, ...] = VERIFICATION_STATUSES[:-1]  # drop "error"
STATUS_TO_ANNOTATION_LABEL: dict[str, str] = {
    "verified": "verified",
    "needs_nuance": "needs_nuance",
    "unsupported": "unsupported",
    "error": "unsupported",
    "no_full_text": "no_full_text",
}
FIGURE_METRICS: tuple[str, ...] = (
    "verified_precision", "verified_recall", "verified_f1", "accuracy", "quote_fidelity",
    "kappa_AB",
)


def _share(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def load_run(results_dir: Path, name: str, run: str) -> tuple[list[dict], dict] | None:
    path = results_dir / f"{name}_run{run}.jsonl"
    if not path.exists():
        return None
    rows = read_jsonl(path)
    meta = read_json(results_dir / f"{name}_run{run}.meta.json", {}) or {}
    return rows, meta


def quote_fidelity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    verified = [r for r in rows if r.get("predicted_status") == "verified"]
    n = len(verified)
    return {
        "n_verified": n,
        "verbatim_share": _share(sum(1 for r in verified if r.get("quote_is_verbatim")), n),
        "casefold_share": _share(
            sum(1 for r in verified if r.get("quote_is_verbatim_casefold")), n
        ),
        "null_quote_share": _share(sum(1 for r in verified if not r.get("evidence_quote")), n),
    }


def machine_reasons_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Tally of guard slugs fired across *rows* (the code guards in ``fulltext.py``).

    ``machine_reasons`` is a list field the production verifier attaches to rows whose status
    a guard changed or annotated; it is absent from any row written under prompt v1 (predates
    the field), which counts as "no reasons" rather than an error. ``by_slug`` is a plain
    ``Counter`` of every slug across every row (a row can carry more than one); ``n_guarded``
    is the count of rows carrying at least one reason slug, regardless of whether that slug
    changed anything -- a report-only guard, or a status-changing guard that caps a status onto
    the value the model already gave, still appends a slug.

    ``guarded_count`` is the stricter definition:
    the count of rows where a guard actually changed the model's own
    status, ``model_status is not None and predicted_status != model_status``. A row with no
    model call (``model_status`` is ``None``, e.g. a deterministic ``no_full_text`` row) never
    counts here even if it carries a reason slug. The two numbers can diverge sharply: on the
    frozen v2 HSS test runs every ``machine_reasons`` firing is a no-op (``n_guarded=5``,
    ``guarded_count=0`` in both runs), because the four ``attribution_mismatch`` firings cap
    ``unsupported`` onto ``unsupported`` and the one ``numeric_not_in_source`` firing is
    report-only by design; a reader must not infer from ``n_guarded`` alone that a guard
    corrected that many rows.
    """
    counts: Counter[str] = Counter()
    n_guarded = 0
    guarded_count = 0
    for r in rows:
        reasons = r.get("machine_reasons") or []
        if reasons:
            n_guarded += 1
        counts.update(reasons)
        model_status = r.get("model_status")
        if model_status is not None and r.get("predicted_status") != model_status:
            guarded_count += 1
    return {
        "by_slug": dict(counts),
        "n_guarded": n_guarded,
        "guarded_count": guarded_count,
        "n": len(rows),
    }


def diagnostics_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Tally of diagnostic slugs (``diagnostics`` never changes
    ``predicted_status`` and is always a sibling of ``machine_reasons``, never one of its
    members), mirroring :func:`machine_reasons_counts`. A row with no
    ``diagnostics`` field counts as carrying no diagnostics, not an error.
    """
    counts: Counter[str] = Counter()
    n_flagged = 0
    for r in rows:
        diags = r.get("diagnostics") or []
        if diags:
            n_flagged += 1
        counts.update(diags)
    return {"by_slug": dict(counts), "n_flagged": n_flagged, "n": len(rows)}


def machine_reasons_by_rule(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """:func:`machine_reasons_counts` computed separately for each HSS construction rule
    present in *rows* (brief section 6): the discriminator between a failed construction and
    a guard-floored stratum, in particular ``attribution_mismatch`` on ``over_specified``.
    """
    out: dict[str, Any] = {}
    rules = sorted({str(r["rule"]) for r in rows if r.get("rule") is not None})
    for rule in rules:
        out[rule] = machine_reasons_counts([r for r in rows if r.get("rule") == rule])
    return out


def _operator_of(alteration: str | None) -> str:
    """First segment of an ``"<operator>:<from>-><to>"`` alteration label (brief section 2.2
    item 4); ``"unknown"`` for a missing or unrecognised value (a row without ``alteration``,
    or one that carries no colon)."""
    if not alteration or ":" not in str(alteration):
        return "unknown"
    return str(alteration).split(":", 1)[0]


def by_alteration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """HSS ``altered`` rows bucketed by :func:`_operator_of`, each carrying ``n``,
    ``correct``, ``accuracy``, ``status_counts`` and ``unsupported_rate`` (the share of the
    bucket predicted ``unsupported`` -- the figure the non-adversarial non-round numeric
    sub-stratum is read against, brief section 6)."""
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(_operator_of(r.get("alteration")), []).append(r)
    out: dict[str, Any] = {}
    for op, sub in buckets.items():
        n = len(sub)
        correct = sum(1 for r in sub if r.get("correct"))
        out[op] = {
            "n": n,
            "correct": correct,
            "accuracy": _share(correct, n),
            "status_counts": dict(Counter(str(r.get("predicted_status")) for r in sub)),
            "unsupported_rate": _share(
                sum(1 for r in sub if r.get("predicted_status") == "unsupported"), n
            ),
        }
    return out


def wilson_ci(n_correct: int, n: int, *, alpha: float = 0.05) -> dict[str, Any]:
    """Wilson 95 % (default) interval on an accuracy of ``n_correct`` of ``n``;
    ``lo``/``hi``/``width`` are ``None``
    when ``n`` is 0."""
    lo, hi = wilson_interval(n_correct, n, alpha=alpha)
    width = None if lo is None or hi is None else hi - lo
    return {"lo": lo, "hi": hi, "width": width, "n": n, "n_correct": n_correct, "method": "wilson"}


def provenance_block(meta: Mapping[str, Any]) -> dict[str, Any]:
    latency = meta.get("latency_s") or {}
    return {
        "model_configured": meta.get("model_configured"),
        "model_reported": meta.get("model_reported"),
        "system_fingerprints": meta.get("system_fingerprints"),
        "temperature": meta.get("temperature"),
        "prompt_version": meta.get("prompt_version"),
        "concurrency": meta.get("concurrency"),
        "dry_run": meta.get("dry_run"),
        "n_errors": meta.get("n_errors"),
        "n_deterministic": meta.get("n_deterministic"),
        "tokens": {
            "input": meta.get("total_input_tokens"),
            "output": meta.get("total_output_tokens"),
            "cache_read": meta.get("total_cache_read_tokens"),
        },
        "total_cost": meta.get("total_cost"),
        "cost_basis": meta.get("cost_basis"),
        "total_cost_flat": meta.get("total_cost_flat"),
        "price": meta.get("price"),
        "latency": {"median": latency.get("median"), "p90": latency.get("p90")},
        "started": meta.get("started"),
        "finished": meta.get("finished"),
    }


def _duplicate_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``item_id``s (``claim_id:doc_id``) that occur more than once: a dev claim listing the
    same document twice in its own ``cited_doc_ids`` (``run_scifact.build_items`` emits one
    item per list entry, so the duplicate is sent to the model and billed, and both copies
    count in every metric), sorted for a stable report."""
    counts = Counter(str(r.get("item_id")) for r in rows)
    dupes = []
    for item_id, n in counts.items():
        if n > 1:
            claim_id, _, doc_id = item_id.partition(":")
            dupes.append({"claim_id": claim_id, "doc_id": doc_id, "count": n})
    return sorted(dupes, key=lambda d: (d["claim_id"], d["doc_id"]))


def _has_contradiction_signal(row: Mapping[str, Any]) -> bool:
    """True when *row* carries a per-assertion ``verdict`` of ``contradicted`` (the prompt
    v2 ``assertions`` field; absent from every v1 row, which predates it)."""
    return any(a.get("verdict") == "contradicted" for a in (row.get("assertions") or []))


def _mapping_block(
    gold_raw: Sequence[str],
    gold_mapped: Sequence[str],
    pred_mapped: Sequence[str],
    classes: tuple[str, ...],
    contradict_bucket: str,
) -> dict[str, Any]:
    """One label-mapping's scores: per-class P/R/F1, macro-F1, the recall of gold CONTRADICT
    rows landing in *contradict_bucket*, and the count of gold CONTRADICT rows this mapping's
    own predictions call ``verified`` -- the number a reviewer reads as the safety check,
    since it does not depend on which unsupported-flavoured bucket a miss falls into."""
    prf = per_class_prf(gold_mapped, pred_mapped, classes)
    contradict_idx = [i for i, g in enumerate(gold_raw) if g == "CONTRADICT"]
    n_contradict = len(contradict_idx)
    n_recalled = sum(1 for i in contradict_idx if pred_mapped[i] == contradict_bucket)
    n_predicted_verified = sum(1 for i in contradict_idx if pred_mapped[i] == "verified")
    return {
        "per_class": prf,
        "macro_f1": macro_f1(prf),
        "contradict_recall": _share(n_recalled, n_contradict),
        "n_gold_contradict_predicted_verified": n_predicted_verified,
    }


def label_mapping_sensitivity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score the same SciFact rows under three label mappings, answering the reviewer
    objection that collapsing SUPPORT/CONTRADICT/NOT_ENOUGH_INFO onto verified/unsupported is
    a choice:

    * ``strict`` -- Table E2-a's own mapping (``SCIFACT_TO_STATUS``); a predicted
      needs_nuance / no_full_text / error is a miss for its gold class and never a hit.
    * ``lenient_needs_nuance_as_verified`` -- identical, except a predicted needs_nuance is
      treated as verified; every other prediction is unchanged.
    * ``three_way_contradiction_signal`` -- only when at least one row carries an
      ``assertions`` field (prompt v2): splits a predicted unsupported row by whether any of
      its own assertions was itself verdict ``contradicted``, and scores CONTRADICT against
      that split bucket instead of the class it otherwise shares with NOT_ENOUGH_INFO.
      ``available`` is False for v1 rows (which predate ``assertions``), with a ``reason``
      string in place of the score blocks.
    """
    gold_raw = [str(r["gold"]) for r in rows]
    pred_raw = [str(r.get("predicted_status")) for r in rows]
    gold_mapped = [SCIFACT_TO_STATUS[g] for g in gold_raw]
    lenient_pred = ["verified" if p == "needs_nuance" else p for p in pred_raw]

    out: dict[str, Any] = {
        "strict": _mapping_block(gold_raw, gold_mapped, pred_raw, STRICT_CLASSES, "unsupported"),
        "lenient_needs_nuance_as_verified": _mapping_block(
            gold_raw, gold_mapped, lenient_pred, STRICT_CLASSES, "unsupported"
        ),
    }

    if any("assertions" in r for r in rows):
        gold_mapped3 = [GOLD_TO_THREE_WAY[g] for g in gold_raw]
        pred_mapped3 = [
            ("unsupported_contradicted" if _has_contradiction_signal(r) else "unsupported_other")
            if p == "unsupported" else p
            for r, p in zip(rows, pred_raw, strict=True)
        ]
        out["three_way_contradiction_signal"] = {
            "available": True,
            **_mapping_block(
                gold_raw, gold_mapped3, pred_mapped3, THREE_WAY_CLASSES,
                "unsupported_contradicted",
            ),
        }
    else:
        out["three_way_contradiction_signal"] = {
            "available": False,
            "reason": (
                "these rows carry no `assertions` field (prompt v1 predates the per-assertion "
                "verdict; only prompt v2 records a per-row contradiction signal)"
            ),
        }
    return out


def scifact_run_summary(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> dict:
    gold_raw = [str(r["gold"]) for r in rows]
    gold_mapped = [SCIFACT_TO_STATUS[g] for g in gold_raw]
    pred = [str(r.get("predicted_status")) for r in rows]
    strict = per_class_prf(gold_mapped, pred, STRICT_CLASSES)
    answered = [(g, p) for g, p, r in zip(gold_mapped, pred, rows, strict=True)
                if not r.get("error")]
    lenient_counts = confusion_counts(
        [1 if g == "verified" else 0 for g, _ in answered],
        [1 if p == "verified" else 0 for _, p in answered],
    )
    lenient = binary_metrics(lenient_counts)
    lenient["accuracy"] = _share(lenient_counts.tp + lenient_counts.tn, lenient_counts.n)
    lenient["n_errors"] = len(rows) - len(answered)
    return {
        "n": len(rows),
        "n_distinct_pairs": len({str(r.get("item_id")) for r in rows}),
        "duplicate_pairs": _duplicate_pairs(rows),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "n_deterministic": sum(1 for r in rows if r.get("deterministic")),
        "gold_counts": dict(Counter(gold_raw)),
        "predicted_counts": dict(Counter(pred)),
        "strict": strict,
        "macro_f1": macro_f1(strict),
        "accuracy": accuracy(gold_mapped, pred),
        "lenient": lenient,
        "confusion": confusion_matrix(gold_raw, pred, SCIFACT_LABELS, VERIFICATION_STATUSES),
        "needs_nuance_by_gold": {
            g: sum(
                1 for gr, p in zip(gold_raw, pred, strict=True) if gr == g and p == "needs_nuance"
            )
            for g in SCIFACT_LABELS
        },
        "label_mapping_sensitivity": label_mapping_sensitivity(rows),
        "quote_fidelity": quote_fidelity(rows),
        "machine_reasons": machine_reasons_counts(rows),
        "diagnostics": diagnostics_counts(rows),
        "provenance": provenance_block(meta),
    }


def hss_run_summary(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> dict:
    by_rule: dict[str, Any] = {}
    for rule in HSS_RULES:
        sub = [r for r in rows if r.get("rule") == rule]
        if not sub:
            continue
        by_rule[rule] = {
            "n": len(sub),
            "expected": sub[0].get("expected"),
            "correct": sum(1 for r in sub if r.get("correct")),
            "accuracy": _share(sum(1 for r in sub if r.get("correct")), len(sub)),
            "needs_nuance": sum(1 for r in sub if r.get("predicted_status") == "needs_nuance"),
            "status_counts": dict(Counter(str(r.get("predicted_status")) for r in sub)),
            "quote_fidelity": quote_fidelity(sub),
        }
    # Binary view over model decisions only: error rows are not correct rejections and the
    # deterministic no_full_text items (gold 0, predicted 0 by construction) are not true
    # negatives; both are excluded and counted.
    answered = [r for r in rows if not r.get("error") and not r.get("deterministic")]
    gold_pos = [1 if "verified" in (r.get("expected") or []) else 0 for r in answered]
    pred_pos = [1 if r.get("predicted_status") == "verified" else 0 for r in answered]
    verified_binary = binary_metrics(confusion_counts(gold_pos, pred_pos))
    verified_binary["n_errors"] = sum(1 for r in rows if r.get("error"))
    verified_binary["n_excluded_deterministic"] = sum(
        1 for r in rows if r.get("deterministic") and not r.get("error")
    )
    n_correct = sum(1 for r in rows if r.get("correct"))
    return {
        "n": len(rows),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "n_deterministic": sum(1 for r in rows if r.get("deterministic")),
        "correct": n_correct,
        "accuracy": _share(n_correct, len(rows)),
        "accuracy_wilson_ci": wilson_ci(n_correct, len(rows)),
        "n_needs_nuance": sum(1 for r in rows if r.get("predicted_status") == "needs_nuance"),
        "by_rule": by_rule,
        "by_alteration": by_alteration([r for r in rows if r.get("rule") == "altered"]),
        "verified_binary": verified_binary,
        "quote_fidelity": quote_fidelity(rows),
        "machine_reasons": machine_reasons_counts(rows),
        "machine_reasons_by_rule": machine_reasons_by_rule(rows),
        "diagnostics": diagnostics_counts(rows),
        "provenance": provenance_block(meta),
    }


def agreement(rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]]) -> dict:
    """A/B agreement over items that carry a *model* decision in both runs.

    Items present in both files are paired; a pair is excluded (and counted) when either row
    is deterministic (``no_full_text``: never reaches the model, agreement guaranteed) or an
    error row. ``n`` is the number of shared items, ``n_compared`` the pairs scored.
    """
    b_by_id = {r["item_id"]: r for r in rows_b}
    shared = [(r, b_by_id[r["item_id"]]) for r in rows_a if r["item_id"] in b_by_id]
    n_det = sum(1 for a, b in shared if a.get("deterministic") or b.get("deterministic"))
    pairs = [(a, b) for a, b in shared
             if not (a.get("deterministic") or b.get("deterministic"))]
    n_err = sum(1 for a, b in pairs if a.get("error") or b.get("error"))
    pairs = [(a, b) for a, b in pairs if not (a.get("error") or b.get("error"))]
    sa = [str(a.get("predicted_status")) for a, _ in pairs]
    sb = [str(b.get("predicted_status")) for _, b in pairs]
    ba = [1 if s == "verified" else 0 for s in sa]
    bb = [1 if s == "verified" else 0 for s in sb]
    return {
        "n": len(shared),
        "n_compared": len(pairs),
        "n_excluded_deterministic": n_det,
        "n_excluded_error": n_err,
        "percent_status": percent_agreement(sa, sb),
        "kappa_status": cohens_kappa(sa, sb),
        "kappa_undefined": kappa_undefined(sa, sb),
        "percent_binary": percent_agreement(ba, bb),
        "kappa_binary": cohens_kappa(ba, bb),
        "kappa_binary_undefined": kappa_undefined(ba, bb),
    }


KAPPA_FOOTNOTE = (
    "kappa undefined (constant runs): both runs gave the same status for every compared "
    "item, so expected agreement is 1 and kappa has no value; '-' is printed."
)


def _agreement_line(agree: Mapping[str, Any], *, binary: bool) -> str:
    head = (
        f"A/B agreement (n={agree['n']}, compared {agree.get('n_compared', agree['n'])}; "
        f"excluded {agree.get('n_excluded_deterministic', 0)} deterministic, "
        f"{agree.get('n_excluded_error', 0)} error): status {fmt(agree['percent_status'])} "
        f"(kappa {fmt(agree['kappa_status'])})"
    )
    if binary:
        head += (f"; binary {fmt(agree['percent_binary'])} "
                 f"(kappa {fmt(agree['kappa_binary'])})")
    return head + "."


def load_annotation_labels(path: Path) -> dict[str, dict[str, Any]]:
    """``{item_id: row}`` from an adjudicated-labels CSV (e.g.
    ``annotation/hss_annotation_adjudicated_v2.csv``): columns ``item_id``, ``final_label``,
    ``expected_label`` (the construction rule's label, ``|``-joined when disjunctive) are
    required; other columns (``ann_id``, ``rationale``, ...) are ignored."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["item_id"]: row for row in csv.DictReader(fh)}


def construction_vs_annotation(labels: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Agreement of the by-construction label with the adjudicated ``final_label``, over the
    labels file alone (no verifier run needed). ``expected_label`` is ``|``-joined when the
    construction rule admits more than one label for that item (``n_disjunctive``)."""
    n = len(labels)
    agree = 0
    n_disjunctive = 0
    for row in labels.values():
        options = str(row.get("expected_label") or "").split("|")
        if len(options) > 1:
            n_disjunctive += 1
        if row.get("final_label") in options:
            agree += 1
    return {"n": n, "agree": agree, "accuracy": _share(agree, n), "n_disjunctive": n_disjunctive}


def _dominant_miss_cell(confusion: Mapping[str, Mapping[str, int]]) -> dict[str, Any] | None:
    """If every off-diagonal (miss) count in a confusion table falls in exactly one
    (adjudicated, predicted) cell, return ``{"gold": ..., "pred": ..., "n": ...}`` for that
    cell; otherwise ``None`` (no misses, or misses spread across more than one cell)."""
    cells = [(g, p, n) for g, row in confusion.items() for p, n in row.items() if g != p and n]
    if len(cells) != 1:
        return None
    g, p, n = cells[0]
    return {"gold": g, "pred": p, "n": n}


def hss_vs_annotation_block(
    results_dir: Path, labels_path: Path, *, name: str = "hss",
) -> dict[str, Any]:
    """Score each run's ``predicted_status`` (mapped through ``STATUS_TO_ANNOTATION_LABEL``)
    against a blind model annotation's ``final_label``, joined on ``item_id``; reports
    accuracy, per-label P/R/F1, a confusion table and Cohen's kappa per run, plus the
    construction-vs-adjudicated agreement computed once from the labels file. Does not call
    the verifier or the LLM: it only reads the committed run files and the labels CSV.

    ``name`` selects which ``<name>_run<A|B>.jsonl`` pair to score: ``"hss"`` (default) for
    the HSS-test-v3 set, ``"real"`` for the real-claims set
    (``real_annotation_adjudicated_v3.csv``); both are read with the plain (non-``auto``)
    name, matching how each set's frozen run files are actually named in ``results/v3/``.

    Each run also reports ``accuracy_construction_tolerant``: a prediction counts as correct
    whenever it falls inside that item's construction-rule ``expected_label`` set (pipe-joined
    when disjunctive), the same tolerance Table E2-b uses to score ``unsupported`` and
    ``needs_nuance`` as equally correct on the ten ``altered`` items. ``accuracy`` itself stays
    a strict single-label comparison against ``final_label``, so the two numbers can disagree
    on exactly the cell the construction rule leaves ambiguous; ``dominant_miss_cell`` names
    that cell when it accounts for every strict miss. ``n_adjudicated_needs_nuance`` is the
    count of adjudicated items whose ``final_label`` is ``needs_nuance`` (0 for the committed
    round-2 set), so a reader can see whether the ``unsupported -> needs_nuance`` miss cell
    reflects an adjudicated ``needs_nuance`` stratum or not."""
    labels = load_annotation_labels(labels_path)
    runs: dict[str, Any] = {}
    for run in RUNS:
        data = load_run(results_dir, name, run)
        if data is None:
            continue
        rows, _meta = data
        pairs = [
            (labels[r["item_id"]]["final_label"],
             STATUS_TO_ANNOTATION_LABEL.get(str(r.get("predicted_status")), "unsupported"),
             str(labels[r["item_id"]].get("expected_label") or "").split("|"))
            for r in rows if r.get("item_id") in labels
        ]
        gold = [g for g, _, _ in pairs]
        pred = [p for _, p, _ in pairs]
        n_tolerant = sum(1 for _, p, options in pairs if p in options)
        confusion = confusion_matrix(gold, pred, ANNOTATION_LABELS, ANNOTATION_LABELS)
        runs[run] = {
            "n": len(pairs),
            "accuracy": accuracy(gold, pred),
            "accuracy_construction_tolerant": _share(n_tolerant, len(pairs)),
            "per_label": per_class_prf(gold, pred, ANNOTATION_LABELS),
            "confusion": confusion,
            "dominant_miss_cell": _dominant_miss_cell(confusion),
            "kappa": cohens_kappa(gold, pred),
        }
    return {
        "labels_source": str(labels_path),
        "n_labels": len(labels),
        "n_adjudicated_needs_nuance": sum(
            1 for row in labels.values() if row.get("final_label") == "needs_nuance"
        ),
        "runs": runs,
        "construction_vs_adjudicated": construction_vs_annotation(labels),
    }


# --------------------------------------------------------------------------------------
# --compare-dir: v1-vs-v2 delta
# --------------------------------------------------------------------------------------

# (markdown label, dotted path into one run's summary block) -- SciFact and HSS each have
# their own metric set because `scifact_run_summary`/`hss_run_summary` are shaped differently.
COMPARE_METRICS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "scifact": (
        ("macro_f1", ("macro_f1",)),
        ("accuracy", ("accuracy",)),
        ("verified precision", ("strict", "verified", "precision")),
        ("verified recall", ("strict", "verified", "recall")),
        ("unsupported precision", ("strict", "unsupported", "precision")),
        ("unsupported recall", ("strict", "unsupported", "recall")),
    ),
    "hss": (
        ("accuracy", ("accuracy",)),
    ),
}


def _get_path(block: Mapping[str, Any] | None, path: tuple[str, ...]) -> Any:
    node: Any = block
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def compare_versions(summary: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Per (set, run, metric) delta of *summary* (this ``--results-dir``, e.g. v2) against
    *other* (another directory's already-written ``summary.json``, e.g. v1's
    ``--compare-dir``). A run or metric missing from either side compares as ``None`` (printed
    as ``-`` by ``common.fmt``); the delta is ``this - other`` and is only computed when both
    sides are numeric.
    """
    out: dict[str, Any] = {}
    for set_name, metrics in COMPARE_METRICS.items():
        runs_this = (summary.get(set_name) or {}).get("runs") or {}
        runs_other = (other.get(set_name) or {}).get("runs") or {}
        set_out: dict[str, Any] = {}
        for run in RUNS:
            this_block = runs_this.get(run)
            other_block = runs_other.get(run)
            if this_block is None and other_block is None:
                continue
            row: dict[str, Any] = {}
            for label, path in metrics:
                a = _get_path(other_block, path)
                b = _get_path(this_block, path)
                both_numeric = isinstance(a, (int, float)) and isinstance(b, (int, float))
                delta = (b - a) if both_numeric else None
                row[label] = {"other": a, "this": b, "delta": delta}
            set_out[run] = row
        if set_out:
            out[set_name] = set_out
    return out


def _md_compare(compare: Mapping[str, Any], compare_dir: str | None) -> list[str]:
    lines = [
        f"## Table E2-e prompt-version comparison (this `--results-dir` vs "
        f"`{compare_dir}`)",
        "",
    ]
    for set_name, runs in compare.items():
        rows = []
        for run, row in runs.items():
            for metric, vals in row.items():
                rows.append([
                    run, metric, fmt(vals["other"]), fmt(vals["this"]), fmt(vals["delta"]),
                ])
        if not rows:
            continue
        lines.append(f"### {set_name}")
        lines.append("")
        lines.append(markdown_table(
            ["run", "metric", "compare-dir", "results-dir", "delta (results-dir - compare-dir)"],
            rows,
        ))
        lines.append("")
    return lines


def baseline_n_mismatch(baseline: Mapping[str, Any] | None, runs: Mapping[str, Any]) -> bool:
    """True when a baseline file exists and its ``n`` differs from any LLM run's ``n``
    (a stale baseline from an earlier ``--limit`` run must not sit next to the full runs)."""
    if not baseline or not runs:
        return False
    return any(s.get("n") != baseline.get("n") for s in runs.values())


def _baseline_label(block: Mapping[str, Any]) -> str:
    base = block.get("baseline") or {}
    if block.get("baseline_n_mismatch"):
        run_ns = sorted({s.get("n") for s in (block.get("runs") or {}).values()})
        return (f"lexical baseline (n mismatch: {base.get('n')} vs "
                f"{'/'.join(str(n) for n in run_ns)})")
    return "lexical baseline"


# (plain name, dev/alternate-run name) per block key, for the ``--<key>-name`` selectors.
# "real" (the real-claims set, added after this brief's own section 4) has no dev/alternate
# name of its own -- ``real_run<X>.jsonl`` is the only shape ``run_hss.py --name real`` ever
# writes -- so both tuple slots are the same plain name and no ``--real-name`` CLI selector is
# exposed; ``resolve_block_name``'s ``auto`` default still resolves it correctly.
_NAME_CANDIDATES: dict[str, tuple[str, str]] = {
    "scifact": ("scifact", "scifact-train"),
    "hss": ("hss", "hss-band2"),
    "real": ("real", "real"),
}


def resolve_block_name(results_dir: Path, block: str, selector: str) -> str:
    """The ``<name>_run<X>.jsonl`` prefix actually used for *block* under *results_dir*.

    ``selector`` is ``"auto"``, the block's plain name or its dev/alternate name.  An
    explicit (non-``"auto"``) selector is used exactly as given, even when no matching run
    file exists -- :func:`load_run` then resolves to an empty ``runs`` dict, never a silent
    switch to the other name. ``"auto"`` (default) tries the plain name first (either run
    file present) and falls back to the alternate name; missing both, it resolves to the
    plain name (brief section 4: ``load_run`` opens ``<name>_run<run>.jsonl``, and a
    directory holding neither carries no run to load either way).
    """
    plain, alt = _NAME_CANDIDATES[block]
    if selector != "auto":
        return selector
    for candidate in (plain, alt):
        if any((results_dir / f"{candidate}_run{run}.jsonl").exists() for run in RUNS):
            return candidate
    return plain


def summarise(
    results_dir: Path, *, hss_name: str = "auto", scifact_name: str = "auto",
) -> dict[str, Any]:
    """One block per E2 set (keys ``"scifact"``, ``"hss"``, ``"real"``, unaffected by which
    run-file name resolved: brief section 4). ``hss_name``/``scifact_name`` select the
    ``<name>_run<X>.jsonl`` prefix within *results_dir* (``"auto"``, :func:`resolve_block_name`)
    so a dev-tuning directory (``scifact-train``, ``hss-band2``) is read as readily as a frozen
    test directory. The ``"real"`` block (the real-claims set, ``real_run<X>.jsonl``, added
    after this brief's own section 4) is scored with the same :func:`hss_run_summary` -- the
    row shape is identical -- but always under the plain ``"real"`` name, since no dev/alternate
    real-claims run file exists; it is present (with empty ``runs``) even when no
    ``real_run*.jsonl`` file exists, matching how ``"hss"``/``"scifact"`` behave when missing.
    ``real_run*.jsonl`` rows carry ``expected: null`` and ``correct: False`` by construction (no
    gold label; scoring is against the blind model annotation, not the construction rule, since
    a real reviewer's claim has none), so this block's own ``accuracy``/``correct``/``by_rule``
    figures are not meaningful -- only
    ``agreement_AB`` (the A/B kappa) is. The meaningful accuracy figures live in
    ``real_vs_annotation`` (see :func:`hss_vs_annotation_block`, ``name="real"``).
    """
    out: dict[str, Any] = {"generated": now_iso(), "results_dir": str(results_dir)}
    selectors = {"scifact": scifact_name, "hss": hss_name, "real": "real"}
    fns: dict[str, Any] = {
        "scifact": scifact_run_summary, "hss": hss_run_summary, "real": hss_run_summary,
    }
    for key, fn in fns.items():
        file_name = resolve_block_name(results_dir, key, selectors[key])
        loaded = {run: load_run(results_dir, file_name, run) for run in RUNS}
        block: dict[str, Any] = {"runs": {}}
        for run, data in loaded.items():
            if data is not None:
                block["runs"][run] = fn(*data)
        if all(loaded.get(r) for r in RUNS):
            block["agreement_AB"] = agreement(loaded["A"][0], loaded["B"][0])
        else:
            block["agreement_AB"] = None
        block["baseline"] = read_json(results_dir / f"{key}_baseline.json")
        block["baseline_n_mismatch"] = baseline_n_mismatch(block["baseline"], block["runs"])
        if block["baseline_n_mismatch"]:
            print(
                f"WARNING: {key}_baseline.json covers n={block['baseline']['n']} items but "
                f"the LLM runs cover {sorted({s['n'] for s in block['runs'].values()})}; "
                "regenerate the baseline with --match-run against the current run file",
                file=sys.stderr,
            )
        out[key] = block
    return out


def figure_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(set_name: str, series: str, values: Mapping[str, Any]) -> None:
        for metric in FIGURE_METRICS:
            rows.append({"set": set_name, "series": series, "metric": metric,
                         "value": values.get(metric)})

    sci = summary.get("scifact") or {}
    kappa = (sci.get("agreement_AB") or {}).get("kappa_status")
    for run, s in (sci.get("runs") or {}).items():
        v = s["strict"]["verified"]
        add("scifact", f"LLM run {run}", {
            "verified_precision": v["precision"], "verified_recall": v["recall"],
            "verified_f1": v["f1"], "accuracy": s["accuracy"],
            "quote_fidelity": s["quote_fidelity"]["verbatim_share"], "kappa_AB": kappa,
        })
    base = sci.get("baseline")
    if base:
        v = (base.get("per_class") or {}).get("verified") or {}
        add("scifact", "lexical baseline", {
            "verified_precision": v.get("precision"), "verified_recall": v.get("recall"),
            "verified_f1": v.get("f1"), "accuracy": base.get("accuracy"),
            "quote_fidelity": None, "kappa_AB": None,
        })
    hss = summary.get("hss") or {}
    kappa = (hss.get("agreement_AB") or {}).get("kappa_status")
    for run, s in (hss.get("runs") or {}).items():
        v = s["verified_binary"]
        add("hss", f"LLM run {run}", {
            "verified_precision": v["precision"], "verified_recall": v["recall"],
            "verified_f1": v["f1"], "accuracy": s["accuracy"],
            "quote_fidelity": s["quote_fidelity"]["verbatim_share"], "kappa_AB": kappa,
        })
    base = hss.get("baseline")
    if base:
        v = base.get("verified_binary") or {}
        add("hss", "lexical baseline", {
            "verified_precision": v.get("precision"), "verified_recall": v.get("recall"),
            "verified_f1": v.get("f1"), "accuracy": base.get("accuracy"),
            "quote_fidelity": None, "kappa_AB": None,
        })
    return rows


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------


def _md_scifact(summary: Mapping[str, Any]) -> list[str]:
    sci = summary.get("scifact") or {}
    lines = ["## Table E2-a SciFact (dev set, one item per cited claim-document row)", ""]
    first_run = next(iter((sci.get("runs") or {}).values()), None)
    dup_pairs = (first_run or {}).get("duplicate_pairs") or []
    if dup_pairs:
        n_total, n_distinct = first_run["n"], first_run.get("n_distinct_pairs", first_run["n"])
        dup_text = "; ".join(
            f"claim {d['claim_id']} cites document {d['doc_id']} "
            + ("twice" if d["count"] == 2 else f"{d['count']} times")
            for d in dup_pairs
        )
        lines.append(
            f"n = {n_total} rows over {n_distinct} distinct claim-document pairs: {dup_text} "
            "in the SciFact dev set's own `cited_doc_ids`, so `run_scifact.build_items` emits "
            "that pair twice; both copies are sent to the model, billed and counted in every "
            "metric below (both copies score the same predicted status in both runs, so this "
            "is a rounding-level effect, not a scoring error). Runs A and B build items from "
            "the same dev set, so the duplicate is identical in both."
        )
        lines.append("")
    rows = []
    for run, s in (sci.get("runs") or {}).items():
        v, u, lenient = s["strict"]["verified"], s["strict"]["unsupported"], s["lenient"]
        rows.append([
            f"LLM run {run}", s["n"], v["precision"], v["recall"], v["f1"], u["precision"],
            u["recall"], u["f1"], s["macro_f1"], s["accuracy"], lenient["precision"],
            lenient["recall"], lenient["f1"], s["quote_fidelity"]["verbatim_share"],
            s["quote_fidelity"]["casefold_share"], s["n_errors"],
        ])
    base = sci.get("baseline")
    if base:
        pv = (base.get("per_class") or {}).get("verified") or {}
        pu = (base.get("per_class") or {}).get("unsupported") or {}
        rows.append([
            _baseline_label(sci), base.get("n"), pv.get("precision"), pv.get("recall"),
            pv.get("f1"), pu.get("precision"), pu.get("recall"), pu.get("f1"),
            base.get("macro_f1"), base.get("accuracy"), pv.get("precision"), pv.get("recall"),
            pv.get("f1"), None, None, 0,
        ])
    lines.append(markdown_table(
        ["run", "n", "verified P", "verified R", "verified F1", "unsupported P",
         "unsupported R", "unsupported F1", "macro-F1", "accuracy", "lenient P", "lenient R",
         "lenient F1", "quote verbatim", "quote verbatim (casefold)", "errors"],
        rows,
    ))
    lines.append("")
    lines.append(
        "Strict: gold SUPPORT -> verified, CONTRADICT/NOT_ENOUGH_INFO -> unsupported; "
        "needs_nuance / no_full_text / error predictions count as misses and never as hits. "
        "Lenient: verified vs not-verified. A class that is never predicted has precision "
        "undefined (-) and F1 = 0 by convention; macro-F1 averages over both classes."
    )
    for run, s in (sci.get("runs") or {}).items():
        lines += ["", f"Confusion (run {run}; rows = gold, columns = predicted):", ""]
        conf = s["confusion"]
        cols = list(next(iter(conf.values())).keys()) if conf else []
        lines.append(markdown_table(["gold", *cols], [[g, *[conf[g][c] for c in cols]]
                                                     for g in conf]))
        nn = s["needs_nuance_by_gold"]
        lines.append("")
        lines.append(f"needs_nuance by gold class (run {run}): " + ", ".join(
            f"{g}={n}" for g, n in nn.items()))
    agree = sci.get("agreement_AB")
    if agree:
        lines += ["", _agreement_line(agree, binary=True)]
        if agree.get("kappa_undefined"):
            lines += ["", KAPPA_FOOTNOTE]
    return lines


def _md_label_mapping_sensitivity(summary: Mapping[str, Any]) -> list[str]:
    sci = summary.get("scifact") or {}
    runs = sci.get("runs") or {}
    if not runs:
        return []
    lines = [
        "## Table E2-f Label-mapping sensitivity (SciFact)",
        "",
        "SciFact's three gold labels (SUPPORT, CONTRADICT, NOT_ENOUGH_INFO) are scored against "
        "the verifier's four statuses under three mappings: strict (Table E2-a's own "
        "convention: SUPPORT -> verified, CONTRADICT/NOT_ENOUGH_INFO -> unsupported, a "
        "predicted needs_nuance / no_full_text / error is a miss for its gold class and never "
        "a hit); lenient (identical, except a predicted needs_nuance counts as verified); and "
        "three-way, available only for rows that carry the per-assertion `verdict` field "
        "(prompt v2), which splits a predicted unsupported row by whether any assertion was "
        "`contradicted`, scoring CONTRADICT against that split bucket instead of the class it "
        "otherwise shares with NOT_ENOUGH_INFO.",
        "",
    ]
    rows: list[list[Any]] = []
    for run, s in runs.items():
        lms = s.get("label_mapping_sensitivity") or {}
        for key, label in MAPPING_LABELS.items():
            m = lms.get(key) or {}
            pc = m.get("per_class") or {}
            v, u = pc.get("verified") or {}, pc.get("unsupported") or {}
            rows.append([
                f"run {run}", label, v.get("precision"), v.get("recall"), v.get("f1"),
                u.get("precision"), u.get("recall"), u.get("f1"), m.get("macro_f1"),
                m.get("contradict_recall"), m.get("n_gold_contradict_predicted_verified"),
            ])
    lines.append(markdown_table(
        ["run", "mapping", "verified P", "verified R", "verified F1", "unsupported P",
         "unsupported R", "unsupported F1", "macro-F1", "CONTRADICT recall",
         "gold-CONTRADICT -> verified"],
        rows,
    ))
    lines.append("")

    three_way_rows: list[list[Any]] = []
    unavailable_reason = None
    for run, s in runs.items():
        tw = (s.get("label_mapping_sensitivity") or {}).get("three_way_contradiction_signal") or {}
        if not tw.get("available"):
            unavailable_reason = unavailable_reason or tw.get("reason")
            continue
        pc = tw.get("per_class") or {}
        v = pc.get("verified") or {}
        uc = pc.get("unsupported_contradicted") or {}
        uo = pc.get("unsupported_other") or {}
        three_way_rows.append([
            f"run {run}", v.get("precision"), v.get("recall"), v.get("f1"),
            uc.get("precision"), uc.get("recall"), uc.get("f1"),
            uo.get("precision"), uo.get("recall"), uo.get("f1"), tw.get("macro_f1"),
            tw.get("contradict_recall"), tw.get("n_gold_contradict_predicted_verified"),
        ])
    if three_way_rows:
        lines.append(
            "Three-way mapping (predicted unsupported split by the per-assertion "
            "contradiction signal; the contradicted bucket is where a gold CONTRADICT row "
            "should land):"
        )
        lines.append("")
        lines.append(markdown_table(
            ["run", "verified P", "verified R", "verified F1", "contradicted-bucket P",
             "contradicted-bucket R", "contradicted-bucket F1", "other-bucket P",
             "other-bucket R", "other-bucket F1", "macro-F1", "CONTRADICT recall",
             "gold-CONTRADICT -> verified"],
            three_way_rows,
        ))
        lines.append("")
    elif unavailable_reason:
        lines.append(f"Third mapping skipped for every run above: {unavailable_reason}.")
        lines.append("")
    lines.append(
        "Sentence the manuscript may state: the mapping choice moves the headline numbers "
        "only at the margin the model's own needs_nuance and per-assertion signals create; "
        "CONTRADICT recall and the count of gold-CONTRADICT rows predicted verified are the "
        "figures a reviewer should read as the safety check, not macro-F1 alone."
    )
    return lines


def _md_hss(summary: Mapping[str, Any]) -> list[str]:
    hss = summary.get("hss") or {}
    lines = ["## Table E2-b HSS set (applied-linguistics full texts; labels are by construction)",
             ""]
    for run, s in (hss.get("runs") or {}).items():
        lines.append(
            f"Run {run}: overall accuracy {fmt(s['accuracy'])} ({s['correct']}/{s['n']}); "
            f"needs_nuance answers: {s.get('n_needs_nuance', 0)}."
        )
        lines.append("")
        rows = []
        for rule, b in s["by_rule"].items():
            rows.append([
                rule, b["n"], " / ".join(b["expected"] or []), b["correct"], b["accuracy"],
                b.get("needs_nuance"),
                ", ".join(f"{k}={v}" for k, v in sorted(b["status_counts"].items())),
                b["quote_fidelity"]["verbatim_share"],
            ])
        lines.append(markdown_table(
            ["rule", "n", "expected", "correct", "accuracy", "needs_nuance",
             "predicted statuses", "quote verbatim (verified rows)"],
            rows,
        ))
        lines.append("")
    base = hss.get("baseline")
    runs = hss.get("runs") or {}
    if runs or base:
        def _per_rule(by_rule: Mapping[str, Any]) -> str:
            return ", ".join(f"{r}={b.get('correct')}/{b.get('n')}" for r, b in by_rule.items())

        combo_rows = []
        for run, s in runs.items():
            v = s.get("verified_binary") or {}
            combo_rows.append([
                f"LLM run {run}", s.get("n"), v.get("n"), s.get("correct"), s.get("accuracy"),
                _per_rule(s.get("by_rule") or {}), v.get("precision"), v.get("recall"),
                v.get("f1"),
            ])
        if base:
            v = base.get("verified_binary") or {}
            combo_rows.append([
                _baseline_label(hss), base.get("n"), v.get("n"), base.get("correct"),
                base.get("accuracy"), _per_rule(base.get("by_rule") or {}), v.get("precision"),
                v.get("recall"), v.get("f1"),
            ])
        lines.append(markdown_table(
            ["system", "n", "binary n", "correct", "accuracy", "correct per rule",
             "verified P", "verified R", "verified F1"],
            combo_rows,
        ))
        lines.append("")
    if base:
        v = base.get("verified_binary") or {}
        n_det = v.get("n_excluded_deterministic")
        lines.append(
            "Lexical baseline: max token-Jaccard between the claim and any chunk sentence "
            f">= {base.get('threshold')} -> verified, else unsupported; items without chunks "
            "-> no_full_text. It cannot answer needs_nuance. Its binary verified view "
            f"(binary n, P/R/F1) excludes the {n_det if n_det is not None else '-'} no-chunk "
            "item(s), which are rejected by construction, on the same denominator as the "
            "LLM rows above."
        )
        lines.append("")
    agree = hss.get("agreement_AB")
    if agree:
        lines.append(_agreement_line(agree, binary=False))
        if agree.get("kappa_undefined"):
            lines += ["", KAPPA_FOOTNOTE]
    lines.append("")
    lines.append(
        "A/B agreement pairs only items with a model decision in both runs: deterministic "
        "no_full_text items (never sent to the model) and error rows are excluded and counted. "
        "The binary verified metrics exclude error rows (reported as errors), so a failed call "
        "is never scored as a correct rejection, and the deterministic no_full_text items, so "
        "a by-construction rejection is never a true negative."
    )
    lines.append("")
    lines.append(
        "needs_nuance counts answers of that status per rule; for paraphrase items they are "
        "reported here and are not scored as correct (expected: verified)."
    )
    lines.append("")
    ci_note = (
        "All figures above are point estimates on the stated n with no confidence interval"
    )
    first_run = next(iter((hss.get("runs") or {}).values()), None)
    if first_run is not None and first_run.get("n"):
        lo, hi = wilson_interval(first_run.get("correct", 0), first_run["n"])
        half_width = (hi - lo) / 2 if lo is not None and hi is not None else None
        ci_note += (
            f"; the 95 % Wilson interval of an accuracy of {fmt(first_run.get('accuracy'))} "
            f"on n={first_run['n']} is about +-{fmt(half_width, 2)}"
        )
    ci_note += (
        ", so a few-point difference between the LLM run and the lexical baseline, or "
        "between runs A and B, is not interpreted as a difference (see README.md, "
        "'No confidence intervals')."
    )
    lines.append(ci_note)
    return lines


def _md_provenance(summary: Mapping[str, Any]) -> list[str]:
    lines = ["## Table E2-c provenance/cost/latency", ""]
    rows = []
    for name in ("scifact", "hss"):
        for run, s in ((summary.get(name) or {}).get("runs") or {}).items():
            p = s["provenance"]
            rows.append([
                f"{name} run {run}", p["model_configured"], ", ".join(p["model_reported"] or []),
                len(p["system_fingerprints"] or []), p["temperature"], p["prompt_version"],
                p["concurrency"], p["n_errors"], p["n_deterministic"], p["tokens"]["input"],
                p["tokens"]["output"], p["tokens"]["cache_read"], p["total_cost"],
                p["latency"]["median"], p["latency"]["p90"], (p["finished"] or "")[:10],
            ])
    lines.append(markdown_table(
        ["run", "model configured", "model reported", "fingerprints", "temperature",
         "prompt version", "concurrency", "errors", "deterministic", "input tok", "output tok",
         "cache-read tok", "cost USD", "latency median s", "latency p90 s", "date"],
        rows,
    ))
    basis = {s["provenance"].get("cost_basis") for n in ("scifact", "hss")
             for s in ((summary.get(n) or {}).get("runs") or {}).values()}
    basis.discard(None)
    if basis:
        lines += ["", "Cost basis: " + "; ".join(sorted(basis)) + "."]
    lines += [
        "",
        "Latency is wall-clock per model call measured while up to 'concurrency' calls were "
        "in flight (harness default 8; production uses analysis_concurrency = 4), so medians "
        "and p90 include provider-side queueing.",
    ]
    return lines


def _md_machine_reasons(summary: Mapping[str, Any]) -> list[str]:
    """Table of guard-slug firing counts (code guards) per run, both sets."""
    rows = []
    for name in ("scifact", "hss"):
        for run, s in ((summary.get(name) or {}).get("runs") or {}).items():
            mr = s.get("machine_reasons") or {}
            by_slug = mr.get("by_slug") or {}
            slug_txt = "; ".join(f"{slug}: {n}" for slug, n in sorted(by_slug.items())) or "-"
            rows.append([
                f"{name} run {run}", mr.get("n_guarded", 0), mr.get("guarded_count", 0),
                mr.get("n", 0), slug_txt,
            ])
    lines = ["## Guard reasons (machine_reasons)", ""]
    if not rows:
        lines.append("No run available.")
        return lines
    lines.append(markdown_table(
        ["run", "rows with a reason", "status changed (guarded_count)", "n", "slug: count"],
        rows,
    ))
    lines += [
        "",
        "A row's own status is never changed twice in a way this table would double count: "
        "each guard that fires appends one slug to the row's own machine_reasons list, so a "
        "row with two reasons is counted once per slug and once in 'rows with a reason'. That "
        "column does not imply the guard changed anything: a report-only guard, or a "
        "status-changing guard that caps a status onto the value the model already gave, still "
        "appends a slug. 'status changed (guarded_count)' is the stricter figure: it counts "
        "only rows where predicted_status differs from the model's own model_status, i.e. a "
        "guard actually made the status stricter. Rows from a prompt version that predates "
        "this field report zero of both counts, not an error.",
    ]
    return lines


def _md_hss_vs_annotation(
    summary: Mapping[str, Any], *, key: str = "hss_vs_annotation",
    table_id: str = "E2-d", set_label: str = "HSS set", has_construction_rule: bool = True,
) -> list[str]:
    """Renders *key* (``"hss_vs_annotation"`` or ``"real_vs_annotation"``) as a markdown table;
    *table_id*/*set_label* vary the heading and prose so the same renderer serves both the
    HSS-test-v3 set and the real-claims set without duplicating this function.
    ``has_construction_rule=False`` (the real-claims set: every row's ``expected_label``
    column is empty, there being no construction rule for a real reviewer's claim) drops the
    construction-tolerant explanation, the dominant-miss-cell note and the closing
    construction-vs-adjudicated line, all of which would otherwise report a vacuous 0 %
    figure against an empty label set rather than an absent one."""
    block = summary.get(key)
    if not block:
        return []
    lines = [
        f"## Table {table_id} {set_label} vs blind model annotation "
        "(independent check on the construction labels)",
        "",
        f"{block['n_labels']} items scored against "
        f"`{Path(block['labels_source']).name}`: a blind triple Claude-Opus annotation of the "
        f"{set_label}, adjudicated on disagreement (see claims/annotation/PROVENANCE.md; these "
        "are model annotations, not expert human annotations). Verifier status is mapped "
        "verified -> verified, needs_nuance -> needs_nuance, unsupported/error -> unsupported, "
        "no_full_text -> no_full_text before comparison with the adjudicated `final_label`.",
        "",
    ]
    rows = []
    for run, s in block["runs"].items():
        rows.append([
            f"LLM run {run}", s["n"], s["accuracy"], s["accuracy_construction_tolerant"],
            s["kappa"], *[s["per_label"][c]["f1"] for c in ANNOTATION_LABELS],
        ])
    lines.append(markdown_table(
        ["run", "n", "accuracy (strict)", "construction-tolerant (legacy, not reported)",
         "kappa vs annotation", *[f"{c} F1" for c in ANNOTATION_LABELS]],
        rows,
    ))
    lines.append("")
    if has_construction_rule:
        lines.append(
            "`accuracy (strict)` is a single-label comparison against the adjudicated "
            "`final_label`; unlike Table E2-b, which scores `unsupported` and `needs_nuance` as "
            "equally correct on the ten `altered` items where the construction rule admits "
            "either label, this column credits only the one adjudicated label. "
            "`construction-tolerant (legacy, not reported)` instead applies that same tolerance "
            "here, crediting a prediction whenever it falls inside the item's construction-rule "
            "label set; it is retained for continuity with earlier reports but is not one of "
            "the accepted metrics (the adjudicators used `unsupported` for all 30 items "
            "and `needs_nuance` for none, so the two columns should agree once prompt v2 "
            "ships). The adjudicated label set contains "
            f"{block.get('n_adjudicated_needs_nuance', 0)} `needs_nuance` label(s)."
        )
        for run, s in block["runs"].items():
            cell = s.get("dominant_miss_cell")
            if cell:
                lines.append(
                    f"Run {run}: every miss counted against `accuracy (strict)` falls in the "
                    f"single cell adjudicated `{cell['gold']}` -> verifier `{cell['pred']}` "
                    f"(n={cell['n']}), which `construction-tolerant (legacy, not reported)` "
                    "counts as correct."
                )
    else:
        lines.append(
            "This set has no construction rule (every item's `expected_label` is empty: a "
            "real reviewer's claim, not a constructed one), so `construction-tolerant "
            "(legacy, not reported)` and the construction-vs-adjudicated line below are "
            "omitted here; `accuracy (strict)` against the adjudicated `final_label` is the "
            "only accuracy figure for this set."
        )
    lines.append("")
    for run, s in block["runs"].items():
        conf = s["confusion"]
        cols = list(ANNOTATION_LABELS)
        lines += [f"Confusion (run {run}; rows = adjudicated label, columns = verifier status):",
                  ""]
        lines.append(markdown_table(
            ["adjudicated", *cols], [[g, *[conf[g][c] for c in cols]] for g in cols]
        ))
        lines.append("")
    if has_construction_rule:
        cva = block["construction_vs_adjudicated"]
        lines.append(
            f"Construction labels vs the adjudicated labels: {cva['agree']} of {cva['n']} "
            f"agree ({fmt(cva['accuracy'])}); {cva['n_disjunctive']} of {cva['n']} items admit "
            "two labels under the construction rule, so that figure is a consistency check on "
            "the construction rule rather than an independent confirmation of it."
        )
    return lines


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# E2 claim verification -- summary", "", f"Generated {summary['generated']}.", ""]
    lines += _md_scifact(summary)
    mapping_sensitivity = _md_label_mapping_sensitivity(summary)
    if mapping_sensitivity:
        lines += [""] + mapping_sensitivity
    lines += [""] + _md_hss(summary) + [""] + _md_provenance(summary)
    lines += [""] + _md_machine_reasons(summary)
    hss_vs_annotation = _md_hss_vs_annotation(summary)
    if hss_vs_annotation:
        lines += [""] + hss_vs_annotation
    real_vs_annotation = _md_hss_vs_annotation(
        summary, key="real_vs_annotation", table_id="E2-g", set_label="real claims set",
        has_construction_rule=False,
    )
    if real_vs_annotation:
        lines += [""] + real_vs_annotation
    version_comparison = summary.get("version_comparison")
    if version_comparison:
        lines += [""] + _md_compare(version_comparison, summary.get("compare_dir"))
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument(
        "--labels", type=Path, default=None,
        help=(
            "CSV of adjudicated final_label per item_id (e.g. "
            "annotation/hss_annotation_adjudicated_v2.csv). Does not run the verifier or call "
            "an LLM: scores the committed hss_run<A|B>.jsonl against the CSV and adds the "
            "hss_vs_annotation block (accuracy, per-label P/R/F1, confusion, Cohen's kappa, "
            "and construction-vs-adjudicated agreement) without touching any other block."
        ),
    )
    ap.add_argument(
        "--real-labels", type=Path, default=None,
        help=(
            "CSV of adjudicated final_label per item_id for the real-claims set (e.g. "
            "annotation/real_annotation_adjudicated_v3.csv). Same scoring as --labels "
            "(hss_vs_annotation_block, name='real'), scored against the committed "
            "real_run<A|B>.jsonl instead, and adds a real_vs_annotation block (independent "
            "of, and in addition to, --labels/hss_vs_annotation) without touching any other "
            "block. Does not run the verifier or call an LLM."
        ),
    )
    ap.add_argument(
        "--compare-dir", type=Path, default=None,
        help=(
            "another results directory whose already-written summary.json is diffed "
            "against this run's summary and rendered as a Table E2-e delta in summary.md, "
            "in addition to (not instead of) every other table. Does not run the verifier "
            "or call an LLM."
        ),
    )
    ap.add_argument(
        "--hss-name", choices=("hss", "hss-band2", "auto"), default="auto",
        help="<name>_run<X>.jsonl prefix for the hss block (band 2 dev set: hss-band2)",
    )
    ap.add_argument(
        "--scifact-name", choices=("scifact", "scifact-train", "auto"), default="auto",
        help="<name>_run<X>.jsonl prefix for the scifact block (--split train: scifact-train)",
    )
    return resolve_path_args(ap.parse_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarise(
        args.results_dir, hss_name=args.hss_name, scifact_name=args.scifact_name,
    )
    if not any(summary[key]["runs"] for key in ("scifact", "hss", "real")):
        print(f"no run files found in {args.results_dir}")
        return 1
    if args.labels:
        summary["hss_vs_annotation"] = hss_vs_annotation_block(args.results_dir, args.labels)
    if args.real_labels:
        summary["real_vs_annotation"] = hss_vs_annotation_block(
            args.results_dir, args.real_labels, name="real",
        )
    if args.compare_dir:
        other = read_json(args.compare_dir / "summary.json")
        if other is None:
            print(
                f"WARNING: --compare-dir {args.compare_dir} has no summary.json; "
                "skipping the version comparison",
                file=sys.stderr,
            )
        else:
            summary["compare_dir"] = str(args.compare_dir)
            summary["version_comparison"] = compare_versions(summary, other)
    write_json(args.results_dir / "summary.json", summary)
    write_json(args.results_dir / "figure4_claims.json", {
        "generated": summary["generated"], "metrics": list(FIGURE_METRICS),
        "rows": figure_rows(summary),
    })
    (args.results_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"wrote summary.json, summary.md, figure4_claims.json in {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

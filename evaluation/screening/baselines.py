"""Non-LLM screening baselines for one SYNERGY dataset (evaluation E1).

Interpreter: ``evaluation/.venv/Scripts/python`` or the system ``python`` (needs scikit-learn).

Baselines
---------
* **include-all** -- every record is included (recall 1, precision = prevalence, WSS 0).
* **TF-IDF cosine** -- cosine similarity between each record (title + abstract) and the
  protocol query, thresholded so that the number of included records equals a count taken
  from the LLM run (``--run`` file or an explicit ``--k``). ``--k-basis predicted`` (default)
  matches the count of records the run marked ``predicted == 1`` (a screener's INCLUDE
  decisions); ``--k-basis screened_in`` matches the count retained for further reading
  instead (``status`` in INCLUDE or NEEDS_REVIEW, falling back to ``predicted`` for a run with
  no ``status`` column), the basis a full-text-inclusion-criterion protocol needs because its
  screener routes every confirmed match to NEEDS_REVIEW rather than INCLUDE, leaving the
  INCLUDE count at zero. The basis used is recorded as ``k_basis`` in the output.
  ``--query rq+criteria`` (default, an
  approximation of the v2 user turn, which also shows the numbered criteria --
  not identical to it) appends the inclusion criteria to the research question;
  ``--query rq`` uses the research question alone; the mode is recorded as ``query_mode`` in
  the output. Because TF-IDF yields a ranking, WSS@95 is also reported for the ranking itself.

Both baselines are evaluated on exactly the records the LLM run *screened* (rows present in
the run file with ``predicted`` not null, so a ``--limit`` smoke run or a run with failed
batches and its baselines share the same n) and, by default, against ``primary_label``
(``--label primary``, ``protocol.extra.get("primary_label", "label_included")``,
"recomputed against label_included"); ``--label sensitivity`` scores the older
``protocol.label_field`` instead. Output: ``results/<dataset>_baselines.json``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    QUERY_MODES,
    Protocol,
    binary_metrics,
    confusion_counts,
    load_protocol,
    now_iso,
    predictions_matching_count,
    protocol_query_text,
    read_jsonl,
    recall,
    resolve_path_args,
    write_json,
    wss_from_ranking,
)

DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results" / "v3"
PROTOCOLS_DIR = HERE / "protocols"


def record_text(record: Mapping[str, Any]) -> str:
    return f"{record.get('title') or ''}. {record.get('abstract') or ''}".strip()


def tfidf_scores(documents: Sequence[str], query: str) -> list[float]:
    """Cosine similarity of ``query`` to each document in a shared TF-IDF space."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not documents:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([*documents, query])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    return [float(x) for x in sims]


def llm_inclusion_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for r in rows if r.get("predicted") == 1)


K_BASES: tuple[str, ...] = ("predicted", "screened_in")


def llm_screened_in_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count of rows the screener retained for further reading: ``status`` in INCLUDE or
    NEEDS_REVIEW, falling back to ``predicted == 1`` for a row with no ``status`` column (a
    v1-style run). Matches ``summarize.py``'s own status-with-fallback reading so a baseline
    built with ``--k-basis screened_in`` lines up with that module's ``screened_in_rate``."""
    count = 0
    for r in rows:
        status = r.get("status")
        if status is not None:
            if status in ("INCLUDE", "NEEDS_REVIEW"):
                count += 1
            continue
        if r.get("predicted") == 1:
            count += 1
    return count


def baseline_metrics(
    records: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    query: str,
    k: int,
) -> dict[str, Any]:
    """Include-all and TF-IDF (top-``k``) metrics for the given records/labels."""
    n = len(records)
    include_all = binary_metrics(confusion_counts(labels, [1] * n))
    scores = tfidf_scores([record_text(r) for r in records], query)
    preds, threshold = predictions_matching_count(scores, k)
    tfidf_counts = confusion_counts(labels, preds)
    achieved = recall(tfidf_counts)
    return {
        "n": n,
        "k": k,
        "include_all": include_all,
        "tfidf": {
            "threshold": threshold,
            "metrics": binary_metrics(tfidf_counts),
            "wss_at_95_ranking": wss_from_ranking(scores, labels, 0.95),
            "wss_at_achieved_recall_ranking": (
                None if achieved is None else wss_from_ranking(scores, labels, achieved)
            ),
        },
    }


def select_records(
    records: Sequence[Mapping[str, Any]],
    run_rows: Sequence[Mapping[str, Any]] | None,
    protocol: Protocol,
    *,
    label_field: str | None = None,
) -> tuple[list[Mapping[str, Any]], list[int]]:
    """Records (and gold labels) restricted to those the LLM run screened, if any.

    Rows with ``predicted = null`` (batches that failed after retries) are excluded so the
    baselines and the LLM are evaluated on the same n. ``label_field`` overrides which gold
    label is scored (``--label primary`` -> ``primary_label``, default
    ``"label_included"``); when omitted, ``protocol.label_field`` is used unchanged (the
    original default), so existing callers keep their exact behaviour.
    """
    field = label_field or protocol.label_field
    wanted = (
        {r["record_id"] for r in run_rows if r.get("predicted") is not None} if run_rows else None
    )
    chosen: list[Mapping[str, Any]] = []
    labels: list[int] = []
    for rec in records:
        if wanted is not None and rec["record_id"] not in wanted:
            continue
        label = rec.get(field)
        if label is None:
            continue
        chosen.append(rec)
        labels.append(int(label))
    return chosen, labels


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", default="A", help="LLM run whose inclusion count is matched (A|B)")
    ap.add_argument("--k", type=int, default=None, help="override the inclusion count to match")
    ap.add_argument(
        "--k-basis",
        choices=K_BASES,
        default="predicted",
        help="predicted (default): match the run's INCLUDE count (predicted == 1); "
        "screened_in: match the run's retained count (status INCLUDE or NEEDS_REVIEW), "
        "needed when a full-text inclusion criterion routes every confirmed match to "
        "NEEDS_REVIEW and leaves the INCLUDE count at zero. Ignored when --k is given.",
    )
    ap.add_argument(
        "--query",
        choices=QUERY_MODES,
        default="rq+criteria",
        help="TF-IDF query document: research question + criteria (default, an "
        "approximation of the v2 user turn, not identical to it) or rq alone",
    )
    ap.add_argument(
        "--label",
        choices=("primary", "sensitivity"),
        default="primary",
        help="primary (default): protocol.extra.get('primary_label', 'label_included'); "
        "sensitivity: protocol.label_field (the earlier default label)",
    )
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--protocols-dir", type=Path, default=PROTOCOLS_DIR)
    return resolve_path_args(ap.parse_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocols_dir / f"{args.dataset}.json")
    records = read_jsonl(args.data_dir / f"{args.dataset}.jsonl")
    if not records:
        raise SystemExit(f"no records for {args.dataset}; run fetch_synergy.py first")
    run_path = args.results_dir / f"{args.dataset}_run{args.run}.jsonl"
    run_rows = read_jsonl(run_path) if run_path.exists() else []
    if not run_rows and args.k is None:
        raise SystemExit(f"{run_path} not found; run run_screening.py first or pass --k")

    label_field = (
        protocol.extra.get("primary_label", "label_included")
        if args.label == "primary"
        else protocol.label_field
    )
    chosen, labels = select_records(records, run_rows or None, protocol, label_field=label_field)
    if args.k is not None:
        k = args.k
    elif args.k_basis == "screened_in":
        k = llm_screened_in_count(run_rows)
    else:
        k = llm_inclusion_count(run_rows)
    result = {
        "dataset": args.dataset,
        "generated": now_iso(),
        "run_file": run_path.name if run_rows else None,
        "label_field": label_field,
        "label_choice": args.label,
        "query_mode": args.query,
        "query_text": protocol_query_text(protocol, args.query),
        "n_unscreened_excluded": sum(1 for r in run_rows if r.get("predicted") is None),
        "k_basis": args.k_basis if args.k is None else "explicit",
        **baseline_metrics(chosen, labels, protocol_query_text(protocol, args.query), k),
    }
    out = args.results_dir / f"{args.dataset}_baselines.json"
    write_json(out, result)
    tf = result["tfidf"]["metrics"]
    print(
        f"wrote {out}: n={result['n']} k={k} include-all precision="
        f"{result['include_all']['precision']:.3f} | TF-IDF recall={tf['recall']} "
        f"precision={tf['precision']} WSS@95(ranking)={result['tfidf']['wss_at_95_ranking']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

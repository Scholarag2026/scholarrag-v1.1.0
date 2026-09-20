"""Sample title-only hydration matches for a non-automated spot-check of the same-title risk (E1-d).

Interpreter: ``evaluation/.venv/Scripts/python`` (httpx). No LLM. Importers: none (CLI;
``tests/test_spot_check_hydration.py`` imports the helpers). The evidence plus the
per-record non-automated judgement is committed as
``protocols/spot_checks/<id>.spot_check.json`` and summarised in ``protocols/<id>.json``
(``dataset_source.hydration.spot_check``).

Title-only Crossref matches (``match_method`` = ``title_exact`` / ``title_exact_year``)
cannot exclude that a comment, reprint or conference abstract with the same title was
matched instead of the article the human screeners saw. This script draws a seeded random
sample of such records from ``data/<dataset>.jsonl``, fetches the matched DOI's Crossref
record (``https://api.crossref.org/works/<doi>``: container title, year, type, ISSN) and
writes ``protocols/spot_checks/<dataset>.spot_check.json`` (or ``--out``) with, per record,
the SYNERGY title / year / abstract head next to the Crossref title / venue / year / type,
so the authors can judge venue plausibility and year agreement record by record.
``synergy_year`` is null for ``title_exact`` matches: that method is used only when the
export carries no year, so the record's ``year`` is the hydrated Crossref year and no year
agreement can be checked (``export_year``).

Judgements are merged in with ``--judgements <json>`` (``{"records": {"<record_id>":
{"judgement": "correct"|"incorrect"|"unsure", "note": "..."}}}``) together with
``--judged-by`` and ``--judged-on`` (ISO date); every sampled record then carries
``judgement`` (null when not judged), ``judgement_note``, ``judged_by`` and ``judged_on``,
and the file reports ``n_correct`` / ``n_incorrect`` / ``n_unsure`` / ``n_unjudged``.
``--out`` writes the file elsewhere (the committed copy lives under ``protocols/spot_checks/``).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import now_iso, read_json, read_jsonl, resolve_path_args, write_json  # noqa: E402

DATA_DIR = HERE / "data"
SPOT_CHECK_DIR = HERE / "protocols" / "spot_checks"
CROSSREF_WORKS = "https://api.crossref.org/works"
JUDGEMENTS = ("correct", "incorrect", "unsure")


def apply_judgements(
    records: list[dict], judgements: dict, *, judged_by: str | None, judged_on: str | None
) -> dict[str, int]:
    """Attach ``judgement`` / ``judgement_note`` / ``judged_by`` / ``judged_on`` to every
    record (``None`` when the record has no entry) and return the totals."""
    totals = {"n_correct": 0, "n_incorrect": 0, "n_unsure": 0, "n_unjudged": 0}
    for rec in records:
        entry = judgements.get(str(rec["record_id"])) or {}
        verdict = entry.get("judgement")
        if verdict is not None and verdict not in JUDGEMENTS:
            raise ValueError(
                f"record {rec['record_id']}: judgement {verdict!r} not in {JUDGEMENTS}"
            )
        rec["judgement"] = verdict
        rec["judgement_note"] = entry.get("note")
        rec["judged_by"] = judged_by if verdict else None
        rec["judged_on"] = judged_on if verdict else None
        totals[f"n_{verdict}" if verdict else "n_unjudged"] += 1
    return totals


def export_year(rec: dict) -> int | None:
    """The export's year, or None when the record was matched by ``title_exact`` (no export
    year existed to check, so ``year`` is the Crossref year attached by hydration)."""
    if rec.get("match_method") == "title_exact":
        return None
    return rec.get("year")


def sample_title_matches(records, n: int, seed: int) -> list[dict]:
    pool = [r for r in records if str(r.get("match_method") or "").startswith("title_exact")]
    rng = random.Random(f"spot-check:{seed}")
    return rng.sample(pool, min(n, len(pool)))


def crossref_record(client: httpx.Client, doi: str) -> dict:
    resp = client.get(f"{CROSSREF_WORKS}/{doi}", timeout=30.0)
    resp.raise_for_status()
    msg = resp.json().get("message") or {}
    parts = ((msg.get("issued") or {}).get("date-parts") or [[None]])[0]
    return {
        "title": (msg.get("title") or [""])[0],
        "container_title": (msg.get("container-title") or [""])[0],
        "year": parts[0] if parts else None,
        "type": msg.get("type"),
        "issn": msg.get("ISSN") or [],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--judgements", type=Path, default=None,
                    help="JSON {'records': {record_id: {judgement, note}}} to merge in")
    ap.add_argument("--judged-by", default=None, help="who made the judgements")
    ap.add_argument("--judged-on", default=None, help="ISO date of the judgements")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file (default protocols/spot_checks/<dataset>.spot_check.json)")
    args = resolve_path_args(ap.parse_args(argv))
    records = read_jsonl(args.data_dir / f"{args.dataset}.jsonl")
    sample = sample_title_matches(records, args.n, args.seed)
    out = []
    with httpx.Client(headers={"User-Agent": "ScholarRAG-evaluation spot check"}) as client:
        for rec in sample:
            try:
                cr = crossref_record(client, rec["doi"])
            except (httpx.HTTPError, ValueError) as exc:
                cr = {"error": f"{type(exc).__name__}: {exc}"}
            out.append({
                "record_id": rec["record_id"],
                "synergy_title": rec.get("title"),
                "synergy_year": export_year(rec),
                "synergy_abstract_head": (rec.get("abstract") or "")[:200],
                "match_method": rec.get("match_method"),
                "doi": rec.get("doi"),
                "issn": rec.get("issn"),
                "crossref": cr,
            })
            print(f"{rec['record_id']}: {rec.get('title')!r}\n    -> {cr}")
    judgements = {}
    if args.judgements:
        judgements = (read_json(args.judgements, {}) or {}).get("records", {})
    totals = apply_judgements(out, judgements, judged_by=args.judged_by, judged_on=args.judged_on)
    path = args.out or SPOT_CHECK_DIR / f"{args.dataset}.spot_check.json"
    write_json(path, {
        "dataset": args.dataset, "generated": now_iso(), "seed": args.seed,
        "n_pool": sum(1 for r in records
                      if str(r.get("match_method") or "").startswith("title_exact")),
        "n": len(out), "judged_by": args.judged_by, "judged_on": args.judged_on, **totals,
        "judgement_values": list(JUDGEMENTS), "records": out,
    })
    print(f"wrote {path} ({totals})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

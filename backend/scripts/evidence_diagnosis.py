"""Offline diagnosis harness for the deep-analysis evidence step.

Runs the deep-analysis agent once per seed paper, straight from stored full-text
chunks, with no API server, no database writes and no background job, and records
every proposed evidence item plus the call's usage. A second subcommand classifies
each proposed item by why the acceptance rules in
``app.services.deep_analysis.validate_evidence_items`` kept or rejected it, and a
third turns the collected runs into the offline test fixture under
``backend/tests/fixtures/evidence``.

Usage, from the repository root (so ``.env`` is picked up by ``app.config``)::

    PYTHONPATH=backend python backend/scripts/evidence_diagnosis.py run \
        --seeds <seeds.json> --out <dir> --label baseline
    PYTHONPATH=backend python backend/scripts/evidence_diagnosis.py classify \
        --seeds <seeds.json> --runs <dir>

``seeds.json`` is a list of ``{"paper_id", "title", "year", "authors", "chunks"}``
records, dumped from a project's own ``papers`` rows. The API key is read by
``app.config.settings`` from ``.env`` and is never printed or written out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.deep_analysis import (  # noqa: E402
    MIN_EVIDENCE_WORDS,
    normalise_ws,
    validate_evidence_items,
)


def load_seeds(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_authors_field(authors: object) -> str:
    from app.services.deep_analysis import format_authors

    return format_authors(authors)


async def _run_one(seed: dict, out_dir: Path, semaphore: asyncio.Semaphore) -> dict:
    from app.agents.deep_analysis_agent import analyze_paper_text_with_provenance

    paper_id = seed["paper_id"]
    record: dict = {"paper_id": paper_id, "title": seed.get("title", "")}
    async with semaphore:
        try:
            result = await analyze_paper_text_with_provenance(
                seed.get("title", ""),
                format_authors_field(seed.get("authors")),
                seed.get("year"),
                seed.get("chunks") or [],
            )
            dumped = result.output.model_dump()
            record["ok"] = True
            record["evidence"] = dumped.get("evidence") or []
            record["provenance"] = json.loads(result.provenance.model_dump_json())
        except Exception as exc:  # noqa: BLE001 - the failure mode is the datum
            record["ok"] = False
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)[:400]
            record["evidence"] = []
    (out_dir / f"{paper_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    status = "ok" if record["ok"] else "FAILED " + str(record.get("error_type"))
    print("  {} {:3d} items  {}".format(paper_id[:8], len(record["evidence"]), status), flush=True)
    return record


async def _run_all(seeds: list[dict], out_dir: Path, concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    return list(await asyncio.gather(*(_run_one(s, out_dir, semaphore) for s in seeds)))


def cmd_run(args: argparse.Namespace) -> int:
    seeds = load_seeds(Path(args.seeds))
    if args.limit:
        seeds = seeds[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Running deep analysis on {} papers, label={}".format(len(seeds), args.label), flush=True)
    records = asyncio.run(_run_all(seeds, out_dir, args.concurrency))
    in_tokens = sum((r.get("provenance") or {}).get("input_tokens") or 0 for r in records)
    out_tokens = sum((r.get("provenance") or {}).get("output_tokens") or 0 for r in records)
    failures = [r for r in records if not r.get("ok")]
    summary = {
        "label": args.label,
        "papers": len(records),
        "failed_calls": len(failures),
        "failed_types": sorted({r.get("error_type", "") for r in failures}),
        "proposed_items": sum(len(r.get("evidence") or []) for r in records),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
    }
    (out_dir / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


# --- classification -------------------------------------------------------------


def _chunk_texts(seed: dict) -> list[str]:
    return [
        (c.get("text", "") if isinstance(c, dict) else "") for c in (seed.get("chunks") or [])
    ]


BASELINE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def baseline_accepts(item: dict, chunk_texts: list[str]) -> bool:
    """The prior, narrower acceptance rules, reimplemented here so
    the before column of the report table can be recomputed at any time from the same
    stored runs: the named ``chunk_index`` had to be in range, the quote had to be at
    least ``MIN_EVIDENCE_WORDS`` words after whitespace normalisation, and it had to
    equal the concatenation of one or more consecutive sentences of that one chunk, split
    on whitespace following a full stop, question mark or exclamation mark.
    """
    quote_norm = normalise_ws(item.get("quote") if isinstance(item, dict) else None)
    chunk_index = item.get("chunk_index") if isinstance(item, dict) else None
    if len(quote_norm.split()) < MIN_EVIDENCE_WORDS:
        return False
    if not isinstance(chunk_index, int) or not 0 <= chunk_index < len(chunk_texts):
        return False
    sentences = [
        s
        for s in BASELINE_SENTENCE_SPLIT_RE.split(normalise_ws(chunk_texts[chunk_index]))
        if s
    ]
    for first in range(len(sentences)):
        joined = sentences[first]
        if joined == quote_norm:
            return True
        for nxt in range(first + 1, len(sentences)):
            if len(joined) >= len(quote_norm):
                break
            joined = "%s %s" % (joined, sentences[nxt])
            if joined == quote_norm:
                return True
    return False


def classify_item(item: dict, chunk_texts: list[str]) -> dict:
    """One proposed item's outcome under both rule sets, and the cause that separates
    them.

    ``cause`` names the single change that decides the item:

    ``too_short``
        Under ``MIN_EVIDENCE_WORDS`` words; rejected by both rule sets.
    ``paraphrase``
        The quote's letters and digits appear nowhere in the paper, so it is the model's
        own wording of something the paper says. No acceptance rule can recover it.
    ``chunk_mismatch``
        A verbatim whole-sentence span of the paper, in a chunk other than the one the
        model named (or the named chunk does not exist). Recovered by searching every
        chunk.
    ``typography``
        A verbatim whole-sentence span of the named chunk once PDF-extraction typography
        is folded on both sides, and not before. Recovered by ``fold_for_match``.
    ``sentence_boundary``
        A verbatim whole-sentence span of the named chunk whose start the old splitter
        could not see, because the extracted text glues a heading, a table row or a
        line-number gutter onto the sentence in front of it. Recovered by the tolerant
        boundary rule.
    ``fragment``
        Present in the paper as a substring, but not as a span that both starts where a
        sentence starts and ends where a sentence ends. Rejected by both rule sets: this
        is the case the whole-sentence rule exists for.
    """
    from app.services import deep_analysis as da

    chunks = [{"text": text} for text in chunk_texts]
    quote = item.get("quote") if isinstance(item, dict) else None
    named_index = item.get("chunk_index") if isinstance(item, dict) else None
    before = baseline_accepts(item, chunk_texts)
    accepted_rows, _ = validate_evidence_items([item], chunks)
    after = bool(accepted_rows)
    found_chunk = accepted_rows[0]["chunk_index"] if accepted_rows else None

    quote_ws = normalise_ws(quote)
    named_ok = isinstance(named_index, int) and 0 <= named_index < len(chunk_texts)
    if len(quote_ws.split()) < MIN_EVIDENCE_WORDS:
        cause = "too_short"
    elif after and before:
        cause = "accepted_by_both"
    elif after and (not named_ok or found_chunk != named_index):
        cause = "chunk_mismatch"
    elif after and quote_ws not in normalise_ws(chunk_texts[named_index]):
        cause = "typography"
    elif after:
        cause = "sentence_boundary"
    else:
        folded_quote = da.fold_for_match(quote)
        anywhere = any(folded_quote in da.fold_for_match(text) for text in chunk_texts)
        cause = "fragment" if anywhere else "paraphrase"
    return {
        "before": "accepted" if before else "rejected",
        "after": "accepted" if after else "rejected",
        "cause": cause,
        "found_chunk": found_chunk,
    }


def cmd_classify(args: argparse.Namespace) -> int:
    seeds = {s["paper_id"]: s for s in load_seeds(Path(args.seeds))}
    runs_dir = Path(args.runs)
    totals: dict[str, int] = {}
    per_paper = []
    for path in sorted(runs_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        seed = seeds.get(record["paper_id"])
        if seed is None:
            continue
        chunk_texts = _chunk_texts(seed)
        rows = [classify_item(i, chunk_texts) for i in record.get("evidence") or []]
        accepted_rows, rejected = validate_evidence_items(
            record.get("evidence") or [], seed.get("chunks") or []
        )
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["cause"]] = counts.get(row["cause"], 0) + 1
            totals[row["cause"]] = totals.get(row["cause"], 0) + 1
        per_paper.append({
            "paper_id": record["paper_id"],
            "title": (seed.get("title") or "")[:48],
            "ok": record.get("ok"),
            "proposed": len(rows),
            "accepted_before": sum(1 for r in rows if r["before"] == "accepted"),
            "accepted": len(accepted_rows),
            "rejected": rejected,
            "causes": counts,
        })
    proposed = sum(p["proposed"] for p in per_paper)
    accepted = sum(p["accepted"] for p in per_paper)
    accepted_before = sum(p["accepted_before"] for p in per_paper)
    rejected = sum(p["rejected"] for p in per_paper)
    total = accepted + rejected
    summary = {
        "papers": len(per_paper),
        "papers_with_items": sum(1 for p in per_paper if p["proposed"]),
        "proposed": proposed,
        "accepted_before": accepted_before,
        "accepted": accepted,
        "rejected": rejected,
        "verbatim_rate_before": round(accepted_before / proposed, 4) if proposed else 1.0,
        "verbatim_rate": round(accepted / total, 4) if total else 1.0,
        "min_accepted_per_paper_with_items": min(
            [p["accepted"] for p in per_paper if p["proposed"]] or [0]
        ),
        "causes": dict(sorted(totals.items(), key=lambda kv: -kv[1])),
        "per_paper": per_paper,
    }
    out = json.dumps(summary, ensure_ascii=False, indent=1)
    print(out)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
    return 0


# --- fixture ---------------------------------------------------------------------------

#: Marks where text was left out of a fixture chunk. Ends in a full stop so it is a
#: sentence boundary of its own: no quote can span the gap, and the lead-in a boundary
#: check reads for the first window of a chunk is the same one the full chunk gives.
GAP_MARKER = "[TEXT OMITTED FROM FIXTURE]."

#: How much raw text either side of a located quote a fixture window keeps, before being
#: widened to whole lines.
WINDOW_CHARS = 400

#: How many whole lines above and below the window are kept, so the boundary checks read
#: the same physical lines they read in the full chunk.
WINDOW_LINES_ABOVE = 3
WINDOW_LINES_BELOW = 2


def _widen_to_lines(text: str, start: int, end: int) -> tuple[int, int]:
    left = start
    for _ in range(WINDOW_LINES_ABOVE + 1):
        left = text.rfind("\n", 0, left)
        if left == -1:
            left = 0
            break
    else:
        left += 1
    right = end
    for _ in range(WINDOW_LINES_BELOW + 1):
        found = text.find("\n", right)
        if found == -1:
            right = len(text)
            break
        right = found + 1
    return left, min(right, len(text))


def _quote_windows(text: str, quote: str, limit: int = 3) -> list[tuple[int, int]]:
    """Every place *quote* folds onto *text*, as a raw range widened to whole lines."""
    from app.services.deep_analysis import fold_for_match, fold_with_offsets

    folded_quote = fold_for_match(quote)
    if not folded_quote:
        return []
    folded_text, offsets = fold_with_offsets(text)
    windows: list[tuple[int, int]] = []
    position = folded_text.find(folded_quote)
    while position != -1 and len(windows) < limit:
        raw_start = offsets[position]
        raw_end = offsets[position + len(folded_quote) - 1] + 1
        windows.append(
            _widen_to_lines(
                text, max(0, raw_start - WINDOW_CHARS), min(len(text), raw_end + WINDOW_CHARS)
            )
        )
        position = folded_text.find(folded_quote, position + 1)
    return windows


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def build_fixture_chunks(chunks: list, items: list) -> list[dict]:
    """*chunks*, reduced to the windows around every place one of *items*' quotes is
    found, joined by ``GAP_MARKER``.

    The chunk count, their order and their ``section`` labels are kept, so a quote is
    still found in the same chunk index it is found in in the full text. A quote present
    nowhere contributes no window and so is still rejected, for the same reason, on the
    fixture. ``cmd_build_fixture`` checks that acceptance on the reduced chunks matches
    acceptance on the full ones, item for item, before writing anything.
    """
    reduced: list[dict] = []
    for chunk in chunks:
        text = chunk.get("text", "") if isinstance(chunk, dict) else ""
        windows: list[tuple[int, int]] = []
        for item in items:
            windows.extend(_quote_windows(text, (item or {}).get("quote") or ""))
        parts = [text[start:end].strip() for start, end in _merge(windows)]
        joined = ("\n" + GAP_MARKER + "\n").join(parts)
        reduced.append({
            "section": chunk.get("section") if isinstance(chunk, dict) else None,
            "text": (GAP_MARKER + "\n" + joined + "\n" + GAP_MARKER) if joined else GAP_MARKER,
        })
    return reduced


def cmd_build_fixture(args: argparse.Namespace) -> int:
    seeds = {s["paper_id"]: s for s in load_seeds(Path(args.seeds))}
    runs_dir = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from app.services.deep_analysis import analysis_prompt_version

    papers = []
    total_accepted = 0
    total_proposed = 0
    for path in sorted(runs_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        seed = seeds[record["paper_id"]]
        items = record.get("evidence") or []
        full_accepted, full_rejected = validate_evidence_items(items, seed.get("chunks") or [])
        reduced = build_fixture_chunks(seed.get("chunks") or [], items)
        reduced_accepted, reduced_rejected = validate_evidence_items(items, reduced)
        if [dict(row) for row in reduced_accepted] != [dict(row) for row in full_accepted]:
            print("MISMATCH for %s: the reduced chunks do not accept what the full ones do"
                  % record["paper_id"])
            return 1
        (out_dir / path.name).write_text(
            json.dumps({
                "paper_id": record["paper_id"],
                "title": seed.get("title", ""),
                "year": seed.get("year"),
                "chunks": reduced,
                "proposed_items": items,
                "accepted_at_capture": len(full_accepted),
                "rejected_at_capture": full_rejected,
            }, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        papers.append({
            "paper_id": record["paper_id"],
            "title": seed.get("title", ""),
            "proposed": len(items),
            "accepted_at_capture": len(full_accepted),
            "rejected_at_capture": full_rejected,
            "chunks": len(reduced),
        })
        total_accepted += len(full_accepted)
        total_proposed += len(items)
    manifest = {
        "label": args.label,
        "papers": len(papers),
        "proposed": total_proposed,
        "accepted_at_capture": total_accepted,
        "verbatim_rate_at_capture": (
            round(total_accepted / total_proposed, 4) if total_proposed else 1.0
        ),
        "analysis_prompt_version": analysis_prompt_version(),
        "per_paper": papers,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=1)[:1200])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the analysis agent once per seed paper")
    p_run.add_argument("--seeds", required=True)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--label", default="run")
    p_run.add_argument("--limit", type=int, default=0)
    p_run.add_argument("--concurrency", type=int, default=4)
    p_run.set_defaults(func=cmd_run)

    p_cls = sub.add_parser("classify", help="Classify every proposed item's outcome")
    p_cls.add_argument("--seeds", required=True)
    p_cls.add_argument("--runs", required=True)
    p_cls.add_argument("--out", default="")
    p_cls.set_defaults(func=cmd_classify)

    p_fix = sub.add_parser("build-fixture", help="Write the offline acceptance fixture")
    p_fix.add_argument("--seeds", required=True)
    p_fix.add_argument("--runs", required=True)
    p_fix.add_argument("--out", required=True)
    p_fix.add_argument("--label", default="")
    p_fix.set_defaults(func=cmd_build_fixture)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

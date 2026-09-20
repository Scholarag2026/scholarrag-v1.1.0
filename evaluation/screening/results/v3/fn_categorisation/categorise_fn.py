"""Apply the FN taxonomy (results/fn_taxonomy.md) as ordered cue-word rules to every row of
the three false-negative categorisation CSVs, filling the ``category`` column only.

Interpreter: ``evaluation/.venv/Scripts/python`` or plain ``python`` (stdlib only: csv, re,
json, random, pathlib, argparse).

For each row this matches the ``reason`` text (case-insensitive) against the five
categories' cue-word lists in the taxonomy's precedence order -- FN-ABS, FN-POP,
FN-DESIGN, FN-DEF, FN-OTHER -- and takes the first category with a hit, mirroring
``fn_taxonomy.md``'s own "automated cue-word pass" (its indicative-distribution table).
A match is flagged ``confidence="low"`` when it hit only a generic, ambiguous cue (see
``WEAK_CUES`` below) or when no rule matched at all (fallback category left blank).

Overrides (assembled by a Claude Code agent, not the authors, by reading every
low-confidence / no-match row and, per the spec, a random 10% seed=20260905 sample of the
high-confidence rule matches) are supplied via
``--overrides <json>``: ``{"<dataset>|<run>|<record_id>": "<CODE>"}``. Overrides always win.

Modes:
    --report            print the per-dataset / per-category tally and list every row that
                         is low-confidence or unmatched (no file is written). Default.
    --dump-review PATH  also write the full row-by-row decision (all datasets) to a JSON
                         file at PATH, for the non-automated review pass and the audit sample.
    --write             write the ``category`` column back into each ``results/
                         <dataset>_fn_categories.csv``: rule category (or override, which
                         always wins) for every row; every other cell is copied verbatim
                         from the existing file. Refuses to run while any row would be left
                         with an empty category (all rows must be resolved by rule or
                         override first).

This script never reads or writes anything under ``evaluation/**/data/`` or any
``*_run?.jsonl`` / ``*.meta.json`` file; it only touches the three
``results/<dataset>_fn_categories.csv`` files, and only their ``category`` column.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATASETS = ["Nagtegaal_2019", "van_de_Schoot_2017", "Smid_2020"]

# --------------------------------------------------------------------------------------
# Taxonomy rules, in precedence order (fn_taxonomy.md, "Precedence rule").
# Each entry: (category, [(compiled_regex, is_weak), ...]).
# "weak" cues are generic/ambiguous on their own and are flagged low-confidence even when
# they are the only hit, so every such row gets a non-automated read.
# --------------------------------------------------------------------------------------


def _pat(phrase: str) -> re.Pattern:
    return re.compile(re.escape(phrase), re.IGNORECASE)


def _rx(pattern: str) -> re.Pattern:
    """A raw (non-escaped) regex cue, for generalising a taxonomy cue phrase to its
    observed variants (e.g. "targets patients" also written "targeting patients" or
    "targets patient outcomes")."""
    return re.compile(pattern, re.IGNORECASE)


FN_ABS_CUES = [
    (_pat("no abstract"), False),
    (_pat("no abstract available"), False),
    (_pat("no abstract confirming"), False),
    (_pat("cannot confirm"), False),
    (_pat("insufficient evidence"), False),
    (_pat("title suggests"), False),
    (_pat("title about"), False),
    (_pat("title indicates"), False),
    (_pat("title does not indicate"), False),
    (_pat("title alone"), False),
    (_pat("truncated"), False),
]

FN_POP_CUES = [
    (_pat("targets patients"), False),
    (_pat("not healthcare professionals"), False),
    (_pat("targets service users"), False),
    (_pat("general population"), False),
    (_pat("general public"), False),
    (_pat("primary school students"), False),
    (_pat("not ptsd"), False),
    (_pat("depression/anxiety trajectories, not ptsd"), False),
    (_pat("not following an event fulfilling criterion a1"), False),
    (_rx(r"criterion a1|a1 (?:trauma )?criterion"), False),
    (_pat("not human"), False),
]

# Generalisations of "targets patients" (also "targeting patients", "targets patient
# outcomes", ...) and of the "patients ... not [professionals]" contrast that recurs in
# Nagtegaal_2019's reasons (the taxonomy's own worked example, record 1474, is exactly
# this pattern) -- Nagtegaal_2019's eligible population is healthcare professionals, so
# "patients" named as the target is reliably a wrong-population signal there. This does
# NOT hold for van_de_Schoot_2017 (its eligible population is PTSD patients themselves,
# so "patients" is usually the *correct* population and the operative ground is instead
# the analytic method, e.g. "PTSD patients ... not trajectory clustering"), so these
# cues are added only for Nagtegaal_2019 (see ``classify``). Flagged weak: every hit
# still gets a non-automated read, since "patient" can also appear as an incidental topic word
# (e.g. "patient safety", "patient characteristics") rather than as the stated
# wrong-population ground.
FN_POP_CUES_NAGTEGAAL_ONLY = [
    (_rx(r"targets?(?:ing)?\s+(?:\w+\s+){0,3}patients?\b"), True),
    (_rx(r"\bpatients?\b.{0,25}?\bnot\b"), True),
    (_rx(r"\bnot\b.{0,25}?\bpatients?\b"), True),
]

FN_DESIGN_CUES = [
    (_pat("systematic review"), False),
    (_pat("review article"), False),
    (_pat("overview article"), False),
    (_pat("cochrane review"), False),
    (_pat("narrative review"), False),
    (_pat("literature review"), False),
    (_pat("commentary"), False),
    (_pat("conceptual paper"), False),
    (_pat("theoretical discussion"), False),
    (_pat("trial protocol"), False),
    (_pat("development of a complex intervention"), False),
    (_pat("economic evaluation"), False),
    (_pat("cost-effectiveness analysis"), False),
    (_pat("not an experiment"), False),
    (_pat("no comparison of two or more interventions"), False),
    (_pat("observational"), True),
    (_pat("retrospective"), True),
    (_pat("qualitative"), True),
    (_pat("grounded theory"), False),
    (_pat("erratum"), False),
    (_pat("dissertation"), False),
    (_pat("cross-sectional"), False),
    (_pat("fewer than three waves"), False),
    (_pat("only two time points"), False),
    (_pat("no simulation study"), False),
    (_pat("cochrane protocol"), False),
    (_rx(r"no simulation"), False),
    (_rx(r"\btwo\b.{0,10}\bwaves?\b"), False),
    (_rx(r"\bwaves?\b.{0,20}\bthree\b"), False),
    (_rx(r"\bconceptual\b"), True),
    (_rx(r"\btheoretical\b"), True),
]

FN_DEF_CUES = [
    (_pat("not a nudge per taxonomy"), False),
    (_pat("decision support, not a nudge"), False),
    (_pat("cdss is not a nudge"), False),
    (_pat("not soft steering"), False),
    (_pat("feedback is not classified as a nudge"), False),
    (_pat("no evidence of nudge framing"), False),
    (_pat("unclear if nudge taxonomy applies"), False),
    (_pat("no clustering method"), False),
    (_pat("lgmm/lcga"), False),
    (_pat("not lgmm/lcga/hierarchical cluster analysis"), False),
    (_pat("hierarchical cluster analysis"), False),
    (_pat("uses sem and hierarchical regressions"), False),
    (_pat("not sem as defined"), False),
    (_pat("no explicit small-sample comparison"), False),
    (_pat("not framed as"), True),
    (_pat("not described as"), True),
    (_pat("not specifically"), True),
    # Broad, dataset-specific generalisations of the taxonomy's cue phrases, all
    # flagged weak so every hit still gets a non-automated read: "nudge" mentioned anywhere
    # (Nagtegaal_2019's protocol topic -- reached here only when no FN-ABS/FN-POP/
    # FN-DESIGN cue already matched, so this is almost always the "not a nudge per
    # taxonomy" ground stated in different words); a clustering/trajectory method
    # negated nearby (van_de_Schoot_2017's "not LGMM/LCGA/hierarchical cluster
    # analysis" cue, generalised); and "small sample(s)" mentioned at all (Smid_2020's
    # protocol topic is exactly a small-sample Bayesian vs frequentist SEM comparison).
    (_rx(r"\bnudg"), True),
    (_rx(r"\b(?:not|no|without|lacks|lacking|unclear)\b.{0,60}?traject"), True),
    (_rx(r"traject\w*.{0,60}?\b(?:not|no|without|lacks|lacking|unclear)\b"), True),
    (_rx(r"\b(?:not|no|without|lacks|lacking|unclear)\b.{0,60}?cluster"), True),
    (_rx(r"cluster\w*.{0,60}?\b(?:not|no|without|lacks|lacking|unclear)\b"), True),
    (_rx(r"small[\s-]samples?"), True),
]

FN_OTHER_CUES = [
    (_pat("unrelated to"), False),
    (_pat("a study from another field"), False),
    (_pat("(no reason recorded)"), False),
    (_pat("pharmacokinetic"), False),
    (_pat("cluster munition"), False),
    (_pat("animal behaviour"), False),
    (_pat("animal behavior"), False),
]

def _rules_for(dataset: str) -> list[tuple[str, list[tuple[re.Pattern, bool]]]]:
    pop_cues = FN_POP_CUES + (FN_POP_CUES_NAGTEGAAL_ONLY if dataset == "Nagtegaal_2019" else [])
    return [
        ("FN-ABS", FN_ABS_CUES),
        ("FN-POP", pop_cues),
        ("FN-DESIGN", FN_DESIGN_CUES),
        ("FN-DEF", FN_DEF_CUES),
        ("FN-OTHER", FN_OTHER_CUES),
    ]


def classify(reason: str, dataset: str) -> tuple[str | None, str, str | None]:
    """Return ``(category_or_None, confidence, matched_cue)`` for one reason string.

    ``category`` is ``None`` when no rule matched (caller must fall back to an override
    decision from the non-automated review pass, not a silent default). ``confidence`` is
    "high" unless the only hit(s) in
    the winning category are all flagged "weak", or no rule matched at all.
    """
    text = (reason or "").strip()
    if not text:
        return "FN-OTHER", "low", "(empty reason)"
    for category, cues in _rules_for(dataset):
        hits = [(pat.pattern, weak) for pat, weak in cues if pat.search(text)]
        if hits:
            all_weak = all(weak for _, weak in hits)
            matched = ", ".join(p for p, _ in hits)
            return category, ("low" if all_weak else "high"), matched
    return None, "low", None


def load_rows(dataset: str) -> list[dict[str, str]]:
    path = HERE / f"{dataset}_fn_categories.csv"
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def build_decisions(overrides: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    decisions: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        rows = load_rows(dataset)
        out = []
        for row in rows:
            k = f"{row['dataset']}|{row['run']}|{row['record_id']}"
            rule_cat, confidence, matched = classify(row["reason"], dataset)
            override = overrides.get(k)
            final_cat = override or rule_cat
            out.append(
                {
                    "key": k,
                    "dataset": row["dataset"],
                    "run": row["run"],
                    "record_id": row["record_id"],
                    "title": row["title"],
                    "reason": row["reason"],
                    "rule_category": rule_cat,
                    "confidence": confidence,
                    "matched_cue": matched,
                    "override_category": override,
                    "final_category": final_cat,
                    "source": "override" if override else ("rule" if rule_cat else None),
                }
            )
        decisions[dataset] = out
    return decisions


def print_report(decisions: dict[str, list[dict[str, Any]]]) -> None:
    for dataset, rows in decisions.items():
        tally: dict[str, int] = {}
        for r in rows:
            cat = r["final_category"] or "(unresolved)"
            tally[cat] = tally.get(cat, 0) + 1
        print(f"=== {dataset} ({len(rows)} rows) ===")
        for cat, n in sorted(tally.items()):
            print(f"  {cat}: {n}")
        review = [r for r in rows if r["source"] is None or r["confidence"] == "low"]
        print(f"  needs a non-automated read (low-confidence or unmatched): {len(review)}")
        for r in review:
            print(f"    [{r['run']}/{r['record_id']}] rule={r['rule_category']} "
                  f"conf={r['confidence']} cue={r['matched_cue']!r} :: {r['reason'][:140]}")
        print()


def write_csvs(decisions: dict[str, list[dict[str, Any]]]) -> None:
    unresolved = [
        r for rows in decisions.values() for r in rows if not r["final_category"]
    ]
    if unresolved:
        raise SystemExit(
            f"refusing to write: {len(unresolved)} rows have no rule match and no override "
            f"(e.g. {unresolved[0]['key']}); supply --overrides for all of them first"
        )
    for dataset, rows in decisions.items():
        path = HERE / f"{dataset}_fn_categories.csv"
        original = load_rows(dataset)
        by_key = {r["key"]: r["final_category"] for r in rows}
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["dataset", "run", "record_id", "title", "reason", "category", "notes"])
            for row in original:
                k = f"{row['dataset']}|{row['run']}|{row['record_id']}"
                writer.writerow(
                    [
                        row["dataset"], row["run"], row["record_id"], row["title"],
                        row["reason"], by_key[k], row["notes"],
                    ]
                )
        print(f"wrote {path}")


def sample_for_audit(
    decisions: dict[str, list[dict[str, Any]]], frac: float, seed: int
) -> list[dict[str, Any]]:
    """Random sample of rule-assigned (non-override) rows for the confirmation read."""
    pool = [
        r
        for rows in decisions.values()
        for r in rows
        if r["source"] == "rule" and r["confidence"] == "high"
    ]
    rnd = random.Random(seed)
    k = max(1, round(len(pool) * frac))
    return rnd.sample(pool, k)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--overrides", type=Path, default=None, help="JSON {key: category}")
    ap.add_argument("--dump-review", type=Path, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument(
        "--audit-sample", type=Path, default=None, help="write the seed=20260905 10%% sample here"
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: dict[str, str] = {}
    if args.overrides and args.overrides.exists():
        overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
    decisions = build_decisions(overrides)
    print_report(decisions)
    if args.dump_review:
        args.dump_review.write_text(
            json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.dump_review}")
    if args.audit_sample:
        sample = sample_for_audit(decisions, 0.10, 20260905)
        args.audit_sample.write_text(
            json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.audit_sample} ({len(sample)} rows)")
    if args.write:
        write_csvs(decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

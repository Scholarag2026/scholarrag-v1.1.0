"""claims/summarize: SciFact + HSS tables, figure JSON and markdown from synthetic results."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from common import append_jsonl, write_json

CLAIMS_DIR = Path(__file__).resolve().parents[1] / "claims"


def load_claims_summarize():
    try:
        from conftest import load_script_module  # provided by the screening side
    except ImportError:
        load_script_module = None
    if load_script_module is not None:
        return load_script_module("claims", "summarize")
    spec = importlib.util.spec_from_file_location("claims_summarize", CLAIMS_DIR / "summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sci_row(item_id, gold, status, quote, verbatim, casefold=None, *, error=None, det=False):
    return {
        "item_id": item_id, "claim_id": int(item_id.split(":")[0]), "doc_id": 1, "gold": gold,
        "claim": "c", "predicted_status": status, "evidence_quote": quote,
        "quote_is_verbatim": verbatim,
        "quote_is_verbatim_casefold": verbatim if casefold is None else casefold,
        "explanation": "e", "suggested_revision": None, "deterministic": det,
        "called_at": "2026-09-02T12:00:00+00:00", "latency_s": None if error else 1.5,
        "input_tokens": None if error else 100, "output_tokens": None if error else 10,
        "cache_read_tokens": None if error else 20, "usage_details": None,
        "model_reported": None if error else "deepseek-v4-flash",
        "system_fingerprint": None if error else "fp-1", "provider_response_id": "r",
        "attempts": 1, "error": error,
    }


SCI_A = [
    sci_row("1:1", "SUPPORT", "verified", "q", True),
    sci_row("2:1", "SUPPORT", "verified", "Q", False, True),
    sci_row("3:1", "SUPPORT", "needs_nuance", "q", True),
    sci_row("4:1", "CONTRADICT", "unsupported", None, None),
    sci_row("5:1", "CONTRADICT", "verified", None, None),
    sci_row("6:1", "NOT_ENOUGH_INFO", "unsupported", None, None),
    sci_row("7:1", "NOT_ENOUGH_INFO", "needs_nuance", None, None),
    sci_row("8:1", "NOT_ENOUGH_INFO", "error", None, None, error="RuntimeError: x"),
]
SCI_B = [dict(r) for r in SCI_A]
SCI_B[4]["predicted_status"] = "unsupported"  # B corrects one item


def meta(name, run, rows):
    ok = [r for r in rows if not r["error"]]
    return {
        "name": name, "run": run, "dry_run": False, "model_configured": "deepseek-chat",
        "model_reported": ["deepseek-v4-flash"], "system_fingerprints": ["fp-1"],
        "temperature": 0.0, "prompt_version": "sha256:abc", "n_rows": len(rows),
        "n_errors": len(rows) - len(ok), "n_deterministic": sum(r["deterministic"] for r in rows),
        "total_input_tokens": 100 * len(ok), "total_output_tokens": 10 * len(ok),
        "total_cache_read_tokens": 20 * len(ok), "total_cost": 0.001234,
        "cost_basis": "list price, tier by call time, cache-hit tokens at cache-hit rate",
        "total_cost_flat": None, "price": {"model": "deepseek-chat"},
        "latency_s": {"n": len(ok), "median": 1.5, "p90": 1.5, "mean": 1.5, "total": 1.5},
        "finished": "2026-09-02T12:30:00+00:00", "started": "2026-09-02T12:00:00+00:00",
    }


def hss_row(rule, status, expected, quote=None, verbatim=None, *, alteration=None):
    return {
        "item_id": f"hss-{rule}-01", "rule": rule, "expected": expected, "claim": "c",
        "alteration": alteration,
        "predicted_status": status, "correct": status in expected, "evidence_quote": quote,
        "quote_is_verbatim": verbatim, "quote_is_verbatim_casefold": verbatim,
        "deterministic": rule == "no_full_text", "called_at": "2026-09-02T12:00:00+00:00",
        "latency_s": 0.0 if rule == "no_full_text" else 2.0, "input_tokens": 50,
        "output_tokens": 5, "cache_read_tokens": 0, "model_reported": "deepseek-v4-flash",
        "system_fingerprint": "fp-1", "attempts": 1, "error": None,
    }


HSS_A = [
    hss_row("verbatim", "verified", ["verified"], "q", True),
    hss_row("paraphrase", "needs_nuance", ["verified", "needs_nuance"]),
    hss_row("altered", "verified", ["unsupported", "needs_nuance"], "q", False),
    hss_row("wrong_paper", "unsupported", ["unsupported"]),
    hss_row("no_full_text", "no_full_text", ["no_full_text"]),
]
HSS_B = [dict(r) for r in HSS_A]
HSS_B[2]["predicted_status"] = "unsupported"
HSS_B[2]["correct"] = True


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    append_jsonl(d / "scifact_runA.jsonl", SCI_A)
    append_jsonl(d / "scifact_runB.jsonl", SCI_B)
    write_json(d / "scifact_runA.meta.json", meta("scifact", "A", SCI_A))
    write_json(d / "scifact_runB.meta.json", meta("scifact", "B", SCI_B))
    write_json(
        d / "scifact_baseline.json",
        {
            "name": "scifact_baseline_lexical", "threshold": 0.5, "n": 8, "accuracy": 0.5,
            "macro_f1": 0.4,
            "per_class": {
                "verified": {"precision": 0.5, "recall": 0.25, "f1": 0.333, "support": 3},
                "unsupported": {"precision": 0.6, "recall": 0.8, "f1": 0.6, "support": 5},
            },
        },
    )
    append_jsonl(d / "hss_runA.jsonl", HSS_A)
    append_jsonl(d / "hss_runB.jsonl", HSS_B)
    write_json(d / "hss_runA.meta.json", meta("hss", "A", HSS_A))
    write_json(d / "hss_runB.meta.json", meta("hss", "B", HSS_B))
    return d


def test_summarize_scifact_tables(results_dir: Path):
    sm = load_claims_summarize()
    with pytest.raises(SystemExit) as exc:
        sm.main(["--help"])
    assert exc.value.code == 0
    assert sm.main(["--results-dir", str(results_dir)]) == 0
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    a = summary["scifact"]["runs"]["A"]
    assert a["n"] == 8 and a["n_errors"] == 1
    assert a["strict"]["verified"]["precision"] == pytest.approx(2 / 3)
    assert a["strict"]["verified"]["recall"] == pytest.approx(2 / 3)
    assert a["strict"]["unsupported"]["precision"] == 1.0
    assert a["strict"]["unsupported"]["recall"] == pytest.approx(0.4)
    assert a["accuracy"] == pytest.approx(4 / 8)
    assert a["lenient"]["precision"] == pytest.approx(2 / 3)
    assert a["lenient"]["recall"] == pytest.approx(2 / 3)
    assert a["lenient"]["accuracy"] == pytest.approx(5 / 7)  # the error row is not a TN
    assert a["lenient"]["n"] == 7 and a["lenient"]["n_errors"] == 1
    assert a["confusion"]["SUPPORT"]["needs_nuance"] == 1
    assert a["confusion"]["NOT_ENOUGH_INFO"]["error"] == 1
    assert set(a["confusion"]["SUPPORT"]) == {
        "verified", "needs_nuance", "unsupported", "no_full_text", "error"
    }
    assert a["needs_nuance_by_gold"] == {"SUPPORT": 1, "CONTRADICT": 0, "NOT_ENOUGH_INFO": 1}
    qf = a["quote_fidelity"]
    assert qf["n_verified"] == 3 and qf["verbatim_share"] == pytest.approx(1 / 3)
    assert qf["casefold_share"] == pytest.approx(2 / 3)
    assert qf["null_quote_share"] == pytest.approx(1 / 3)
    agree = summary["scifact"]["agreement_AB"]
    assert agree["n"] == 8 and agree["n_compared"] == 7  # the error pair is excluded
    assert agree["n_excluded_error"] == 1 and agree["percent_status"] == pytest.approx(6 / 7)
    assert agree["percent_binary"] == pytest.approx(6 / 7)
    assert 0 < agree["kappa_status"] < 1 and 0 < agree["kappa_binary"] < 1
    b = summary["scifact"]["runs"]["B"]
    assert b["strict"]["verified"]["precision"] == 1.0
    assert summary["scifact"]["baseline"]["per_class"]["verified"]["recall"] == 0.25
    prov = a["provenance"]
    assert prov["model_reported"] == ["deepseek-v4-flash"] and prov["temperature"] == 0.0
    assert prov["prompt_version"] == "sha256:abc" and prov["tokens"]["cache_read"] == 140
    assert prov["total_cost"] == 0.001234 and prov["latency"]["median"] == 1.5


def test_summarize_hss_figure_and_markdown(results_dir: Path):
    sm = load_claims_summarize()
    sm.main(["--results-dir", str(results_dir)])
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    h = summary["hss"]["runs"]["A"]
    assert h["n"] == 5 and h["accuracy"] == pytest.approx(0.8)
    assert h["by_rule"]["altered"]["correct"] == 0 and h["by_rule"]["altered"]["n"] == 1
    assert h["by_rule"]["altered"]["status_counts"] == {"verified": 1}
    assert h["by_rule"]["verbatim"]["quote_fidelity"]["verbatim_share"] == 1.0
    assert h["by_rule"]["altered"]["quote_fidelity"]["verbatim_share"] == 0.0
    assert summary["hss"]["runs"]["B"]["accuracy"] == 1.0
    agree = summary["hss"]["agreement_AB"]
    assert agree["n"] == 5 and agree["n_compared"] == 4  # the no_full_text pair is excluded
    assert agree["percent_status"] == pytest.approx(0.75)

    fig = json.loads((results_dir / "figure4_claims.json").read_text(encoding="utf-8"))
    rows = fig["rows"]
    assert {r["set"] for r in rows} == {"scifact", "hss"}
    assert {r["series"] for r in rows if r["set"] == "scifact"} == {
        "LLM run A", "LLM run B", "lexical baseline"
    }
    metrics = {r["metric"] for r in rows}
    assert metrics >= {
        "verified_precision", "verified_recall", "verified_f1", "accuracy", "quote_fidelity",
        "kappa_AB",
    }
    base = {r["metric"]: r["value"] for r in rows
            if r["set"] == "scifact" and r["series"] == "lexical baseline"}
    assert base["verified_recall"] == 0.25 and base["accuracy"] == 0.5
    run_a = {r["metric"]: r["value"] for r in rows
             if r["set"] == "scifact" and r["series"] == "LLM run A"}
    assert run_a["verified_precision"] == pytest.approx(2 / 3)
    assert run_a["kappa_AB"] == summary["scifact"]["agreement_AB"]["kappa_status"]

    md = (results_dir / "summary.md").read_text(encoding="utf-8")
    for heading in ("Table E2-a SciFact", "E2-b HSS set", "E2-c provenance/cost/latency"):
        assert heading in md
    assert "labels are by construction" in md
    assert "0.667" in md


def test_summarize_tolerates_missing_runs(tmp_path: Path):
    """A directory carrying at least one dataset's run files (here scifact only) still
    summarises cleanly, with the missing dataset (hss) reported as empty rather than
    raising."""
    sm = load_claims_summarize()
    d = tmp_path / "partial"
    d.mkdir()
    append_jsonl(d / "scifact_runA.jsonl", SCI_A)
    append_jsonl(d / "scifact_runB.jsonl", SCI_B)
    write_json(d / "scifact_runA.meta.json", meta("scifact", "A", SCI_A))
    write_json(d / "scifact_runB.meta.json", meta("scifact", "B", SCI_B))
    assert sm.main(["--results-dir", str(d)]) == 0
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert summary["scifact"]["runs"] and summary["hss"]["runs"] == {}


def test_summarize_exits_non_zero_on_a_results_dir_with_no_run_files(tmp_path: Path, capsys):
    """A results directory holding no run files at all (wrong or stale --results-dir)
    must fail loudly instead of writing a summary every block of which is empty."""
    sm = load_claims_summarize()
    d = tmp_path / "empty"
    d.mkdir()
    assert sm.main(["--results-dir", str(d)]) == 1
    assert not (d / "summary.json").exists()
    assert "no run files found" in capsys.readouterr().out


# ---------------------------------------------------------------- edge cases


def test_markdown_survives_runs_sharing_no_items_and_empty_hss_run(tmp_path: Path):
    sm = load_claims_summarize()
    d = tmp_path / "results"
    a = [dict(r) for r in SCI_A[:2]]
    b = [dict(r, item_id=f"9{i}:1") for i, r in enumerate(SCI_A[2:4])]  # disjoint ids
    append_jsonl(d / "scifact_runA.jsonl", a)
    append_jsonl(d / "scifact_runB.jsonl", b)
    write_json(d / "scifact_runA.meta.json", meta("scifact", "A", a))
    write_json(d / "scifact_runB.meta.json", meta("scifact", "B", b))
    (d / "hss_runA.jsonl").write_text("", encoding="utf-8")  # zero rows
    write_json(d / "hss_runA.meta.json", meta("hss", "A", []))
    assert sm.main(["--results-dir", str(d)]) == 0
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert summary["scifact"]["agreement_AB"]["n"] == 0
    assert summary["scifact"]["agreement_AB"]["kappa_status"] is None
    assert summary["hss"]["runs"]["A"]["accuracy"] is None
    md = (d / "summary.md").read_text(encoding="utf-8")
    assert ("A/B agreement (n=0, compared 0; excluded 0 deterministic, 0 error): "
            "status - (kappa -)") in md
    assert "overall accuracy - (0/0)" in md


def test_hss_summary_reports_needs_nuance_and_lexical_baseline(results_dir: Path):
    sm = load_claims_summarize()
    write_json(
        results_dir / "hss_baseline.json",
        {
            "name": "hss_baseline_lexical", "threshold": 0.5, "n": 5, "accuracy": 0.6,
            "by_rule": {
                "verbatim": {"n": 1, "correct": 1, "accuracy": 1.0},
                "paraphrase": {"n": 1, "correct": 1, "accuracy": 1.0},
                "altered": {"n": 1, "correct": 0, "accuracy": 0.0},
                "wrong_paper": {"n": 1, "correct": 1, "accuracy": 1.0},
                "no_full_text": {"n": 1, "correct": 0, "accuracy": 0.0},
            },
            "verified_binary": {"n": 4, "n_excluded_deterministic": 1, "precision": 0.5,
                                "recall": 1.0, "f1": 2 / 3},
        },
    )
    assert sm.main(["--results-dir", str(results_dir)]) == 0
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    h = summary["hss"]["runs"]["A"]
    assert h["by_rule"]["paraphrase"]["needs_nuance"] == 1
    assert h["by_rule"]["verbatim"]["needs_nuance"] == 0
    assert h["n_needs_nuance"] == 1
    assert summary["hss"]["baseline"]["accuracy"] == 0.6
    md = (results_dir / "summary.md").read_text(encoding="utf-8")
    assert "needs_nuance" in md and "lexical baseline" in md
    assert "| LLM run A | 5 | 4 |" in md  # same denominator as the baseline row below it
    assert "| lexical baseline | 5 | 4 |" in md  # n items, then n of the binary view
    assert "excludes the 1 no-chunk item" in md
    fig = json.loads((results_dir / "figure4_claims.json").read_text(encoding="utf-8"))
    base = {r["metric"]: r["value"] for r in fig["rows"]
            if r["set"] == "hss" and r["series"] == "lexical baseline"}
    assert base["verified_recall"] == 1.0 and base["accuracy"] == 0.6


# ---------------------------------------------------------------- more edge cases


def _pipe_counts(md: str, heading: str) -> list[int]:
    block = md.split(heading, 1)[1].split("\n\n")
    table = next(b for b in block if b.startswith("| rule"))
    return [line.count("|") for line in table.splitlines()]


def test_hss_table_rows_have_as_many_pipes_as_the_header(results_dir: Path):
    sm = load_claims_summarize()
    sm.main(["--results-dir", str(results_dir)])
    md = (results_dir / "summary.md").read_text(encoding="utf-8")
    counts = _pipe_counts(md, "## Table E2-b")
    assert len(set(counts)) == 1, counts
    assert "unsupported / needs_nuance" in md and "unsupported|needs_nuance" not in md


def test_agreement_excludes_deterministic_and_error_rows_and_flags_undefined_kappa():
    sm = load_claims_summarize()
    a = [
        hss_row("verbatim", "verified", ["verified"]),
        hss_row("paraphrase", "verified", ["verified"]),
        hss_row("no_full_text", "no_full_text", ["no_full_text"]),
        dict(hss_row("altered", "error", ["unsupported"]), error="RuntimeError: x"),
    ]
    for i, r in enumerate(a):
        r["item_id"] = f"it-{i}"
    b = [dict(r) for r in a]
    agree = sm.agreement(a, b)
    assert agree["n"] == 4 and agree["n_compared"] == 2
    assert agree["n_excluded_deterministic"] == 1 and agree["n_excluded_error"] == 1
    assert agree["percent_status"] == 1.0
    assert agree["kappa_status"] is None and agree["kappa_undefined"] is True
    b[1]["predicted_status"] = "unsupported"
    agree = sm.agreement(a, b)
    assert agree["percent_status"] == 0.5 and agree["kappa_undefined"] is False


def test_error_rows_are_not_true_negatives_in_lenient_and_binary_views():
    sm = load_claims_summarize()
    sci = [
        sci_row("1:1", "SUPPORT", "verified", "q", True),
        sci_row("2:1", "CONTRADICT", "unsupported", None, None),
        sci_row("3:1", "CONTRADICT", "error", None, None, error="RuntimeError: x"),
    ]
    s = sm.scifact_run_summary(sci, meta("scifact", "A", sci))
    assert s["lenient"]["n"] == 2 and s["lenient"]["tn"] == 1 and s["lenient"]["n_errors"] == 1
    hss = [
        hss_row("verbatim", "verified", ["verified"]),
        hss_row("wrong_paper", "unsupported", ["unsupported"]),
        dict(hss_row("altered", "error", ["unsupported"]), error="RuntimeError: y"),
    ]
    h = sm.hss_run_summary(hss, meta("hss", "A", hss))
    assert h["verified_binary"]["n"] == 2 and h["verified_binary"]["tn"] == 1
    assert h["verified_binary"]["n_errors"] == 1


def test_scifact_duplicate_cited_doc_id_is_counted_and_disclosed():
    """A dev claim that lists the same document twice in its own cited_doc_ids (claim 1245 /
    doc 7662395 in the real SciFact dev set) makes run_scifact.build_items emit the same
    item_id twice; scifact_run_summary must report the row/pair-count split and the specific
    duplicate, and _md_scifact must disclose it in prose (not silently drop or dedupe it)."""
    sm = load_claims_summarize()
    sci = [
        sci_row("1:1", "SUPPORT", "verified", "q", True),
        sci_row("1245:7662395", "CONTRADICT", "unsupported", None, None),
        sci_row("1245:7662395", "CONTRADICT", "unsupported", None, None),
    ]
    s = sm.scifact_run_summary(sci, meta("scifact", "A", sci))
    assert s["n"] == 3 and s["n_distinct_pairs"] == 2
    assert s["duplicate_pairs"] == [{"claim_id": "1245", "doc_id": "7662395", "count": 2}]

    summary = {"scifact": {"runs": {"A": s}}}
    md = "\n".join(sm._md_scifact(summary))
    assert "n = 3 rows over 2 distinct claim-document pairs" in md
    assert "claim 1245 cites document 7662395 twice" in md

    # No duplicate: the field is present but empty, and no disclosure line is written.
    clean = [sci_row("1:1", "SUPPORT", "verified", "q", True),
             sci_row("2:2", "SUPPORT", "verified", "q", True)]
    s_clean = sm.scifact_run_summary(clean, meta("scifact", "A", clean))
    assert s_clean["n"] == s_clean["n_distinct_pairs"] == 2 and s_clean["duplicate_pairs"] == []
    md_clean = "\n".join(sm._md_scifact({"scifact": {"runs": {"A": s_clean}}}))
    assert "distinct claim-document pairs" not in md_clean


def test_kappa_undefined_footnote_in_claims_markdown(tmp_path: Path):
    sm = load_claims_summarize()
    d = tmp_path / "results"
    rows = [sci_row("1:1", "SUPPORT", "verified", "q", True),
            sci_row("2:1", "SUPPORT", "verified", "q", True)]
    append_jsonl(d / "scifact_runA.jsonl", rows)
    append_jsonl(d / "scifact_runB.jsonl", rows)
    write_json(d / "scifact_runA.meta.json", meta("scifact", "A", rows))
    write_json(d / "scifact_runB.meta.json", meta("scifact", "B", rows))
    assert sm.main(["--results-dir", str(d)]) == 0
    md = (d / "summary.md").read_text(encoding="utf-8")
    assert "kappa undefined (constant runs)" in md
    assert "A/B agreement (n=2, compared 2" in md


# ---------------------------------------------------------------- annotation scoring


def test_verified_binary_excludes_deterministic_rows():
    sm = load_claims_summarize()
    hss = [
        hss_row("verbatim", "verified", ["verified"]),
        hss_row("wrong_paper", "unsupported", ["unsupported"]),
        hss_row("no_full_text", "no_full_text", ["no_full_text"]),
        dict(hss_row("altered", "error", ["unsupported"]), error="RuntimeError: y"),
    ]
    h = sm.hss_run_summary(hss, meta("hss", "A", hss))
    vb = h["verified_binary"]
    assert vb["n"] == 2 and vb["tn"] == 1  # the no_full_text item is not a true negative
    assert vb["n_errors"] == 1 and vb["n_excluded_deterministic"] == 1


def test_baseline_n_mismatch_is_flagged(results_dir: Path):
    sm = load_claims_summarize()
    base = json.loads((results_dir / "scifact_baseline.json").read_text(encoding="utf-8"))
    base["n"] = 22  # stale baseline from an earlier --limit run
    write_json(results_dir / "scifact_baseline.json", base)
    summary = sm.summarise(results_dir)
    assert summary["scifact"]["baseline_n_mismatch"] is True
    assert summary["hss"]["baseline_n_mismatch"] is False  # no HSS baseline file -> no flag
    md = sm.render_markdown(summary)
    assert "| lexical baseline (n mismatch: 22 vs 8) |" in md
    base["n"] = 8
    write_json(results_dir / "scifact_baseline.json", base)
    summary = sm.summarise(results_dir)
    assert summary["scifact"]["baseline_n_mismatch"] is False
    assert "n mismatch" not in sm.render_markdown(summary)


def _write_labels_csv(path: Path, rows: list[dict]) -> None:
    import csv

    fieldnames = ["ann_id", "item_id", "final_label", "source", "rationale", "evidence_mode",
                  "construction_category", "expected_label", "agrees_with_construction"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            full = {
                "ann_id": f"ann-{i:02d}", "source": "unanimous", "rationale": "r",
                "evidence_mode": "passage", "construction_category": row["item_id"].split("-")[1],
                "agrees_with_construction": "True",
            }
            full.update(row)
            writer.writerow(full)


LABELS_ROWS = [
    {"item_id": "hss-verbatim-01", "final_label": "verified", "expected_label": "verified"},
    {"item_id": "hss-paraphrase-01", "final_label": "needs_nuance",
     "expected_label": "verified|needs_nuance"},
    {"item_id": "hss-altered-01", "final_label": "unsupported",
     "expected_label": "unsupported|needs_nuance"},
    {"item_id": "hss-wrong_paper-01", "final_label": "unsupported",
     "expected_label": "unsupported"},
    {"item_id": "hss-no_full_text-01", "final_label": "no_full_text",
     "expected_label": "no_full_text"},
]


@pytest.fixture
def labels_csv(tmp_path: Path) -> Path:
    path = tmp_path / "hss_annotation_adjudicated_v2.csv"
    _write_labels_csv(path, LABELS_ROWS)
    return path


def test_hss_vs_annotation_scores_runs_against_adjudicated_labels(
    results_dir: Path, labels_csv: Path
):
    """run A's altered item is a mismatch (predicted verified, mapped verified, adjudicated
    unsupported); run B corrects it, so accuracy must differ between the two runs."""
    sm = load_claims_summarize()
    assert sm.main(["--results-dir", str(results_dir), "--labels", str(labels_csv)]) == 0
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    block = summary["hss_vs_annotation"]
    assert block["n_labels"] == 5
    a = block["runs"]["A"]
    assert a["n"] == 5 and a["accuracy"] == pytest.approx(0.8)
    assert a["per_label"]["unsupported"]["precision"] == pytest.approx(1.0)
    assert a["per_label"]["unsupported"]["recall"] == pytest.approx(0.5)
    assert a["per_label"]["verified"]["precision"] == pytest.approx(0.5)
    assert a["confusion"]["unsupported"]["verified"] == 1  # altered item mismatch
    assert 0 < a["kappa"] < 1
    b = block["runs"]["B"]
    assert b["n"] == 5 and b["accuracy"] == pytest.approx(1.0)
    assert b["kappa"] == pytest.approx(1.0)
    # existing blocks are untouched by --labels
    assert summary["scifact"]["runs"]["A"]["n"] == 8
    assert summary["hss"]["runs"]["A"]["n"] == 5


def test_hss_vs_annotation_reports_construction_vs_adjudicated_agreement(
    results_dir: Path, labels_csv: Path
):
    sm = load_claims_summarize()
    sm.main(["--results-dir", str(results_dir), "--labels", str(labels_csv)])
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    cva = summary["hss_vs_annotation"]["construction_vs_adjudicated"]
    assert cva == {"n": 5, "agree": 5, "accuracy": 1.0, "n_disjunctive": 2}

    # a mismatch must be visible: adjudicated label outside the construction rule's set
    rows = [dict(r) for r in LABELS_ROWS]
    rows[3]["final_label"] = "verified"  # wrong_paper item, expected_label = "unsupported" only
    _write_labels_csv(labels_csv, rows)
    sm.main(["--results-dir", str(results_dir), "--labels", str(labels_csv)])
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    cva = summary["hss_vs_annotation"]["construction_vs_adjudicated"]
    assert cva["agree"] == 4 and cva["accuracy"] == pytest.approx(0.8)


def test_hss_vs_annotation_absent_without_labels_flag(results_dir: Path):
    sm = load_claims_summarize()
    sm.main(["--results-dir", str(results_dir)])
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert "hss_vs_annotation" not in summary


def test_hss_vs_annotation_appears_in_markdown(results_dir: Path, labels_csv: Path):
    sm = load_claims_summarize()
    sm.main(["--results-dir", str(results_dir), "--labels", str(labels_csv)])
    md = (results_dir / "summary.md").read_text(encoding="utf-8")
    assert "E2-d" in md and "blind" in md.lower()
    assert "hss_annotation_adjudicated_v2.csv" in md
    assert "Construction labels vs the adjudicated labels: 5 of 5 agree" in md
    assert "2 of 5 items admit two labels" in md


def test_committed_claims_summary_has_hss_vs_annotation_for_both_runs():
    """The committed evaluation/claims/results/v6/summary.json must carry the
    hss_vs_annotation block for both runs A and B, so a reviewer reading the shipped JSON
    (not just re-running the harness) sees the independent-annotation cross-check."""
    summary = json.loads(
        (CLAIMS_DIR / "results" / "v6" / "summary.json").read_text(encoding="utf-8")
    )
    hva = summary.get("hss_vs_annotation")
    assert hva is not None, "summary.json must be regenerated with --labels <adjudicated csv>"
    assert set(hva["runs"].keys()) == {"A", "B"}
    for run in ("A", "B"):
        r = hva["runs"][run]
        assert "accuracy" in r and "accuracy_construction_tolerant" in r


def test_hss_vs_annotation_construction_tolerant_accuracy_and_dominant_miss_cell(
    tmp_path: Path, labels_csv: Path
):
    """The verifier's real disagreement pattern is predicting `needs_nuance` on an item the
    adjudication called `unsupported`; Table E2-b already scores that as correct on `altered`
    items, so E2-d must report a construction-rule-tolerant accuracy alongside the strict one
    and name the single cell it is being lenient about, both in JSON and in the markdown."""
    sm = load_claims_summarize()
    hss_a = [dict(r) for r in HSS_A]
    hss_a[2] = dict(hss_a[2], predicted_status="needs_nuance", correct=False)
    d = tmp_path / "results"
    append_jsonl(d / "hss_runA.jsonl", hss_a)
    append_jsonl(d / "hss_runB.jsonl", HSS_B)
    write_json(d / "hss_runA.meta.json", meta("hss", "A", hss_a))
    write_json(d / "hss_runB.meta.json", meta("hss", "B", HSS_B))
    assert sm.main(["--results-dir", str(d), "--labels", str(labels_csv)]) == 0
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    block = summary["hss_vs_annotation"]

    a = block["runs"]["A"]
    assert a["accuracy"] == pytest.approx(0.8)
    assert a["accuracy_construction_tolerant"] == pytest.approx(1.0)
    assert a["dominant_miss_cell"] == {"gold": "unsupported", "pred": "needs_nuance", "n": 1}
    b = block["runs"]["B"]
    assert b["accuracy"] == pytest.approx(1.0)
    assert b["accuracy_construction_tolerant"] == pytest.approx(1.0)
    assert b["dominant_miss_cell"] is None  # no misses to concentrate
    assert block["n_adjudicated_needs_nuance"] == 1  # hss-paraphrase-01's final_label

    md = (d / "summary.md").read_text(encoding="utf-8")
    assert "accuracy (strict)" in md
    assert "construction-tolerant (legacy, not reported)" in md
    assert "Table E2-b" in md
    assert "adjudicated label set contains 1" in md and "`needs_nuance` label" in md
    assert "adjudicated `unsupported` -> verifier `needs_nuance`" in md


# --------------------------------------------------------------------------------------
# real-claims set (real_run{A,B}.jsonl, real_annotation_adjudicated_v3.csv; no dev
# counterpart, no gold label, no baseline)
# --------------------------------------------------------------------------------------

def real_row(item_id: str, status: str) -> dict:
    return {
        "item_id": item_id, "rule": "real", "expected": None, "claim": "c",
        "alteration": None, "predicted_status": status, "correct": False,
        "evidence_quote": None, "quote_is_verbatim": None, "quote_is_verbatim_casefold": None,
        "deterministic": False, "called_at": "2026-09-02T12:00:00+00:00", "latency_s": 2.0,
        "input_tokens": 50, "output_tokens": 5, "cache_read_tokens": 0,
        "model_reported": "deepseek-v4-flash", "system_fingerprint": "fp-1", "attempts": 1,
        "error": None,
    }


REAL_A = [
    real_row("real-test-01", "verified"),
    real_row("real-test-02", "unsupported"),
    real_row("real-test-03", "needs_nuance"),
]
REAL_B = [dict(r) for r in REAL_A]
REAL_B[1]["predicted_status"] = "verified"  # B disagrees with A on one item


def _write_real_runs(results_dir: Path) -> None:
    append_jsonl(results_dir / "real_runA.jsonl", REAL_A)
    append_jsonl(results_dir / "real_runB.jsonl", REAL_B)
    write_json(results_dir / "real_runA.meta.json", meta("real", "A", REAL_A))
    write_json(results_dir / "real_runB.meta.json", meta("real", "B", REAL_B))


def test_real_block_present_with_agreement_ab(results_dir: Path):
    """The real-claims set gets its own top-level `real` block once `real_run{A,B}.jsonl` are
    present, with a meaningful `agreement_AB` (the A/B kappa) even though `accuracy`/`correct`/
    `by_rule` are vacuous: `expected` is null on every row by construction (no gold label;
    `real_claims_test.jsonl` carries no construction rule), so `correct` is False throughout
    and `by_rule` is empty (`"real"` is not one of `HSS_RULES`)."""
    sm = load_claims_summarize()
    _write_real_runs(results_dir)
    summary = sm.summarise(results_dir)
    real = summary["real"]
    assert set(real["runs"].keys()) == {"A", "B"}
    a = real["runs"]["A"]
    assert a["n"] == 3 and a["correct"] == 0 and a["by_rule"] == {}
    agree = real["agreement_AB"]
    assert agree is not None
    assert agree["n"] == 3 and agree["n_compared"] == 3
    assert agree["kappa_status"] is not None
    assert real["baseline"] is None and real["baseline_n_mismatch"] is False
    # existing blocks are untouched by the new "real" block
    assert summary["hss"]["runs"]["A"]["n"] == 5


def test_real_block_empty_when_no_real_run_files(results_dir: Path):
    """A directory with no `real_run*.jsonl` (every directory before this task, and every v2
    directory) still gets a `real` key, matching how `hss`/`scifact` behave when absent."""
    sm = load_claims_summarize()
    summary = sm.summarise(results_dir)
    assert summary["real"]["runs"] == {}
    assert summary["real"]["agreement_AB"] is None


REAL_LABELS_ROWS = [
    {"item_id": "real-test-01", "final_label": "verified", "expected_label": ""},
    {"item_id": "real-test-02", "final_label": "unsupported", "expected_label": ""},
    {"item_id": "real-test-03", "final_label": "needs_nuance", "expected_label": ""},
]


@pytest.fixture
def real_labels_csv(tmp_path: Path) -> Path:
    path = tmp_path / "real_annotation_adjudicated_v3.csv"
    _write_labels_csv(path, REAL_LABELS_ROWS)
    return path


def test_real_vs_annotation_scores_runs_against_adjudicated_labels(
    results_dir: Path, real_labels_csv: Path
):
    """--real-labels scores real_run<A|B>.jsonl (not hss_run<A|B>.jsonl) against the
    real-claims adjudication CSV, independently of --labels/hss_vs_annotation."""
    sm = load_claims_summarize()
    _write_real_runs(results_dir)
    assert sm.main(["--results-dir", str(results_dir), "--real-labels", str(real_labels_csv)]) == 0
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert "hss_vs_annotation" not in summary  # --labels was not given
    block = summary["real_vs_annotation"]
    assert block["n_labels"] == 3
    a = block["runs"]["A"]
    assert a["n"] == 3 and a["accuracy"] == pytest.approx(1.0)  # A matches all three labels
    assert a["per_label"]["unsupported"]["recall"] == pytest.approx(1.0)
    assert a["kappa"] == pytest.approx(1.0)
    b = block["runs"]["B"]
    assert b["accuracy"] == pytest.approx(2 / 3)  # B's flip on real-test-02 is now a miss
    assert b["per_label"]["unsupported"]["recall"] == pytest.approx(0.0)


def test_labels_and_real_labels_both_scored_in_one_invocation(
    results_dir: Path, labels_csv: Path, real_labels_csv: Path
):
    """A single summarize.py invocation with both --labels and --real-labels writes both
    hss_vs_annotation and real_vs_annotation, each scored against its own run files and CSV."""
    sm = load_claims_summarize()
    _write_real_runs(results_dir)
    assert sm.main([
        "--results-dir", str(results_dir), "--labels", str(labels_csv),
        "--real-labels", str(real_labels_csv),
    ]) == 0
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["hss_vs_annotation"]["n_labels"] == 5
    assert summary["real_vs_annotation"]["n_labels"] == 3


def test_real_vs_annotation_markdown_omits_construction_prose(
    results_dir: Path, real_labels_csv: Path
):
    """The real-claims set has no construction rule (every `expected_label` is empty), so
    Table E2-g must not report the HSS-only construction-tolerant explanation or the
    construction-vs-adjudicated line, which would otherwise print a misleading 0 % figure."""
    sm = load_claims_summarize()
    _write_real_runs(results_dir)
    sm.main(["--results-dir", str(results_dir), "--real-labels", str(real_labels_csv)])
    md = (results_dir / "summary.md").read_text(encoding="utf-8")
    assert "Table E2-g real claims set vs blind model annotation" in md
    assert "This set has no construction rule" in md
    assert "Construction labels vs the adjudicated labels" not in md
    assert "adjudicated label set contains" not in md


def test_e2c_prov_table_records_concurrency(results_dir: Path):
    """README.md says the concurrency at which latency was measured is 'printed in Tables
    E1-c / E2-c'; E2-c must therefore carry it too, matching E1-c's regression test."""
    sm = load_claims_summarize()
    for f in ("scifact_runA.meta.json", "hss_runA.meta.json"):
        m = json.loads((results_dir / f).read_text(encoding="utf-8"))
        m["concurrency"] = 8
        write_json(results_dir / f, m)
    sm.main(["--results-dir", str(results_dir)])
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["scifact"]["runs"]["A"]["provenance"]["concurrency"] == 8
    md = (results_dir / "summary.md").read_text(encoding="utf-8")
    prov_section = md.split("## Table E2-c")[1]
    assert "concurrency" in prov_section.lower()
    assert "| scifact run A |" in prov_section
    row = next(line for line in prov_section.splitlines() if line.startswith("| scifact run A |"))
    assert "| 8 |" in row


# ---------------------------------------------------------------- --compare-dir


def test_compare_dir_adds_a_version_comparison_block_and_table(
    results_dir: Path, tmp_path: Path
):
    """--compare-dir reads another directory's already-written summary.json (never re-runs
    the verifier) and diffs it against this run's own summary into a Table E2-e delta,
    in addition to every other table, not instead of it."""
    sm = load_claims_summarize()
    v1_dir = tmp_path / "v1"
    # build the "v1" comparison summary from the same fixture rows (delta 0 everywhere)
    v1_dir.mkdir()
    for name in ("scifact_runA.jsonl", "scifact_runB.jsonl", "scifact_runA.meta.json",
                 "scifact_runB.meta.json", "hss_runA.jsonl", "hss_runB.jsonl",
                 "hss_runA.meta.json", "hss_runB.meta.json"):
        (v1_dir / name).write_bytes((results_dir / name).read_bytes())
    assert sm.main(["--results-dir", str(v1_dir)]) == 0

    # "v2": run B's SciFact accuracy improves (item 5 corrected), so a nonzero delta appears
    v2_dir = tmp_path / "v2"
    v2_sci_b = [dict(r) for r in SCI_B]
    v2_sci_b[4]["predicted_status"] = "unsupported"  # already true in fixture SCI_B; keep for B
    append_jsonl(v2_dir / "scifact_runA.jsonl", SCI_A)
    append_jsonl(v2_dir / "scifact_runB.jsonl", v2_sci_b)
    write_json(v2_dir / "scifact_runA.meta.json", meta("scifact", "A", SCI_A))
    write_json(v2_dir / "scifact_runB.meta.json", meta("scifact", "B", v2_sci_b))
    append_jsonl(v2_dir / "hss_runA.jsonl", HSS_A)
    append_jsonl(v2_dir / "hss_runB.jsonl", HSS_B)
    write_json(v2_dir / "hss_runA.meta.json", meta("hss", "A", HSS_A))
    write_json(v2_dir / "hss_runB.meta.json", meta("hss", "B", HSS_B))

    rc = sm.main(["--results-dir", str(v2_dir), "--compare-dir", str(v1_dir)])
    assert rc == 0
    summary = json.loads((v2_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["compare_dir"] == str(v1_dir)
    comparison = summary["version_comparison"]
    assert comparison["scifact"]["A"]["accuracy"]["delta"] == pytest.approx(0.0)
    assert comparison["hss"]["A"]["accuracy"]["delta"] == pytest.approx(0.0)
    md = (v2_dir / "summary.md").read_text(encoding="utf-8")
    assert "Table E2-e" in md and str(v1_dir) in md
    assert "| A | accuracy |" in md


def test_compare_dir_missing_summary_json_warns_and_does_not_fail(
    results_dir: Path, tmp_path: Path, capsys
):
    sm = load_claims_summarize()
    empty_dir = tmp_path / "no-summary-here"
    empty_dir.mkdir()
    rc = sm.main(["--results-dir", str(results_dir), "--compare-dir", str(empty_dir)])
    assert rc == 0
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert "version_comparison" not in summary
    assert "no summary.json" in capsys.readouterr().err


def test_scifact_markdown_prints_zero_f1_for_never_predicted_class(results_dir: Path):
    sm = load_claims_summarize()
    base = json.loads((results_dir / "scifact_baseline.json").read_text(encoding="utf-8"))
    base["per_class"]["verified"] = {"precision": None, "recall": 0.0, "f1": 0.0, "support": 3,
                                     "predicted": 0}
    write_json(results_dir / "scifact_baseline.json", base)
    md = sm.render_markdown(sm.summarise(results_dir))
    assert "| lexical baseline | 8 | - | 0.000 | 0.000 |" in md
    assert "never predicted" in md and "F1 = 0" in md


# ---------------------------------------------------------------- machine_reasons


def test_machine_reasons_counts_tallies_slugs_and_guarded_rows():
    sm = load_claims_summarize()
    rows = [
        {"machine_reasons": ["attribution_mismatch"]},
        {"machine_reasons": ["attribution_mismatch", "quote_not_verbatim"]},
        {"machine_reasons": []},
        {},  # no key at all (v1 rows predate the field)
    ]
    counts = sm.machine_reasons_counts(rows)
    assert counts["by_slug"] == {"attribution_mismatch": 2, "quote_not_verbatim": 1}
    assert counts["n_guarded"] == 2
    assert counts["n"] == 4
    # None of these rows carry predicted_status/model_status (they predate the guard fields
    # entirely), so guarded_count (a guard actually changing the status) must read 0, not
    # silently count the reason-only rows.
    assert counts["guarded_count"] == 0


def test_machine_reasons_counts_guarded_count_only_counts_actual_status_changes():
    """guarded_count is stricter than n_guarded: a fired slug does not by itself mean the
    status changed (a report-only guard, or a status-changing guard capping a status onto
    itself, still appends a slug)."""
    sm = load_claims_summarize()
    rows = [
        # guard fired and tightened the status: model said needs_nuance, guard capped it.
        {"machine_reasons": ["attribution_mismatch"], "model_status": "needs_nuance",
         "predicted_status": "unsupported"},
        # guard fired but is a no-op: it capped the status onto the value the model already gave.
        {"machine_reasons": ["numeric_not_in_source"], "model_status": "unsupported",
         "predicted_status": "unsupported"},
        # no reason, no status change.
        {"machine_reasons": [], "model_status": "verified", "predicted_status": "verified"},
        # deterministic row: no model call was made (model_status is None), so a "change" from
        # None is never counted, even though a guard slug is present.
        {"machine_reasons": ["no_full_text_with_chunks"], "model_status": None,
         "predicted_status": "unsupported"},
    ]
    counts = sm.machine_reasons_counts(rows)
    assert counts["n_guarded"] == 3  # three rows carry at least one reason
    assert counts["guarded_count"] == 1  # only the first row's status was actually changed


def test_machine_reasons_appear_in_scifact_and_hss_run_summaries_and_markdown(
    results_dir: Path,
):
    sm = load_claims_summarize()
    sci_a = json.loads((results_dir / "scifact_runA.jsonl").read_text(encoding="utf-8")
                        .splitlines()[0])
    assert "machine_reasons" not in sci_a  # fixture rows predate the field; must not crash
    summary = sm.summarise(results_dir)
    a = summary["scifact"]["runs"]["A"]
    assert a["machine_reasons"] == {"by_slug": {}, "n_guarded": 0, "guarded_count": 0, "n": 8}
    h = summary["hss"]["runs"]["A"]
    assert h["machine_reasons"] == {"by_slug": {}, "n_guarded": 0, "guarded_count": 0, "n": 5}

    # Rows that do carry guard reasons roll up correctly, including the guarded_count/n_guarded
    # split: row 0 is a no-op firing (model_status == predicted_status), row 1 is a real change.
    sci_guarded = [dict(r) for r in SCI_A]
    sci_guarded[0]["machine_reasons"] = ["numeric_not_in_source"]
    sci_guarded[0]["model_status"] = sci_guarded[0]["predicted_status"]
    sci_guarded[1]["machine_reasons"] = ["numeric_not_in_source", "attribution_mismatch"]
    sci_guarded[1]["model_status"] = "verified"
    sci_guarded[1]["predicted_status"] = "needs_nuance"
    d = results_dir.parent / "guarded"
    append_jsonl(d / "scifact_runA.jsonl", sci_guarded)
    write_json(d / "scifact_runA.meta.json", meta("scifact", "A", sci_guarded))
    summary2 = sm.summarise(d)
    reasons = summary2["scifact"]["runs"]["A"]["machine_reasons"]
    assert reasons["by_slug"] == {"numeric_not_in_source": 2, "attribution_mismatch": 1}
    assert reasons["n_guarded"] == 2
    assert reasons["guarded_count"] == 1  # only row 1's status actually changed

    md = sm.render_markdown(summary2)
    assert "numeric_not_in_source: 2" in md
    assert "attribution_mismatch: 1" in md
    assert "rows with a reason" in md  # renamed column (was "guarded rows")
    assert "guarded_count" in md


# ---------------------------------------------------------------- label-mapping sensitivity


def test_label_mapping_sensitivity_lenient_lets_a_contradict_row_count_as_verified():
    """A gold-CONTRADICT row the model called needs_nuance is a miss under the strict mapping
    but becomes a `predicted verified` row under the lenient mapping (needs_nuance ->
    verified); the sensitivity block must show that shift, not hide it."""
    sm = load_claims_summarize()
    rows = [
        sci_row("1:1", "SUPPORT", "verified", "q", True),
        sci_row("2:1", "CONTRADICT", "needs_nuance", None, None),
        sci_row("3:1", "NOT_ENOUGH_INFO", "unsupported", None, None),
    ]
    block = sm.label_mapping_sensitivity(rows)
    strict = block["strict"]
    assert strict["n_gold_contradict_predicted_verified"] == 0
    assert strict["contradict_recall"] == pytest.approx(0.0)
    assert strict["macro_f1"] == pytest.approx(5 / 6)

    lenient = block["lenient_needs_nuance_as_verified"]
    assert lenient["n_gold_contradict_predicted_verified"] == 1
    assert lenient["contradict_recall"] == pytest.approx(0.0)
    assert lenient["per_class"]["verified"]["precision"] == pytest.approx(0.5)
    assert lenient["per_class"]["verified"]["recall"] == pytest.approx(1.0)
    assert lenient["macro_f1"] == pytest.approx(2 / 3)

    assert block["three_way_contradiction_signal"]["available"] is False
    assert "assertions" in block["three_way_contradiction_signal"]["reason"]


def test_label_mapping_sensitivity_three_way_uses_the_v2_contradiction_signal():
    """v2 rows carry a per-assertion `verdict`; the third mapping splits a predicted
    `unsupported` row by whether any assertion was `contradicted`, exposing that the strict
    two-class CONTRADICT recall (which counts either a NOT_ENOUGH_INFO- or CONTRADICT-flavoured
    `unsupported` as a hit) can be more optimistic than the model's own contradiction signal
    supports."""
    sm = load_claims_summarize()
    rows = [
        sci_row("1:1", "SUPPORT", "verified", "q", True),
        dict(sci_row("2:1", "CONTRADICT", "unsupported", None, None),
             assertions=[{"verdict": "contradicted"}]),
        dict(sci_row("3:1", "NOT_ENOUGH_INFO", "unsupported", None, None),
             assertions=[{"verdict": "absent"}]),
        dict(sci_row("4:1", "CONTRADICT", "unsupported", None, None),
             assertions=[{"verdict": "absent"}]),
    ]
    block = sm.label_mapping_sensitivity(rows)
    three_way = block["three_way_contradiction_signal"]
    assert three_way["available"] is True
    assert three_way["contradict_recall"] == pytest.approx(0.5)
    assert three_way["n_gold_contradict_predicted_verified"] == 0
    pc = three_way["per_class"]
    assert pc["verified"]["f1"] == pytest.approx(1.0)
    assert pc["unsupported_contradicted"]["precision"] == pytest.approx(1.0)
    assert pc["unsupported_contradicted"]["recall"] == pytest.approx(0.5)
    assert pc["unsupported_other"]["precision"] == pytest.approx(0.5)
    assert pc["unsupported_other"]["recall"] == pytest.approx(1.0)
    assert three_way["macro_f1"] == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)

    # strict CONTRADICT recall on the same rows is more optimistic (both rows counted correct
    # since NOT_ENOUGH_INFO and CONTRADICT are conflated under "unsupported")
    assert block["strict"]["contradict_recall"] == pytest.approx(1.0)


def test_label_mapping_sensitivity_present_in_scifact_run_summary_and_markdown(results_dir: Path):
    sm = load_claims_summarize()
    sm.main(["--results-dir", str(results_dir)])
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    a = summary["scifact"]["runs"]["A"]
    assert "label_mapping_sensitivity" in a
    assert a["label_mapping_sensitivity"]["three_way_contradiction_signal"]["available"] is False

    md = (results_dir / "summary.md").read_text(encoding="utf-8")
    assert "Label-mapping sensitivity" in md
    assert "lenient" in md.lower()
    assert "assertions" in md  # explains why the third mapping is skipped for v1-shaped rows


def test_label_mapping_sensitivity_three_way_table_appears_when_assertions_present(
    tmp_path: Path,
):
    sm = load_claims_summarize()
    d = tmp_path / "results"
    rows_a = [
        sci_row("1:1", "SUPPORT", "verified", "q", True),
        dict(sci_row("2:1", "CONTRADICT", "unsupported", None, None),
             assertions=[{"verdict": "contradicted"}]),
    ]
    append_jsonl(d / "scifact_runA.jsonl", rows_a)
    write_json(d / "scifact_runA.meta.json", meta("scifact", "A", rows_a))
    assert sm.main(["--results-dir", str(d)]) == 0
    md = (d / "summary.md").read_text(encoding="utf-8")
    assert "three_way" not in md  # the raw JSON key is not printed verbatim; prose only
    assert "contradiction signal" in md.lower()
    assert "unsupported_contradicted" not in md  # readable labels, not the raw key, in markdown


def test_committed_claims_summary_has_label_mapping_sensitivity_for_both_runs():
    """The committed evaluation/claims/results/v6/summary.json carries the new block for
    both runs, with a non-null CONTRADICT recall (v6 rows all carry the per-assertion
    `assertions` field the third mapping needs). A prompt whose rows carry no `assertions`
    field (the shape every run had before that field was added) must instead leave the third
    mapping unavailable, checked here against a small synthetic row set rather than a
    committed file, since no shipped run still has that shape."""
    summary = json.loads(
        (CLAIMS_DIR / "results" / "v6" / "summary.json").read_text(encoding="utf-8")
    )
    for run in ("A", "B"):
        lms = summary["scifact"]["runs"][run]["label_mapping_sensitivity"]
        assert "strict" in lms and "lenient_needs_nuance_as_verified" in lms
        three_way = lms["three_way_contradiction_signal"]
        assert three_way["available"] is True
        assert three_way["contradict_recall"] is not None

    sm = load_claims_summarize()
    rows_no_assertions = [
        sci_row("1:1", "SUPPORT", "verified", "q", True),
        sci_row("2:1", "CONTRADICT", "unsupported", None, None),
    ]
    block = sm.label_mapping_sensitivity(rows_no_assertions)
    assert block["three_way_contradiction_signal"]["available"] is False


# ---------------------------------------------------------------- six HSS rules


def test_hss_rules_extended_to_six_and_over_specified_bucket_appears():
    sm = load_claims_summarize()
    assert sm.HSS_RULES == (
        "verbatim", "paraphrase", "altered", "over_specified", "wrong_paper", "no_full_text",
    )
    rows = [
        hss_row("verbatim", "verified", ["verified"]),
        hss_row("paraphrase", "verified", ["verified"]),
        hss_row("altered", "unsupported", ["unsupported", "needs_nuance"]),
        hss_row("over_specified", "needs_nuance", ["needs_nuance"]),
        hss_row("wrong_paper", "unsupported", ["unsupported"]),
        hss_row("no_full_text", "no_full_text", ["no_full_text"]),
    ]
    s = sm.hss_run_summary(rows, meta("hss", "A", rows))
    assert set(s["by_rule"]) == set(sm.HSS_RULES)
    assert s["by_rule"]["over_specified"]["n"] == 1
    assert s["by_rule"]["over_specified"]["correct"] == 1


# ---------------------------------------------------------------- by_alteration


def test_operator_of_splits_on_first_colon_and_unknown_for_missing():
    sm = load_claims_summarize()
    assert sm._operator_of("numeric:26.3->52.6") == "numeric"
    assert sm._operator_of("numeric:26.3->31.3") == "numeric"
    assert sm._operator_of("direction_flip:higher->lower") == "direction_flip"
    assert sm._operator_of(None) == "unknown"
    assert sm._operator_of("") == "unknown"
    assert sm._operator_of("no-colon-here") == "unknown"


def test_by_alteration_buckets_by_operator_and_reports_counts():
    sm = load_claims_summarize()
    rows = [
        hss_row("altered", "unsupported", ["unsupported"], alteration="numeric:26.3->52.6"),
        hss_row("altered", "verified", ["unsupported"], alteration="numeric:26.3->31.3"),
        hss_row("altered", "unsupported", ["unsupported"],
                alteration="direction_flip:higher->lower"),
        hss_row("altered", "unsupported", ["unsupported"], alteration=None),
    ]
    out = sm.by_alteration(rows)
    assert out["numeric"]["n"] == 2 and out["numeric"]["correct"] == 1
    assert out["numeric"]["status_counts"] == {"unsupported": 1, "verified": 1}
    assert out["numeric"]["unsupported_rate"] == pytest.approx(0.5)
    assert out["direction_flip"]["n"] == 1 and out["direction_flip"]["accuracy"] == 1.0
    assert out["unknown"]["n"] == 1


def test_by_alteration_round_trips_from_run_hss_item_keys():
    """The alteration field flows item_id -> run_hss.ITEM_KEYS -> the run row (brief section
    2.2 item 4: run_hss.ITEM_KEYS gains ``alteration``, which it drops today)."""
    import run_hss as rh

    claim_row = {
        "item_id": "hss-altered-01", "rule": "altered", "expected": ["unsupported"],
        "claim": "c", "source_doi": "d", "chunk_doi": "d", "chunk_index": 0,
        "alteration": "numeric:10->20", "title": "T", "authors": None,
    }
    item = {k: claim_row.get(k) for k in rh.ITEM_KEYS}
    assert item["alteration"] == "numeric:10->20"
    sm = load_claims_summarize()
    row_without_alteration = {"rule": "altered"}  # a pre-v3 row
    assert sm._operator_of(row_without_alteration.get("alteration")) == "unknown"


# ---------------------------------------------------------------- machine_reasons_by_rule


def test_machine_reasons_by_rule_buckets_per_rule():
    sm = load_claims_summarize()
    rows = [
        {"rule": "over_specified", "machine_reasons": ["attribution_mismatch"]},
        {"rule": "over_specified", "machine_reasons": []},
        {"rule": "altered", "machine_reasons": ["assertion_status_inconsistent"]},
    ]
    out = sm.machine_reasons_by_rule(rows)
    assert out["over_specified"]["by_slug"] == {"attribution_mismatch": 1}
    assert out["over_specified"]["n"] == 2
    assert out["altered"]["by_slug"] == {"assertion_status_inconsistent": 1}


# ---------------------------------------------------------------- diagnostics_counts


def test_diagnostics_counts_mirrors_machine_reasons_counts():
    sm = load_claims_summarize()
    rows = [
        {"diagnostics": ["numeric_not_in_source"]},
        {"diagnostics": ["numeric_not_in_source", "centrality_unmarked"]},
        {"diagnostics": []},
        {},  # pre-v3 row: no diagnostics field at all -> counts as no diagnostics
    ]
    out = sm.diagnostics_counts(rows)
    assert out["by_slug"] == {"numeric_not_in_source": 2, "centrality_unmarked": 1}
    assert out["n_flagged"] == 2
    assert out["n"] == 4


def test_diagnostics_is_a_sibling_of_machine_reasons_in_every_run_block(results_dir: Path):
    sm = load_claims_summarize()
    summary = sm.summarise(results_dir)
    for name in ("scifact", "hss"):
        for run, s in summary[name]["runs"].items():
            assert "diagnostics" in s and "machine_reasons" in s
            assert s["diagnostics"] == {"by_slug": {}, "n_flagged": 0, "n": s["n"]}
            assert "diagnostics" not in s["machine_reasons"]  # never a member of the other


# ---------------------------------------------------------------- wilson_ci


def test_wilson_ci_returns_the_documented_fields():
    sm = load_claims_summarize()
    ci = sm.wilson_ci(54, 60)
    assert set(ci) == {"lo", "hi", "width", "n", "n_correct", "method"}
    assert ci["n"] == 60 and ci["n_correct"] == 54 and ci["method"] == "wilson"
    assert ci["width"] == pytest.approx(ci["hi"] - ci["lo"])
    assert 0.0 < ci["lo"] < ci["hi"] < 1.0


# ---------------------------------------------------------------- block selectors


def test_hss_name_selector_auto_prefers_plain_falls_back_to_prefixed(tmp_path: Path):
    sm = load_claims_summarize()
    d = tmp_path / "band2_only"
    append_jsonl(d / "hss-band2_runA.jsonl", HSS_A)
    write_json(d / "hss-band2_runA.meta.json", meta("hss", "A", HSS_A))
    summary = sm.summarise(d)  # default hss_name="auto"
    assert summary["hss"]["runs"]["A"]["n"] == 5

    d2 = tmp_path / "plain_only"
    append_jsonl(d2 / "hss_runA.jsonl", HSS_A)
    write_json(d2 / "hss_runA.meta.json", meta("hss", "A", HSS_A))
    summary2 = sm.summarise(d2)
    assert summary2["hss"]["runs"]["A"]["n"] == 5


def test_hss_name_selector_explicit_with_no_matching_file_is_empty_not_wrong(tmp_path: Path):
    sm = load_claims_summarize()
    d = tmp_path / "results"
    append_jsonl(d / "hss_runA.jsonl", HSS_A)
    write_json(d / "hss_runA.meta.json", meta("hss", "A", HSS_A))
    summary = sm.summarise(d, hss_name="hss-band2")
    assert summary["hss"]["runs"] == {}


def test_scifact_name_selector_resolves_scifact_train_under_auto_and_explicit(tmp_path: Path):
    sm = load_claims_summarize()
    d = tmp_path / "results"
    append_jsonl(d / "scifact-train_runA.jsonl", SCI_A)
    write_json(d / "scifact-train_runA.meta.json", meta("scifact-train", "A", SCI_A))
    summary = sm.summarise(d)  # default scifact_name="auto"
    assert summary["scifact"]["runs"]["A"]["n"] == 8
    summary2 = sm.summarise(d, scifact_name="scifact-train")
    assert summary2["scifact"]["runs"]["A"]["n"] == 8
    summary3 = sm.summarise(d, scifact_name="scifact")
    assert summary3["scifact"]["runs"] == {}


def test_cli_name_selectors_wired_through_main(tmp_path: Path):
    sm = load_claims_summarize()
    d = tmp_path / "results"
    append_jsonl(d / "scifact-train_runA.jsonl", SCI_A)
    write_json(d / "scifact-train_runA.meta.json", meta("scifact-train", "A", SCI_A))
    assert sm.main(["--results-dir", str(d), "--scifact-name", "scifact-train"]) == 0
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert summary["scifact"]["runs"]["A"]["n"] == 8
    args = sm.parse_args([])
    assert args.hss_name == "auto" and args.scifact_name == "auto"

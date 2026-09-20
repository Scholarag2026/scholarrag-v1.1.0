"""spot_check_hydration.py: per-record hand judgements and the committed evidence file."""

from __future__ import annotations

import json

import pytest
from conftest import load_script_module

httpx = pytest.importorskip("httpx")


def test_apply_judgements_fills_per_record_fields_and_totals():
    sc = load_script_module("screening", "spot_check_hydration")
    records = [{"record_id": 1}, {"record_id": 2}, {"record_id": 3}]
    judgements = {"1": {"judgement": "correct"},
                  "2": {"judgement": "incorrect", "note": "yearbook reprint"}}
    totals = sc.apply_judgements(records, judgements, judged_by="me", judged_on="2026-09-03")
    assert records[0]["judgement"] == "correct" and records[0]["judged_by"] == "me"
    assert records[0]["judged_on"] == "2026-09-03" and records[0]["judgement_note"] is None
    assert records[1]["judgement"] == "incorrect"
    assert records[1]["judgement_note"] == "yearbook reprint"
    assert records[2]["judgement"] is None and records[2]["judged_by"] is None
    assert totals == {"n_correct": 1, "n_incorrect": 1, "n_unsure": 0, "n_unjudged": 1}
    with pytest.raises(ValueError):
        sc.apply_judgements(records, {"1": {"judgement": "maybe"}}, judged_by="me",
                            judged_on="2026-09-03")


def test_main_merges_judgements_and_writes_to_out(tmp_path, monkeypatch):
    sc = load_script_module("screening", "spot_check_hydration")
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"record_id": i, "title": f"T {i}", "doi": f"10.1/{i}", "issn": ["1"],
             "match_method": "title_exact_year", "year": 2010} for i in range(5)]
    (data / "X.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(sc, "crossref_record", lambda client, doi: {"title": doi, "year": 2010})
    judgements = tmp_path / "j.json"
    judgements.write_text(json.dumps({"records": {"0": {"judgement": "correct"},
                                                  "1": {"judgement": "unsure"}}}), encoding="utf-8")
    out = tmp_path / "protocols" / "spot_checks" / "X.spot_check.json"
    rc = sc.main(["--dataset", "X", "--n", "2", "--seed", "1", "--data-dir", str(data),
                  "--judgements", str(judgements), "--judged-by", "me",
                  "--judged-on", "2026-09-03", "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n"] == 2 and payload["judged_by"] == "me"
    assert {r["judgement"] for r in payload["records"]} <= {"correct", "unsure", "incorrect", None}
    judged = ("n_correct", "n_unsure", "n_incorrect", "n_unjudged")
    assert sum(payload[k] for k in judged) == 2
    assert all("judgement" in r and "judged_on" in r for r in payload["records"])


def test_title_exact_matches_carry_no_export_year(tmp_path, monkeypatch):
    """match_method title_exact means the export had no year to check, so the 'year' on the
    record is the hydrated Crossref year: the evidence file must not present it as SYNERGY's."""
    sc = load_script_module("screening", "spot_check_hydration")
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"record_id": 0, "title": "T 0", "doi": "10.1/0", "issn": ["1"],
             "match_method": "title_exact", "year": 2010},
            {"record_id": 1, "title": "T 1", "doi": "10.1/1", "issn": ["1"],
             "match_method": "title_exact_year", "year": 2011}]
    (data / "X.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(sc, "crossref_record", lambda client, doi: {"title": doi, "year": 2010})
    out = tmp_path / "X.spot_check.json"
    assert sc.main(["--dataset", "X", "--n", "2", "--seed", "1", "--data-dir", str(data),
                    "--out", str(out)]) == 0
    by_id = {r["record_id"]: r for r in json.loads(out.read_text(encoding="utf-8"))["records"]}
    assert by_id[0]["synergy_year"] is None and by_id[1]["synergy_year"] == 2011
    assert sc.export_year(rows[0]) is None and sc.export_year(rows[1]) == 2011

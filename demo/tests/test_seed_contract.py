"""Offline contract tests for demo/protocol.json, demo/seed_dois.json and the audit files."""
import json
import re
from datetime import date
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Verbatim copy of backend/app/services/fulltext.py::_CITATION_RE. The backend only
# extracts sentences whose citation matches this pattern, so every protocol claim must.
BACKEND_CITATION_RE = re.compile(
    r"[\[\(]"
    r"([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?)"
    r",?\s*"
    r"(\d{4})"
    r"[\]\)]"
)
SURNAME_TOKEN_RE = re.compile(r"^[A-Z][a-z]+$")


def _load(name):
    return json.loads((DEMO / name).read_text(encoding="utf-8"))


def _backend_key(claim_text: str) -> str:
    """'surname_year' exactly as fulltext.py::_extract_claims derives it."""
    match = BACKEND_CITATION_RE.search(claim_text)
    assert match, f"backend regex does not match: {claim_text!r}"
    surname = match.group(1).split()[0]
    assert SURNAME_TOKEN_RE.match(surname), surname
    return f"{surname.lower()}_{match.group(2)}"


def test_protocol_contract():
    p = _load("protocol.json")
    assert set(p) == {"topic", "research_question", "inclusion_criteria", "exclusion_criteria",
                      "wos_filter", "publication_date_max", "section", "claims", "notes"}
    assert p["wos_filter"] == "off"
    # The day before the last promoted demo run's date (summary.json's own
    # run_finished_at), so a replayed run sees the same OpenAlex corpus.
    assert DATE_RE.match(p["publication_date_max"])
    assert 3 <= len(p["inclusion_criteria"]) <= 4
    assert 3 <= len(p["exclusion_criteria"]) <= 4
    assert set(p["section"]) == {"title", "instructions", "target_words"}
    assert isinstance(p["section"]["target_words"], int) and p["section"]["target_words"] > 0
    assert [c["id"] for c in p["claims"]] == ["supported-1", "unsupported-1"]
    statuses = {c["id"]: c["expected_status"] for c in p["claims"]}
    assert statuses == {"supported-1": "verified", "unsupported-1": "unsupported"}
    for c in p["claims"]:
        assert set(c) == {"id", "text", "source_doi", "expected_status", "construction"}
        assert c["text"].endswith(").")
        _backend_key(c["text"])
        # the backend splits on sentence boundaries; a claim must stay one sentence
        assert len(re.split(r"(?<=[.!?])\s+", c["text"])) == 1, c["text"]
        assert c["construction"]
    assert "authors' run are recorded in demo/expected/summary.json" in p["notes"]


def test_claim_source_is_a_seed_paper():
    p = _load("protocol.json")
    seeds = {s["doi"] for s in _load("seed_dois.json")["papers"]}
    for c in p["claims"]:
        assert c["source_doi"] in seeds


def test_claim_citation_keys_to_the_seed_paper_and_only_it():
    """Cited surname == last token of the source paper's first author; key unique in library."""
    p = _load("protocol.json")
    selected = {s["doi"]: s for s in _load("tools/selected.json")}
    seeds = _load("seed_dois.json")["papers"]
    keys = {}
    for seed in seeds:
        sel = selected[seed["doi"]]
        surname = sel["authors_crossref"][0].split(",")[0].split()[-1]
        assert sel["first_author_surname"] == surname
        assert sel["citation_key"] == f"{surname.lower()}_{seed['year']}"
        keys.setdefault(sel["citation_key"], []).append(seed["doi"])
    for c in p["claims"]:
        key = _backend_key(c["text"])
        assert key == selected[c["source_doi"]]["citation_key"], (key, c["source_doi"])
        assert keys[key] == [c["source_doi"]], "surname_year key must be unique in the library"


def test_seed_dois_contract():
    s = _load("seed_dois.json")
    assert set(s) == {"topic", "selected_at", "selection_method", "papers"}
    assert DATE_RE.match(s["selected_at"])
    date.fromisoformat(s["selected_at"])
    assert "fetch_pdf_from_url" in s["selection_method"]
    papers = s["papers"]
    assert len(papers) == 12
    assert len({p["doi"] for p in papers}) == 12
    for p in papers:
        assert set(p) == {"doi", "title", "year", "journal", "openalex_id", "oa_pdf_url",
                          "oa_verified_via", "oa_verified_at", "pdf_http_status",
                          "pdf_content_type"}
        assert p["doi"].startswith("10.") and "doi.org" not in p["doi"]
        assert isinstance(p["year"], int) and 2015 <= p["year"] <= 2025
        assert p["openalex_id"].startswith("W")
        assert p["oa_pdf_url"].startswith("https://")
        assert p["oa_verified_via"] in {"unpaywall", "openalex"}
        assert DATE_RE.match(p["oa_verified_at"])
        assert p["pdf_http_status"] == 200
        assert p["pdf_content_type"] == "application/pdf"
    assert len({p["journal"] for p in papers}) >= 4
    assert len({p["year"] for p in papers}) >= 4


def test_seeds_passed_the_backend_client_check():
    """Every seed's audit record was produced with the backend-equivalent client and passed."""
    verified = {r["doi"]: r for r in _load("tools/verified.json")}
    for seed in _load("seed_dois.json")["papers"]:
        rec = verified[seed["doi"]]
        assert rec["client"] == "backend" and rec["ok"] is True, seed["doi"]
        assert rec["http"]["status"] == 200 and rec["http"]["ctype"] == "application/pdf"
        assert rec["http"]["is_pdf"] is True
        assert rec["checked_url"] == seed["oa_pdf_url"]
        assert rec["route"] == seed["oa_verified_via"]


def test_sources_md_counts_match_the_audit_files():
    """The PRISMA-style numbers in SOURCES.md are derived from the audit JSON, not typed."""
    text = (DEMO / "SOURCES.md").read_text(encoding="utf-8")
    verified = _load("tools/verified.json")
    seeds = _load("seed_dois.json")["papers"]
    passed = [r for r in verified if r["ok"]]
    rejected = [r for r in verified if not r["ok"]]
    kept = {p["doi"] for p in seeds}
    not_selected = [r for r in passed if r["doi"] not in kept]
    assert f"## Verified and kept ({len(seeds)})" in text
    assert f"## Verified but not selected ({len(not_selected)})" in text
    assert f"## Rejected ({len(rejected)})" in text
    assert f"{len(verified)} candidates" in text
    assert f"{len(passed)} of {len(verified)} passed" in text
    journals = len({p['journal'] for p in seeds})
    assert f"{journals} journals" in text
    sources = _load("tools/sources.json")
    assert f"{len(sources)} journal" in text
    for r in rejected:
        assert r["doi"] in text
    for r in not_selected:
        assert r["doi"] in text

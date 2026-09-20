"""Pure helpers of screening/fetch_synergy.py (no network)."""

from __future__ import annotations

import json

import pytest

httpx = pytest.importorskip("httpx")

from fetch_synergy import (  # noqa: E402
    OPENALEX_SELECT,
    dataset_counts,
    normalise_doi,
    normalise_title,
    parse_ids_csv,
    parse_v1_csv,
    reconstruct_abstract,
    short_openalex_id,
    to_int_label,
    work_to_fields,
)

V1_CSV = (
    "record_id,title,abstract,label_included,label_abstract_screening,duplicate_record_id\n"
    "1,A trial of reminders,Reminders increased compliance.,0,1,\n"
    "2,Unrelated paper,,0,0,\n"
    "3,Nudging physicians,Defaults changed prescribing.,1,1,\n"
)

IDS_CSV = (
    "doi,openalex_id,label_included,pmid,method\n"
    "https://doi.org/10.1037/a0030642,https://openalex.org/W2159532703,1,,search_title\n"
    ",,0,,\n"
)


def test_to_int_label():
    assert to_int_label("1") == 1 and to_int_label("0") == 0 and to_int_label("") is None
    assert to_int_label(None) is None and to_int_label(True) == 1 and to_int_label("1.0") == 1
    assert to_int_label("x") is None


def test_parse_v1_csv():
    recs = parse_v1_csv(V1_CSV)
    assert [r["record_id"] for r in recs] == [1, 2, 3]
    assert recs[0]["label_included"] == 0 and recs[0]["label_abstract_screening"] == 1
    assert recs[1]["abstract"] == "" and recs[1]["issn"] == [] and recs[1]["openalex_id"] is None
    assert recs[2]["source_route"] == "v1"
    counts = dataset_counts(recs)
    assert counts["n"] == 3 and counts["n_included"] == 1
    assert counts["n_abstract_screening_included"] == 2 and counts["n_missing_abstract"] == 1


def test_parse_ids_csv_and_id_normalisation():
    recs = parse_ids_csv(IDS_CSV)
    assert recs[0]["openalex_id"] == "W2159532703" and recs[0]["doi"] == "10.1037/a0030642"
    assert recs[0]["label_included"] == 1 and recs[0]["method"] == "search_title"
    assert recs[1]["openalex_id"] is None and recs[1]["record_id"] == 2
    assert short_openalex_id("w1") == "W1" and short_openalex_id("") is None
    assert normalise_doi("DOI:10.1/X") == "10.1/x" and normalise_doi(None) is None


def test_reconstruct_abstract_and_work_fields():
    inv = {"compliance.": [2], "Reminders": [0], "increased": [1]}
    assert reconstruct_abstract(inv) == "Reminders increased compliance."
    assert reconstruct_abstract(None) == ""
    work = {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.1/abc",
        "title": "A Title",
        "publication_year": 2019,
        "type": "article",
        "abstract_inverted_index": inv,
        "primary_location": {
            "source": {
                "issn_l": "1234-5678",
                "issn": ["1234-5678", "8765-4321"],
                "display_name": "J",
            }
        },
    }
    fields = work_to_fields(work)
    assert fields["openalex_id"] == "W1" and fields["doi"] == "10.1/abc"
    assert fields["issn"] == ["1234-5678", "8765-4321"] and fields["year"] == 2019
    assert work_to_fields({"id": "https://openalex.org/W2"})["issn"] == []


def test_work_to_fields_carries_is_paratext_beside_type():
    """A re-fetched route v2 export must carry is_paratext beside the type it already
    fetches, so app.agents.relevance_screener_agent.apply_type_demotion can be applied to
    it downstream without a second OpenAlex request."""
    assert "is_paratext" in OPENALEX_SELECT.split(",")
    fields = work_to_fields({"id": "https://openalex.org/W3", "type": "paratext", "is_paratext": True})
    assert fields["type"] == "paratext"
    assert fields["is_paratext"] is True
    assert work_to_fields({"id": "https://openalex.org/W4"})["is_paratext"] is False


def test_normalise_title():
    assert normalise_title("  The Effect: of Nudges!! ") == "the effect of nudges"
    assert normalise_title(None) == ""


V1_CSV_WITH_IDS = (
    "record_id,title,abstract,keywords,authors,year,date,doi,label_included,"
    "label_abstract_screening,duplicate_record_id\n"
    "1,PTSD trajectories,Three waves.,ptsd,A B,2010,2010-01-01,https://doi.org/10.1/ABC,1,1,\n"
    "2,No doi paper,,,C D,,,,0,0,\n"
    "3,Bad year,,,E F,n/a,,DOI:10.2/xyz,0,1,\n"
)


def test_parse_v1_csv_keeps_doi_and_year_when_present():
    recs = parse_v1_csv(V1_CSV_WITH_IDS)
    assert recs[0]["doi"] == "10.1/abc" and recs[0]["year"] == 2010
    assert recs[1]["doi"] is None and recs[1]["year"] is None
    assert recs[2]["doi"] == "10.2/xyz" and recs[2]["year"] is None
    assert all(r["openalex_id"] is None and r["issn"] == [] for r in recs)
    plain = parse_v1_csv(V1_CSV)
    assert all(r["doi"] is None and r["year"] is None for r in plain)
    assert dataset_counts(recs)["n_with_doi"] == 2


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    """Answers any OpenAlex ``works`` query with a canned payload; records the calls."""

    def __init__(self, works):
        self.works = works
        self.calls: list[dict] = []

    def get(self, url, params=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        return _FakeResponse({"results": self.works})


def _work(oid, doi, title, year=2011, issn=("1234-5678",), abstract=None):
    return {
        "id": f"https://openalex.org/{oid}",
        "doi": f"https://doi.org/{doi}",
        "title": title,
        "publication_year": year,
        "type": "article",
        "abstract_inverted_index": ({w: [i] for i, w in enumerate(abstract.split())}
                                    if abstract else None),
        "primary_location": {"source": {"issn_l": issn[0], "issn": list(issn)}},
    }


def test_hydrate_by_doi_fills_ids_and_leaves_unmatched(monkeypatch):
    import fetch_synergy as fs

    client = _FakeClient([_work("W1", "10.1/abc", "PTSD trajectories", abstract="From OA")])
    monkeypatch.setattr(fs, "fetch_with_retries", lambda c, url, **kw: c.get(url, kw.get("params")))
    recs = parse_v1_csv(V1_CSV_WITH_IDS)
    hits = fs.hydrate_by_doi(recs, client, "x@example.org", fs.RateLimiter(1000))
    assert hits == 1
    assert recs[0]["openalex_id"] == "W1" and recs[0]["issn"] == ["1234-5678"]
    assert recs[0]["match_method"] == "doi" and recs[0]["year"] == 2010
    assert recs[0]["abstract"] == "Three waves." and "abstract_source" not in recs[0]
    # record 3 has a DOI but no OpenAlex hit: untouched
    assert recs[2]["openalex_id"] is None and recs[2]["match_method"] is None
    # record 2 (no doi) is never queried
    assert len(client.calls) == 1
    assert "doi:10.1/abc|10.2/xyz" in client.calls[0]["params"]["filter"]
    assert client.calls[0]["params"]["mailto"] == "x@example.org"

    # an empty abstract is NOT filled from OpenAlex (the export text is what humans screened)
    recs[0].update(abstract="", openalex_id=None, match_method=None)
    fs.hydrate_by_doi(recs, client, None, fs.RateLimiter(1000))
    assert recs[0]["abstract"] == "" and "abstract_source" not in recs[0]


def test_main_v1_writes_jsonl_and_fetch_json(tmp_path, monkeypatch):
    import fetch_synergy as fs

    index = tmp_path / "index.csv"
    index.write_text(
        "dataset_id,url,reference\nX_2017,https://example.org/X.csv,https://doi.org/10.1/r\n",
        encoding="utf-8",
    )
    oa = _FakeClient([_work("W1", "10.1/abc", "PTSD trajectories")])

    class _Text:
        text = V1_CSV_WITH_IDS

    def fake_fetch(client, url, **kw):
        if url == "https://example.org/X.csv":
            return _Text()
        return oa.get(url, kw.get("params"))

    monkeypatch.setattr(fs, "fetch_with_retries", fake_fetch)
    out_dir = tmp_path / "out"
    rc = fs.main(
        ["--dataset", "X_2017", "--route", "v1", "--hydrate", "included",
         "--index", str(index), "--out-dir", str(out_dir), "--email", "e@x.org"]
    )
    assert rc == 0
    rows = [json.loads(line) for line in (out_dir / "X_2017.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["openalex_id"] == "W1" and rows[0]["match_method"] == "doi"
    info = json.loads((out_dir / "X_2017.fetch.json").read_text(encoding="utf-8"))
    assert info["dataset_id"] == "X_2017" and info["route"] == "v1"
    assert info["source_url"] == "https://example.org/X.csv" and info["hydrate"] == "included"
    assert info["n"] == 3 and info["n_included"] == 1 and info["n_abstract_screening_included"] == 2
    assert info["n_missing_abstract"] == 2
    assert info["share_missing_abstract"] == pytest.approx(2 / 3)
    assert info["n_with_doi"] == 2 and info["n_with_openalex_id"] == 1 and info["n_with_issn"] == 1
    assert info["fetched_at"].startswith("20") and info["fetched_at"].endswith("+00:00")


def test_main_v2_never_opens_index(tmp_path, monkeypatch):
    import fetch_synergy as fs

    oa = _FakeClient([_work("W2159532703", "10.1037/a0030642", "Hydrated title", abstract="A b")])

    class _Text:
        text = IDS_CSV

    def fake_fetch(client, url, **kw):
        if url.endswith("_ids.csv"):
            assert "Smid_2020" in url
            return _Text()
        return oa.get(url, kw.get("params"))

    monkeypatch.setattr(fs, "fetch_with_retries", fake_fetch)

    def boom(*a, **k):
        raise AssertionError("index must not be touched on route v2")

    monkeypatch.setattr(fs, "index_row", boom)
    monkeypatch.setattr(fs, "ensure_index", boom)
    out_dir = tmp_path / "out"
    rc = fs.main(["--dataset", "Smid_2020", "--route", "v2", "--out-dir", str(out_dir),
                  "--index", str(tmp_path / "absent.csv")])
    assert rc == 0
    rows = [json.loads(line) for line in (out_dir / "Smid_2020.jsonl").read_text().splitlines()]
    assert rows[0]["title"] == "Hydrated title" and rows[0]["match_method"] == "openalex_id"
    info = json.loads((out_dir / "Smid_2020.fetch.json").read_text(encoding="utf-8"))
    assert info["route"] == "v2" and info["source_url"].endswith("Smid_2020_ids.csv")
    assert info["n"] == 1 and info["hydrate"] == "openalex_id"  # row without id dropped
    assert info["n_rows_in_ids_csv"] == 2 and info["n_dropped_no_openalex_id"] == 1
    assert info["n_duplicate_ids"] == 0


def test_title_query_strips_punctuation_and_lookup_failures_are_skipped(monkeypatch):
    import fetch_synergy as fs

    assert fs.title_query('Is resilience the modal outcome? "PTSD": a, b | c') == (
        "Is resilience the modal outcome PTSD a b c"
    )

    def failing_fetch(client, url, **kw):
        raise RuntimeError("giving up: HTTP 400")

    monkeypatch.setattr(fs, "fetch_with_retries", failing_fetch)
    recs = [{"record_id": 1, "title": "A sufficiently long title for lookup", "abstract": ""}]
    assert fs.hydrate_by_title(recs, _FakeClient([]), None, fs.RateLimiter(1000)) == 0
    assert recs[0].get("openalex_id") is None


def test_dedupe_v2_records_drops_unresolved_and_merges_duplicates():
    import fetch_synergy as fs

    recs = parse_ids_csv(
        "doi,openalex_id,label_included,pmid,method\n"
        "10.1/a,https://openalex.org/W1,0,,id_retrieval_doi\n"
        ",,0,,\n"
        "10.1/a,https://openalex.org/W1,1,,search_title\n"
        "10.1/b,https://openalex.org/W2,0,,search_title\n"
    )
    kept, info = fs.dedupe_v2_records(recs)
    assert [r["openalex_id"] for r in kept] == ["W1", "W2"]
    assert kept[0]["label_included"] == 1 and kept[0]["record_id"] == 1
    assert info == {"n_rows_in_ids_csv": 4, "n_dropped_no_openalex_id": 1, "n_duplicate_ids": 1}


def test_retry_after_is_capped_and_api_key_is_passed(monkeypatch):
    import fetch_synergy as fs

    sleeps: list[float] = []
    monkeypatch.setattr(fs.time, "sleep", lambda s: sleeps.append(s))

    class _Resp:
        status_code = 429
        headers = {"Retry-After": "84951"}
        request = None
        text = "quota"

    class _Client:
        def get(self, url, params=None):
            return _Resp()

    with pytest.raises(RuntimeError):
        fs.fetch_with_retries(_Client(), "https://api.openalex.org/works", attempts=2)
    assert sleeps and max(sleeps) <= fs.MAX_RETRY_DELAY_S
    params = fs.openalex_params("e@x.org", api_key="k123")
    assert params["api_key"] == "k123" and params["mailto"] == "e@x.org"
    assert "api_key" not in fs.openalex_params("e@x.org")


# ---------------------------------------------------------------- additional cases


def test_hydration_targets_cover_every_positive_label():
    import fetch_synergy as fs

    recs = [
        {"record_id": 1, "label_included": 1, "label_abstract_screening": 1},
        {"record_id": 2, "label_included": 0, "label_abstract_screening": 1},  # TA-only
        {"record_id": 3, "label_included": 0, "label_abstract_screening": 0},
        {"record_id": 4, "label_included": 0, "label_abstract_screening": None},
    ]
    assert [r["record_id"] for r in fs.hydration_targets(recs, "included")] == [1, 2]
    assert [r["record_id"] for r in fs.hydration_targets(recs, "all")] == [1, 2, 3, 4]
    assert fs.hydration_targets(recs, "none") == []


def test_openalex_hydration_never_fills_abstracts(monkeypatch):
    import fetch_synergy as fs

    client = _FakeClient([_work("W1", "10.1/abc", "PTSD trajectories", abstract="From OA")])
    monkeypatch.setattr(fs, "fetch_with_retries", lambda c, url, **kw: c.get(url, kw.get("params")))
    recs = parse_v1_csv(V1_CSV_WITH_IDS)
    recs[0]["abstract"] = ""
    fs.hydrate_by_doi(recs, client, "x@example.org", fs.RateLimiter(1000))
    assert recs[0]["issn"] == ["1234-5678"] and recs[0]["abstract"] == ""
    assert "abstract_source" not in recs[0] and recs[0]["lookup_source"] == "openalex"
    recs = [{"record_id": 9, "title": "PTSD trajectories", "abstract": "", "openalex_id": None}]
    fs.hydrate_by_title(recs, client, None, fs.RateLimiter(1000))
    assert recs[0]["issn"] == ["1234-5678"] and recs[0]["abstract"] == ""
    assert recs[0]["lookup_source"] == "openalex"


class _FakeCrossref:
    """Answers ``/works/<doi>`` and ``/works?query.bibliographic=`` from a canned table."""

    def __init__(self, table):
        self.table = table  # doi -> (title, issn list, year)
        self.calls: list[dict] = []

    def get(self, url, params=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        if params and "query.bibliographic" in params:
            items = [self._item(d) for d in self.table]
            return _CrossrefResponse({"message": {"items": items}}, 200)
        doi = url.split("/works/", 1)[1]
        if doi in self.table:
            return _CrossrefResponse({"message": self._item(doi)}, 200)
        return _CrossrefResponse({"status": "error"}, 404)

    def _item(self, doi):
        title, issn, year = self.table[doi]
        return {"DOI": doi, "title": [title], "ISSN": list(issn),
                "issued": {"date-parts": [[year, 1]]}}


class _CrossrefResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {}
        self.request = None

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_crossref_lookup_by_doi_and_title(monkeypatch):
    import fetch_synergy as fs

    table = {
        "10.1/abc": ("PTSD trajectories", ["0027-3171", "1532-7906"], 2010),
        "10.9/other": ("Something else entirely", ["1111-2222"], 2012),
    }
    client = _FakeCrossref(table)
    monkeypatch.setattr(fs, "fetch_with_retries", lambda c, url, **kw: c.get(url, kw.get("params")))
    recs = parse_v1_csv(V1_CSV_WITH_IDS)
    hits = fs.hydrate_by_doi_crossref(recs, client, "x@example.org", fs.RateLimiter(1000))
    assert hits == 1
    assert recs[0]["issn"] == ["0027-3171", "1532-7906"] and recs[0]["match_method"] == "doi"
    assert recs[0]["lookup_source"] == "crossref" and recs[0]["openalex_id"] is None
    assert recs[2]["issn"] == [] and recs[2]["match_method"] is None  # 404 -> untouched
    assert all("mailto" in c["params"] for c in client.calls)
    # title: exact normalised match only
    recs = [
        {"record_id": 5, "title": "PTSD Trajectories.", "abstract": "", "doi": None, "issn": []},
        {"record_id": 6, "title": "A completely unknown title here", "abstract": "", "issn": []},
    ]
    hits = fs.hydrate_by_title_crossref(recs, client, "x@example.org", fs.RateLimiter(1000))
    assert hits == 1
    assert recs[0]["doi"] == "10.1/abc" and recs[0]["issn"] == ["0027-3171", "1532-7906"]
    assert recs[0]["match_method"] == "title_exact" and recs[0]["lookup_source"] == "crossref"
    assert recs[0]["year"] == 2010 and recs[1]["issn"] == []


def test_main_v1_crossref_lookup_targets_ta_included(tmp_path, monkeypatch):
    import fetch_synergy as fs

    index = tmp_path / "index.csv"
    index.write_text(
        "dataset_id,url,reference\nX_2017,https://example.org/X.csv,https://doi.org/10.1/r\n",
        encoding="utf-8",
    )
    cr = _FakeCrossref({"10.1/abc": ("PTSD trajectories", ["0027-3171"], 2010),
                        "10.2/xyz": ("Third", ["3333-4444"], 2011)})

    class _Text:
        text = V1_CSV_WITH_IDS

    def fake_fetch(client, url, **kw):
        if url == "https://example.org/X.csv":
            return _Text()
        assert "openalex" not in url
        return cr.get(url, kw.get("params"))

    monkeypatch.setattr(fs, "fetch_with_retries", fake_fetch)
    out_dir = tmp_path / "out"
    rc = fs.main(["--dataset", "X_2017", "--route", "v1", "--hydrate", "included",
                  "--lookup", "crossref", "--index", str(index), "--out-dir", str(out_dir),
                  "--email", "e@x.org"])
    assert rc == 0
    rows = [json.loads(line) for line in (out_dir / "X_2017.jsonl").read_text().splitlines()]
    # rows 1 and 3 carry a positive label (row 3 is TA-included only); row 2 is never looked up
    assert rows[0]["issn"] == ["0027-3171"] and rows[0]["lookup_source"] == "crossref"
    assert rows[2]["issn"] == ["3333-4444"] and rows[2]["lookup_source"] == "crossref"
    assert rows[1]["issn"] == [] and rows[1].get("lookup_source") is None
    assert not any("query.bibliographic" in c["params"] for c in cr.calls)  # DOIs sufficed
    info = json.loads((out_dir / "X_2017.fetch.json").read_text(encoding="utf-8"))
    assert info["hydrate"] == "included" and info["hydrate_lookup"] == "crossref"
    assert info["n_hydrate_targets"] == 2 and info["n_targets_with_issn"] == 2
    assert info["n_with_issn"] == 2


# ---------------------------------------------------------------- additional cases


def test_title_match_requires_year_agreement_when_the_record_has_a_year(monkeypatch):
    import fetch_synergy as fs

    client = _FakeCrossref({"10.1/abc": ("PTSD trajectories", ["0027-3171"], 2010)})
    monkeypatch.setattr(fs, "fetch_with_retries", lambda c, url, **kw: c.get(url, kw.get("params")))
    recs = [
        {"record_id": 1, "title": "PTSD Trajectories.", "abstract": "", "issn": [], "year": 1998},
        {"record_id": 2, "title": "PTSD Trajectories.", "abstract": "", "issn": [], "year": 2011},
        {"record_id": 3, "title": "PTSD Trajectories.", "abstract": "", "issn": [], "year": None},
    ]
    hits = fs.hydrate_by_title_crossref(recs, client, "x@example.org", fs.RateLimiter(1000))
    assert hits == 2
    assert recs[0]["issn"] == [] and recs[0].get("match_method") is None  # 1998 vs 2010
    assert recs[1]["match_method"] == "title_exact_year"  # within +-1 year
    assert recs[2]["match_method"] == "title_exact"  # no year to check


def test_title_match_skips_yearbook_reprints(monkeypatch):
    """Elsevier 'Yearbook of ...' series re-publish articles under identical titles (spot check
    2026-09-03: 3 of 40 sampled title matches); such hits must not attribute the venue."""
    import fetch_synergy as fs

    class _Client:
        def get(self, url, params=None):
            items = [
                {"DOI": "10.1016/j.ypsy.2011.07.049", "title": ["PTSD trajectories"],
                 "ISSN": ["0084-3970"], "issued": {"date-parts": [[2013]]},
                 "container-title": ["Yearbook of Psychiatry and Applied Mental Health"]},
                {"DOI": "10.1192/bjp.1", "title": ["PTSD trajectories"], "ISSN": ["0007-1250"],
                 "issued": {"date-parts": [[2012]]},
                 "container-title": ["British Journal of Psychiatry"]},
            ]
            return _FakeResponse({"message": {"items": items}})

    monkeypatch.setattr(
        fs, "fetch_with_retries", lambda c, url, **kw: c.get(url, kw.get("params"))
    )
    recs = [{"record_id": 1, "title": "PTSD trajectories", "abstract": "", "issn": [],
             "year": 2012}]
    assert fs.hydrate_by_title_crossref(recs, _Client(), None, fs.RateLimiter(1000)) == 1
    assert recs[0]["issn"] == ["0007-1250"] and recs[0]["doi"] == "10.1192/bjp.1"
    fields = fs.crossref_work_to_fields({"container-title": ["Yearbook of X"]})
    assert fields["container_title"] == "Yearbook of X"
    assert fs.is_reprint_venue("Yearbook of Psychiatry and Applied Mental Health")
    assert not fs.is_reprint_venue("Psychological Medicine")


# ---------------------------------------------------------------- additional cases


def test_is_valid_doi_rejects_malformed_export_values():
    import fetch_synergy as fs

    assert fs.is_valid_doi("10.1037/a0037593")
    assert fs.is_valid_doi("10.1016/j.pnpbp.2011.03.006")
    assert not fs.is_valid_doi("10.1037.a0037593")  # dot instead of slash
    assert not fs.is_valid_doi("ss1016/j.cpem.2014.09.002")
    assert not fs.is_valid_doi("10.1159/000152377; 10.1159/000152377")  # two DOIs in one cell
    assert not fs.is_valid_doi(None) and not fs.is_valid_doi("")



# ---------------------------------------------------------------- --route plus

PLUS_LABELS_CSV = (
    "openalex_id,doi,pmid,lens_id,label_included,label_abstract_included\n"
    "https://openalex.org/w1,https://doi.org/10.1/a,,,1,1\n"
    "https://openalex.org/w2,,,,0,0\n"
    "https://openalex.org/w3,,,,0,1\n"
)


def test_parse_plus_labels_maps_abstract_included_to_screening_and_numbers_by_row():
    import fetch_synergy as fs

    recs = fs.parse_plus_labels(PLUS_LABELS_CSV)
    assert [r["record_id"] for r in recs] == [1, 2, 3]
    assert recs[0]["openalex_id"] == "W1" and recs[0]["doi"] == "10.1/a"
    assert recs[0]["label_included"] == 1 and recs[0]["label_abstract_screening"] == 1
    assert recs[2]["label_included"] == 0 and recs[2]["label_abstract_screening"] == 1
    assert all(r["source_route"] == "plus" for r in recs)
    assert all(r["title"] == "" and r["abstract"] == "" and r["issn"] == [] for r in recs)
    assert recs[1]["doi"] is None


def _plus_work(oid, title, abstract="From plus", issn=("1111-2222",)):
    return {
        "id": f"https://openalex.org/{oid}",
        "doi": None,
        "title": title,
        "publication_year": 2020,
        "type": "article",
        "abstract_inverted_index": {w: [i] for i, w in enumerate(abstract.split())},
        "primary_location": {"source": {"issn_l": issn[0], "issn": list(issn)}},
    }


def _write_works_zip(path, chunks):
    import json
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        for i, chunk in enumerate(chunks):
            zf.writestr(f"works_{i}.json", json.dumps(chunk))


def test_load_plus_works_merges_every_json_entry_in_the_zip(tmp_path):
    import fetch_synergy as fs

    zpath = tmp_path / "works_1.zip"
    _write_works_zip(
        zpath,
        [[_plus_work("W1", "Title one")], [_plus_work("W2", "Title two", abstract="More text")]],
    )
    works = fs.load_plus_works(zpath)
    assert set(works) == {"W1", "W2"}
    assert works["W1"]["title"] == "Title one" and works["W2"]["abstract"] == "More text"


def test_apply_plus_works_fills_fields_and_counts_hits():
    import fetch_synergy as fs

    records = fs.parse_plus_labels(PLUS_LABELS_CSV)
    works = {
        "W1": {"title": "T1", "abstract": "A1", "issn": ["1111-2222"], "year": 2020, "doi": None},
        "W3": {"title": "T3", "abstract": "A3", "issn": [], "year": 2019, "doi": "10.9/z"},
    }
    hits = fs.apply_plus_works(records, works)
    assert hits == 2
    assert records[0]["title"] == "T1" and records[0]["match_method"] == "openalex_id"
    assert records[2]["doi"] == "10.9/z"  # filled because the record had none
    assert records[1]["title"] == ""  # W2 not in the works map: untouched


def test_plus_metadata_reads_local_cache_only(tmp_path):
    import json

    import fetch_synergy as fs

    cache = tmp_path / "cache" / "X_2021"
    cache.mkdir(parents=True)
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "publication": {"doi": "10.1/rev", "eligibility_criteria": "criteria text"},
                "data": {"n_records": 10, "n_records_included": 2},
            }
        ),
        encoding="utf-8",
    )
    meta = fs.plus_metadata(tmp_path / "cache", "X_2021")
    assert meta["publication_doi"] == "10.1/rev"
    assert meta["eligibility_criteria"] == "criteria text"
    assert meta["catalogue_n_records"] == 10 and meta["catalogue_n_records_included"] == 2


def test_main_route_plus_uses_cached_labels_and_works_archive(tmp_path, monkeypatch):
    import json

    import fetch_synergy as fs

    cache_dir = tmp_path / "cache"
    ds_dir = cache_dir / "Y_2020"
    ds_dir.mkdir(parents=True)
    (ds_dir / "metadata.json").write_text(
        json.dumps(
            {
                "publication": {"doi": "10.1/y", "eligibility_criteria": "crit"},
                "data": {"n_records": 3, "n_records_included": 1},
            }
        ),
        encoding="utf-8",
    )
    (ds_dir / "labels.csv").write_text(PLUS_LABELS_CSV, encoding="utf-8")
    _write_works_zip(
        ds_dir / "works_1.zip",
        [[_plus_work("W1", "T1", "abs one"), _plus_work("W2", "T2", "abs two")]],
    )

    def boom(*a, **k):
        raise AssertionError("no network call is allowed when labels.csv and works are cached")

    monkeypatch.setattr(fs, "fetch_with_retries", boom)
    out_dir = tmp_path / "out"
    rc = fs.main(
        ["--dataset", "Y_2020", "--route", "plus", "--cache-dir", str(cache_dir),
         "--out-dir", str(out_dir)]
    )
    assert rc == 0
    rows = [json.loads(x) for x in (out_dir / "Y_2020.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["title"] == "T1" and rows[0]["label_abstract_screening"] == 1
    assert rows[2]["title"] == "" and rows[2]["label_abstract_screening"] == 1  # W3: no work entry
    info = json.loads((out_dir / "Y_2020.fetch.json").read_text(encoding="utf-8"))
    assert info["route"] == "plus" and info["n"] == 3 and info["n_included"] == 1
    assert info["label_column_mapping"] == {"label_abstract_included": "label_abstract_screening"}
    assert info["works_source"] == "cache" and info["labels_source"] == "cache"
    assert info["catalogue_n_records"] == 3 and info["catalogue_agrees_n"] is True
    assert "raw.githubusercontent.com/asreview/synergy-dataset" in info["source_url"]


def test_main_route_plus_downloads_labels_and_hydrates_by_id_when_nothing_is_cached(
    tmp_path, monkeypatch
):
    import json

    import fetch_synergy as fs

    cache_dir = tmp_path / "cache"
    ds_dir = cache_dir / "Z_2022"
    ds_dir.mkdir(parents=True)
    (ds_dir / "metadata.json").write_text(
        json.dumps(
            {
                "publication": {"doi": "10.1/z", "eligibility_criteria": "crit"},
                "data": {"n_records": 999, "n_records_included": 999},
            }
        ),
        encoding="utf-8",
    )
    oa = _FakeClient([_work("W1", "10.1/a", "Hydrated", abstract="hydrated text")])

    class _Text:
        text = PLUS_LABELS_CSV

    def fake_fetch(client, url, **kw):
        if "raw.githubusercontent.com" in url:
            assert "Z_2022/labels.csv" in url
            return _Text()
        return oa.get(url, kw.get("params"))

    monkeypatch.setattr(fs, "fetch_with_retries", fake_fetch)
    out_dir = tmp_path / "out"
    rc = fs.main(
        ["--dataset", "Z_2022", "--route", "plus", "--cache-dir", str(cache_dir),
         "--out-dir", str(out_dir), "--email", "e@x.org"]
    )
    assert rc == 0
    rows = [json.loads(x) for x in (out_dir / "Z_2022.jsonl").read_text().splitlines()]
    assert rows[0]["title"] == "Hydrated" and rows[0]["match_method"] == "openalex_id"
    info = json.loads((out_dir / "Z_2022.fetch.json").read_text(encoding="utf-8"))
    assert info["labels_source"] == "download" and info["works_source"] == "openalex_id"
    assert info["catalogue_agrees_n"] is False  # 999 vs the real 3


def test_main_route_plus_limit_applies_after_parsing(tmp_path, monkeypatch):
    import json

    import fetch_synergy as fs

    cache_dir = tmp_path / "cache"
    ds_dir = cache_dir / "W_2020"
    ds_dir.mkdir(parents=True)
    (ds_dir / "metadata.json").write_text(
        json.dumps({"publication": {}, "data": {}}), encoding="utf-8"
    )
    (ds_dir / "labels.csv").write_text(PLUS_LABELS_CSV, encoding="utf-8")
    out_dir = tmp_path / "out"
    empty = _FakeClient([])
    monkeypatch.setattr(
        fs, "fetch_with_retries", lambda c, url, **kw: empty.get(url, kw.get("params"))
    )
    rc = fs.main(
        ["--dataset", "W_2020", "--route", "plus", "--cache-dir", str(cache_dir),
         "--out-dir", str(out_dir), "--limit", "2"]
    )
    assert rc == 0
    rows = [json.loads(x) for x in (out_dir / "W_2020.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    info = json.loads((out_dir / "W_2020.fetch.json").read_text(encoding="utf-8"))
    assert info["limit"] == 2 and info["n"] == 2


def test_title_match_prefers_crossref_doi_when_the_export_doi_is_malformed(monkeypatch):
    import fetch_synergy as fs

    client = _FakeCrossref({"10.1037/a0037593": ("PTSD trajectories", ["0021-843X"], 2014)})
    monkeypatch.setattr(fs, "fetch_with_retries", lambda c, url, **kw: c.get(url, kw.get("params")))
    recs = [
        {"record_id": 1, "title": "PTSD Trajectories.", "abstract": "", "issn": [], "year": 2014,
         "doi": "10.1037.a0037593"},
        {"record_id": 2, "title": "PTSD Trajectories.", "abstract": "", "issn": [], "year": 2014,
         "doi": "10.9999/valid-but-unknown"},
    ]
    assert fs.hydrate_by_title_crossref(recs, client, None, fs.RateLimiter(1000)) == 2
    assert recs[0]["doi"] == "10.1037/a0037593" and recs[0]["doi_export"] == "10.1037.a0037593"
    assert recs[1]["doi"] == "10.9999/valid-but-unknown" and "doi_export" not in recs[1]

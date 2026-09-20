"""Screening record: auditable per-paper trail of a Smart Search job.

Pure-function tests for ``app.services.screening_record`` plus the owner-only export
endpoint ``GET /api/v1/tasks/{task_id}/screening-record?format=json|csv``.
"""

from __future__ import annotations

import csv
import io
from uuid import UUID, uuid4

import pytest

from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.services.screening_record import (
    CSV_HEADER,
    build_screening_record,
    paper_record,
    screening_record_csv,
)

EXPECTED_HEADER = (
    "outcome,stage,status,criterion,quote,needs_review_reason,to_confirm,second_pass,"
    "second_pass_input_tokens,second_pass_output_tokens,second_pass_latency_s,reason,"
    "title,doi,year,journal,journal_issn,openalex_id,abstract"
)
BOM = "\ufeff"


def _job_result() -> dict:
    return {
        "papers": [
            {
                "title": "Included, with reason",
                "doi": "10.1/inc1",
                "year": 2021,
                "journal_name": "Journal A",
                "journal_issn": "1111-1111",
                "source_api": "openalex",
                "external_id": "W1",
                "wos_indexed": True,
                "abstract": "This paper studies teacher burnout in depth.",
                "screening_status": "INCLUDE",
                "screening_reason": "Directly studies the topic",
            },
            {
                "title": "Included, legacy (no reason)",
                "doi": None,
                "year": None,
                "journal_name": None,
                "journal_issn": None,
                "source_api": "openalex",
                "external_id": "W2",
            },
        ],
        "needs_review": [
            {
                "title": "Cannot decide from the shown text",
                "doi": "10.1/nr1",
                "year": 2022,
                "journal_name": "Journal F",
                "journal_issn": "6666-6666",
                "source_api": "openalex",
                "external_id": "W7",
                "screening_status": "NEEDS_REVIEW",
                "screening_criterion": "",
                "screening_quote": "",
                "screening_reason": "no abstract",
            },
        ],
        "excluded": [
            {
                "title": "Not in WoS",
                "doi": "10.1/ex1",
                "year": 2020,
                "journal": "Journal B",
                "journal_issn": "2222-2222",
                "openalex_id": "W3",
                "stage": "wos",
                "reason": "excluded at stage 1: journal not in the imported Web of Science list",
            },
            {
                "title": 'Off topic, "quoted", with, commas',
                "doi": "10.1/ex2",
                "year": 2019,
                "journal": "Journal C",
                "journal_issn": "3333-3333",
                "openalex_id": "W4",
                "abstract": "",
                "stage": "llm",
                "status": "EXCLUDE",
                "criterion": "E2",
                "quote": "shares a keyword",
                "needs_review_reason": "",
                "to_confirm": "",
                "second_pass": "",
                "second_pass_input_tokens": "",
                "second_pass_output_tokens": "",
                "second_pass_latency_s": "",
                "reason": "Different field, shares a keyword",
            },
            {
                "title": "=SUM(A1:A9) looks like a formula",
                "doi": "10.1/ex3",
                "year": 2018,
                "journal": "Journal E",
                "journal_issn": "5555-5555",
                "openalex_id": "W6",
                "stage": "llm",
                "status": "EXCLUDE",
                "criterion": "E1",
                "quote": "=not a real formula either",
                "reason": "-not about the topic",
            },
        ],
        "unscreened": [
            {
                "title": "Batch failed",
                "doi": "10.1/un1",
                "year": 2022,
                "journal": "Journal D",
                "journal_issn": "4444-4444",
                "openalex_id": "W5",
                "stage": "llm",
                "reason": "ScreeningError: TimeoutError: model timed out",
            }
        ],
        "criteria": {
            "query": "teacher burnout",
            "inclusion_criteria": [],
            "exclusion_criteria": [],
            "wos_filter": "auto",
            "inclusion_criteria_stages": [],
            "exclusion_criteria_stages": ["full_text"],
        },
        "flow": {
            "identified": 10,
            "duplicates_removed": 4,
            "stage1_screened": 5,
            "stage1_excluded": 1,
            "stage2_screened": 4,
            "stage2_excluded": 2,
            "needs_review": 1,
            "unscreened": 1,
            "included": 2,
            "rounds": 2,
            "stop_reason": "no_new_papers",
        },
        "provenance": {
            "screener": {"model_reported": ["deepseek-v4-flash"], "calls": 1},
            "query_generator": {"model_configured": "deepseek-chat"},
        },
        "total_included": 2,
        "total_scanned": 10,
        "rounds": 2,
        "elapsed_minutes": 0.5,
        "stop_reason": "no_new_papers",
        "stage1_applied": True,
    }


# --- paper_record ---------------------------------------------------------------------


def test_paper_record_maps_pipeline_paper_dicts_to_the_record_shape():
    row = paper_record(_job_result()["papers"][0], stage="llm", reason="why")
    assert row == {
        "title": "Included, with reason",
        "doi": "10.1/inc1",
        "year": 2021,
        "journal": "Journal A",
        "journal_issn": "1111-1111",
        "openalex_id": "W1",
        "abstract": "This paper studies teacher burnout in depth.",
        "stage": "llm",
        "status": "",
        "criterion": "",
        "quote": "",
        "needs_review_reason": "",
        "to_confirm": "",
        "second_pass": "",
        "second_pass_input_tokens": "",
        "second_pass_output_tokens": "",
        "second_pass_latency_s": "",
        "reason": "why",
    }


def test_paper_record_projects_status_criterion_and_quote_when_given():
    """``paper_record`` carries ``status``/``criterion``/``quote``
    passthroughs, the screener's protocol-eligibility decision for the record."""
    row = paper_record(
        _job_result()["papers"][0],
        stage="llm",
        reason="off topic",
        status="EXCLUDE",
        criterion="E3",
        quote="a verbatim phrase",
    )
    assert row["status"] == "EXCLUDE"
    assert row["criterion"] == "E3"
    assert row["quote"] == "a verbatim phrase"


def test_paper_record_accepts_an_existing_record_entry():
    """Round-trips an already-built EXCLUDE record entry (the shape ``excluded`` already
    carries), including its ``status``/``criterion``/``quote``."""
    entry = _job_result()["excluded"][1]
    row = paper_record(
        entry,
        stage=entry["stage"],
        reason=entry["reason"],
        status=entry["status"],
        criterion=entry["criterion"],
        quote=entry["quote"],
    )
    assert row == entry


def test_paper_record_only_uses_external_id_as_openalex_id_for_openalex_papers():
    row = paper_record({"title": "t", "source_api": "manual", "external_id": "x"}, stage="llm",
                       reason="")
    assert row["openalex_id"] is None


# --- build_screening_record ------------------------------------------------------------


def test_build_screening_record_lists_every_paper_once_with_its_outcome():
    record = build_screening_record(_job_result())

    assert set(record) == {"criteria", "flow", "provenance", "retrieval_failures", "records"}
    assert record["criteria"]["query"] == "teacher burnout"
    # The stage arrays -- the one input needed to
    # reproduce the routing -- pass through into the exported screening record's criteria
    # block unchanged.
    assert record["criteria"]["inclusion_criteria_stages"] == []
    assert record["criteria"]["exclusion_criteria_stages"] == ["full_text"]
    assert record["flow"]["identified"] == 10
    assert record["flow"]["needs_review"] == 1
    assert record["provenance"]["screener"]["calls"] == 1

    rows = record["records"]
    assert len(rows) == 7
    assert [r["outcome"] for r in rows] == [
        "included", "included", "needs_review",
        "excluded", "excluded", "excluded", "unscreened",
    ]
    included = rows[0]
    assert included["stage"] == "llm"
    assert included["status"] == "INCLUDE"
    assert included["reason"] == "Directly studies the topic"
    assert included["journal"] == "Journal A"
    assert included["openalex_id"] == "W1"
    assert included["abstract"] == "This paper studies teacher burnout in depth."
    assert rows[1]["reason"] == ""  # legacy paper without a stored reason
    assert rows[1]["status"] == ""
    assert rows[1]["abstract"] == ""  # legacy paper carries no abstract at all
    needs_review = rows[2]
    assert needs_review["stage"] == "llm"
    assert needs_review["status"] == "NEEDS_REVIEW"
    assert needs_review["reason"] == "no abstract"
    assert needs_review["doi"] == "10.1/nr1"
    assert rows[3]["stage"] == "wos"
    assert rows[3]["status"] == ""  # Stage 1 exclusions never reach the LLM screener
    assert rows[4]["status"] == "EXCLUDE"
    assert rows[4]["criterion"] == "E2"
    assert rows[4]["quote"] == "shares a keyword"
    # JSON keeps the raw text; only the CSV neutralises spreadsheet formula triggers.
    assert rows[5]["title"].startswith("=SUM")
    assert rows[5]["quote"].startswith("=not a real formula")
    assert rows[6]["reason"].startswith("ScreeningError:")
    for row in rows:
        assert set(row) == set(CSV_HEADER)


def test_build_screening_record_carries_the_retrieval_failures():
    result = _job_result()
    result["retrieval_failures"] = [("q1", "HTTPStatusError: 429 Too Many Requests")]
    record = build_screening_record(result)
    assert record["retrieval_failures"] == [["q1", "HTTPStatusError: 429 Too Many Requests"]]
    assert build_screening_record(_job_result())["retrieval_failures"] == []


def test_build_screening_record_passes_through_the_round_log_and_stop_rule_settings():
    """``provenance.rounds`` and ``provenance.settings`` are already part of the
    job result's ``provenance`` block written by ``run_smart_search``, so the wholesale
    ``provenance`` passthrough in ``build_screening_record`` carries both without any
    dedicated handling."""
    result = _job_result()
    result["provenance"]["rounds"] = [
        {
            "round": 1,
            "queries": [{"query": "teacher burnout", "returned": 4, "new_unique": 4}],
            "screened": 4,
            "included_new": 2,
            "needs_review_new": 1,
        }
    ]
    result["provenance"]["settings"] = {"min_rounds": 3, "dry_round_patience": 2}

    record = build_screening_record(result)

    assert record["provenance"]["rounds"] == result["provenance"]["rounds"]
    assert record["provenance"]["settings"] == {"min_rounds": 3, "dry_round_patience": 2}
    # The CSV export is unaffected: it is built from ``records``, not ``provenance``.
    assert screening_record_csv(result) == screening_record_csv(_job_result())


def test_build_screening_record_needs_review_never_appears_as_included_or_excluded():
    """NEEDS_REVIEW is a fourth outcome, routed neither into the
    library (``included``) nor into ``excluded``."""
    record = build_screening_record(_job_result())
    outcomes_by_doi = {r["doi"]: r["outcome"] for r in record["records"]}
    assert outcomes_by_doi["10.1/nr1"] == "needs_review"
    assert "10.1/nr1" not in {
        r["doi"] for r in record["records"] if r["outcome"] in ("included", "excluded")
    }


def test_build_screening_record_tolerates_a_v1_0_0_result():
    legacy = {"papers": [{"title": "Old", "doi": "10.1/old"}], "total_included": 1}
    record = build_screening_record(legacy)
    assert record["criteria"] == {}
    assert record["flow"] == {}
    assert record["provenance"] == {}
    assert len(record["records"]) == 1
    assert record["records"][0]["outcome"] == "included"
    assert record["records"][0]["reason"] == ""


# --- screening_record_csv --------------------------------------------------------------


def test_csv_has_the_exact_header_and_one_row_per_record():
    text = screening_record_csv(_job_result())
    # UTF-8 BOM so Excel on Windows decodes non-ASCII titles/reasons correctly.
    assert text.startswith(BOM)
    lines = text.lstrip(BOM).splitlines()
    assert lines[0] == EXPECTED_HEADER
    assert ",".join(CSV_HEADER) == EXPECTED_HEADER
    assert len(lines) == 1 + 7

    parsed = list(csv.DictReader(io.StringIO(text.lstrip(BOM))))
    assert [r["outcome"] for r in parsed] == [
        "included", "included", "needs_review",
        "excluded", "excluded", "excluded", "unscreened",
    ]
    assert parsed[2]["status"] == "NEEDS_REVIEW"
    # Commas and quotes in fields round-trip.
    assert parsed[4]["title"] == 'Off topic, "quoted", with, commas'
    assert parsed[4]["reason"] == "Different field, shares a keyword"
    assert parsed[4]["criterion"] == "E2"
    # None becomes an empty cell, integers are written as text.
    assert parsed[1]["doi"] == ""
    assert parsed[0]["year"] == "2021"
    # abstract is the last column and round-trips through the CSV (the workbook exporter
    # needs the abstract text to judge NEEDS_REVIEW rows).
    assert parsed[0]["abstract"] == "This paper studies teacher burnout in depth."
    assert parsed[1]["abstract"] == ""


def test_csv_neutralises_spreadsheet_formula_triggers():
    """Untrusted OpenAlex metadata and model text must not execute as formulas (CSV
    injection); the verbatim ``quote`` column is untrusted model output too."""
    text = screening_record_csv(_job_result()).lstrip(BOM)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert parsed[5]["title"] == "'=SUM(A1:A9) looks like a formula"
    assert parsed[5]["reason"] == "'-not about the topic"
    assert parsed[5]["quote"] == "'=not a real formula either"
    # Ordinary cells are untouched.
    assert parsed[0]["title"] == "Included, with reason"

    for trigger in ("=", "+", "-", "@", "\t", "\r"):
        row = screening_record_csv({"papers": [{"title": f"{trigger}x"}]}).lstrip(BOM)
        assert list(csv.DictReader(io.StringIO(row)))[0]["title"] == f"'{trigger}x"


# ---------------------------------------------------------------- needs_review_reason


def test_csv_header_has_nineteen_columns():
    """CSV_HEADER carries ``second_pass`` after ``to_confirm``, plus
    ``second_pass_input_tokens``/``second_pass_output_tokens``/
    ``second_pass_latency_s`` right after it, for nineteen columns total. The demo's own
    frozen ``demo/expected/screening_record.csv`` and its independently duplicated fifteen-column
    tuple in ``demo/tests/test_expected_baseline.py`` read only that golden file, never
    ``CSV_HEADER``, so neither is affected by this module's own column count growing."""
    assert len(CSV_HEADER) == 19
    assert CSV_HEADER.index("needs_review_reason") == CSV_HEADER.index("quote") + 1
    assert CSV_HEADER.index("to_confirm") == CSV_HEADER.index("needs_review_reason") + 1
    assert CSV_HEADER.index("second_pass") == CSV_HEADER.index("to_confirm") + 1
    assert CSV_HEADER.index("second_pass_input_tokens") == CSV_HEADER.index("second_pass") + 1
    assert (
        CSV_HEADER.index("second_pass_output_tokens")
        == CSV_HEADER.index("second_pass_input_tokens") + 1
    )
    assert (
        CSV_HEADER.index("second_pass_latency_s")
        == CSV_HEADER.index("second_pass_output_tokens") + 1
    )
    # abstract is appended last so field order stays stable for existing readers.
    assert CSV_HEADER[-1] == "abstract"


def test_needs_review_row_carries_the_guard_reason_in_the_reason_cell():
    result = _job_result()
    result["needs_review"][0]["screening_guard_reason"] = "full_text_criterion"
    record = build_screening_record(result)
    row = next(r for r in record["records"] if r["outcome"] == "needs_review")
    assert row["needs_review_reason"] == "full_text_criterion"


def test_needs_review_row_falls_back_to_no_abstract_or_undecidable():
    result = _job_result()
    result["needs_review"][0]["abstract"] = ""
    record = build_screening_record(result)
    row = next(r for r in record["records"] if r["outcome"] == "needs_review")
    assert row["needs_review_reason"] == "no_abstract"

    result = _job_result()
    result["needs_review"][0]["abstract"] = "some text"
    record = build_screening_record(result)
    row = next(r for r in record["records"] if r["outcome"] == "needs_review")
    assert row["needs_review_reason"] == "undecidable"


def test_included_and_excluded_rows_carry_an_empty_needs_review_reason():
    record = build_screening_record(_job_result())
    for row in record["records"]:
        if row["outcome"] != "needs_review":
            assert row["needs_review_reason"] == ""


# ---------------------------------------------------------------- to_confirm


def test_included_row_comma_joins_screening_to_confirm():
    result = _job_result()
    result["papers"][0]["screening_to_confirm"] = ["I1", "I3"]
    record = build_screening_record(result)
    row = record["records"][0]
    assert row["outcome"] == "included"
    assert row["to_confirm"] == "I1,I3"


def test_non_included_rows_carry_an_empty_to_confirm_cell():
    record = build_screening_record(_job_result())
    for row in record["records"]:
        if row["outcome"] not in ("included", "needs_review"):
            assert row["to_confirm"] == ""
    # the second included paper (legacy, no screening_to_confirm at all) is empty too
    assert record["records"][1]["outcome"] == "included"
    assert record["records"][1]["to_confirm"] == ""


def test_needs_review_row_carries_screening_to_confirm_too():
    """A record demoted for an unconfirmed full-text inclusion criterion (guard
    reason "full_text_to_confirm") is routed to needs_review
    rather than shipped as included, so its still-open criterion ids must surface there
    too, not only on an included row."""
    result = _job_result()
    result["needs_review"][0]["screening_guard_reason"] = "full_text_to_confirm"
    result["needs_review"][0]["screening_to_confirm"] = ["I1"]
    record = build_screening_record(result)
    row = next(r for r in record["records"] if r["outcome"] == "needs_review")
    assert row["needs_review_reason"] == "full_text_to_confirm"
    assert row["to_confirm"] == "I1"


# ---------------------------------------------------------------- second_pass


def test_included_row_summarises_the_second_pass_answer():
    result = _job_result()
    result["papers"][0]["screening_second_pass"] = {
        "population": "established",
        "outcome": "established",
        "study_type": "study",
    }
    record = build_screening_record(result)
    row = record["records"][0]
    assert row["outcome"] == "included"
    assert row["second_pass"] == "population=established;outcome=established;study_type=study"


def test_needs_review_row_summarises_a_demoting_second_pass_answer():
    result = _job_result()
    result["needs_review"][0]["screening_guard_reason"] = "not_established"
    result["needs_review"][0]["screening_second_pass"] = {
        "population": "established",
        "outcome": "not_established",
        "study_type": "study",
    }
    record = build_screening_record(result)
    row = next(r for r in record["records"] if r["outcome"] == "needs_review")
    assert row["second_pass"] == (
        "population=established;outcome=not_established;study_type=study"
    )


def test_rows_carry_an_empty_second_pass_cell_when_it_never_ran():
    record = build_screening_record(_job_result())
    for row in record["records"]:
        assert row["second_pass"] == ""


def test_csv_of_an_empty_result_is_just_the_header():
    text = screening_record_csv({})
    assert text.startswith(BOM)
    assert text.lstrip(BOM).splitlines() == [EXPECTED_HEADER]


# --- GET /tasks/{task_id}/screening-record ---------------------------------------------


async def _register(client, email: str) -> str:
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Record User",
            "expertise_level": "researcher",
        },
    )
    assert res.status_code == 201, res.text
    return res.json()["access_token"]


async def _seed_job(
    client,
    db_session,
    token: str,
    *,
    job_type: JobType = JobType.smart_search,
    result: dict | None = None,
) -> UUID:
    headers = {"Authorization": f"Bearer {token}"}
    me = await client.get("/api/v1/auth/me", headers=headers)
    user_id = UUID(me.json()["id"])
    proj = await client.post("/api/v1/projects", json={"title": "Record Project"}, headers=headers)
    project_id = UUID(proj.json()["id"])
    job_id = uuid4()
    db_session.add(
        AnalysisJob(
            id=job_id,
            project_id=project_id,
            user_id=user_id,
            job_type=job_type,
            status=JobStatus.completed,
            progress=1.0,
            result=result,
        )
    )
    await db_session.commit()
    return job_id


@pytest.mark.asyncio
async def test_screening_record_json_returns_the_built_record(client, db_session):
    token = await _register(client, "record-json@example.com")
    job_id = await _seed_job(client, db_session, token, result=_job_result())

    res = await client.get(
        f"/api/v1/tasks/{job_id}/screening-record",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 200, res.text
    body = res.json()
    assert body == build_screening_record(_job_result())
    assert len(body["records"]) == 7


@pytest.mark.asyncio
async def test_screening_record_csv_is_an_attachment(client, db_session):
    token = await _register(client, "record-csv@example.com")
    job_id = await _seed_job(client, db_session, token, result=_job_result())

    res = await client.get(
        f"/api/v1/tasks/{job_id}/screening-record",
        params={"format": "csv"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("text/csv")
    assert res.headers["content-disposition"] == (
        f'attachment; filename="screening-record-{job_id}.csv"'
    )
    assert res.content.startswith(BOM.encode("utf-8"))
    assert res.text == screening_record_csv(_job_result())
    assert res.text.lstrip(BOM).splitlines()[0] == EXPECTED_HEADER


@pytest.mark.asyncio
async def test_screening_record_rejects_unknown_format(client, db_session):
    token = await _register(client, "record-fmt@example.com")
    job_id = await _seed_job(client, db_session, token, result=_job_result())

    res = await client.get(
        f"/api/v1/tasks/{job_id}/screening-record",
        params={"format": "xlsx"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_screening_record_is_owner_only(client, db_session):
    owner = await _register(client, "record-owner@example.com")
    other = await _register(client, "record-other@example.com")
    job_id = await _seed_job(client, db_session, owner, result=_job_result())

    res = await client.get(
        f"/api/v1/tasks/{job_id}/screening-record",
        headers={"Authorization": f"Bearer {other}"},
    )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_screening_record_404s_for_other_job_types_and_missing_results(
    client, db_session
):
    token = await _register(client, "record-404@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    not_smart = await _seed_job(
        client, db_session, token, job_type=JobType.deep_search, result={"papers": []}
    )
    res = await client.get(f"/api/v1/tasks/{not_smart}/screening-record", headers=headers)
    assert res.status_code == 404

    no_result = await _seed_job(client, db_session, token, result=None)
    res = await client.get(f"/api/v1/tasks/{no_result}/screening-record", headers=headers)
    assert res.status_code == 404

    res = await client.get(f"/api/v1/tasks/{uuid4()}/screening-record", headers=headers)
    assert res.status_code == 404

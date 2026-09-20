"""Claim-verification record: auditable per-claim trace of a claim-verify job.

Pure-function tests for ``app.services.claim_record`` plus the owner-only export endpoint
``GET /api/v1/tasks/{task_id}/claim-record?format=json|csv``.
"""

from __future__ import annotations

import csv
import io
from uuid import UUID, uuid4

import pytest

from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.services.claim_record import CSV_HEADER, build_claim_record, claim_record_csv

EXPECTED_HEADER = (
    "status,model_status,machine_reasons,claim,claim_sentence,citation,doi,title,"
    "acquisition_route,evidence_quote,evidence_location,model_reported,diagnostics,"
    "unstated_details,evidence_quotes,n_assertions,central_assertion,pass_count,"
    "first_pass_machine_reasons"
)
BOM = "﻿"

#: All ten ``ClaimAssertion`` fields (backend/app/schemas/fulltext.py), the shape
#: ``_assertion_record`` normalises every exported assertion to.
_ASSERTION_FIELDS = frozenset(
    {
        "text",
        "kind",
        "verdict",
        "claim_value",
        "source_value",
        "quote",
        "central",
        "entity",
        "measure",
        "quotes",
    }
)


def _job_result() -> dict:
    return {
        "draft_id": "11111111-1111-1111-1111-111111111111",
        "verifications": [
            {
                "claim_text": "Test scores improved by 12% (Smith, 2020).",
                # claim_sentence is the full sentence the claim was cut from,
                # deliberately different from claim_text here to exercise the
                # narrowed-by-a-proposition case.
                "claim_sentence": (
                    "According to the study, test scores improved by 12% (Smith, 2020)."
                ),
                "paper_id": "22222222-2222-2222-2222-222222222222",
                "status": "verified",
                "model_status": "verified",
                "machine_reasons": [],
                "evidence_quote": "scores rose by 12 percent",
                "explanation": "Matches the reported effect size.",
                "suggested_revision": None,
                "model_reported": "deepseek-v4-flash",
                "citation": "(Smith, 2020)",
                "paper_doi": "10.1/verified",
                "paper_title": "A Verified Paper",
                "acquisition_route": "unpaywall",
                "evidence_location": "chunk 1 (results)",
            },
            {
                "claim_text": "Effects were dramatic (Lee, 2019).",
                "paper_id": "33333333-3333-3333-3333-333333333333",
                "status": "needs_nuance",
                # A guard demoted the model's own "verified" answer: the export
                # must be able to show that, not just the final status.
                "model_status": "verified",
                "machine_reasons": ["quote_not_verbatim"],
                "evidence_quote": "a modest, non-significant increase",
                "explanation": "The claim overstates a modest, non-significant finding.",
                "suggested_revision": "Effects were modest (Lee, 2019).",
                "model_reported": "deepseek-v4-flash",
                "citation": "(Lee, 2019)",
                "paper_doi": "10.1/nuance",
                "paper_title": 'Off, "topic", title\nSecond line',
                "acquisition_route": "stored_url",
                "evidence_location": "chunk 2 (discussion)",
            },
            {
                "claim_text": "Method X eliminates errors (Chen, 2018).",
                "paper_id": "44444444-4444-4444-4444-444444444444",
                "status": "unsupported",
                # Two guards fired: the CSV cell must join them, not show a Python repr.
                "model_status": "verified",
                "machine_reasons": ["numeric_not_in_source", "quote_not_verbatim"],
                "evidence_quote": None,
                "explanation": "The text never discusses eliminating errors.\nSee section 3.",
                "suggested_revision": None,
                "model_reported": "deepseek-v4-flash",
                "citation": "(Chen, 2018)",
                "paper_doi": "10.1/unsupported",
                "paper_title": "=SUM(A1:A9) looks like a formula",
                "acquisition_route": "metadata_oa_url",
                "evidence_location": None,
            },
            {
                "claim_text": "Widely cited work shows Y (Park, 2021).",
                "paper_id": "00000000-0000-0000-0000-000000000000",
                "status": "no_full_text",
                # Deterministic zero-chunk path: no model call, so no model_status.
                "model_status": None,
                "machine_reasons": [],
                "evidence_quote": None,
                "explanation": "Could not match citation to a paper in the library.",
                "suggested_revision": None,
                "model_reported": None,
                "citation": "(Park, 2021)",
                "paper_doi": None,
                "paper_title": None,
                "acquisition_route": None,
                "evidence_location": None,
            },
            {
                "claim_text": "Numbers, with commas (Kim, 2017).",
                "paper_id": "55555555-5555-5555-5555-555555555555",
                "status": "error",
                "model_status": None,
                "machine_reasons": [],
                "evidence_quote": None,
                "explanation": "Verification agent error: boom",
                "suggested_revision": None,
                "model_reported": None,
                "citation": "(Kim, 2017)",
                "paper_doi": "10.1/error",
                "paper_title": 'Comma, "quoted" title',
                "acquisition_route": "unpaywall",
                "evidence_location": None,
            },
        ],
        "verified_count": 1,
        "unsupported_count": 1,
        "nuance_count": 1,
        "abstract_only_count": 1,
        "error_count": 1,
        "needs_rewrite": 1,
        "needs_removal": 1,
        "provenance": {
            "agent": "claim_verification",
            "model_reported": ["deepseek-v4-flash"],
            "calls": 4,
        },
        "full_text_coverage": 0.8,
        "verified": 1,
    }


def _v3_job_result() -> dict:
    """A job result with diagnostics, unstated_details, evidence_quotes and
    assertions present. One entry also carries an older-shape assertion (only the first
    six ``ClaimAssertion`` fields) to check that the export fills in the rest."""
    return {
        "verifications": [
            {
                "claim_text": "Test scores improved by 12% for all learners (Smith, 2020).",
                "paper_id": "22222222-2222-2222-2222-222222222222",
                "status": "needs_nuance",
                "model_status": "needs_nuance",
                "machine_reasons": ["assertion_status_inconsistent"],
                "diagnostics": ["centrality_unmarked", "quote_relocated"],
                "unstated_details": ["the claim does not state the cohort size"],
                "evidence_quotes": ["scores rose by 12 percent", "across all learners"],
                "evidence_quote": "scores rose by 12 percent",
                "quote_relocations": [
                    {
                        "type": "quote_relocated",
                        "original": "scores rose by 12 percent overall we found",
                        "replacement": "scores rose by 12 percent overall we noted",
                        "edit_distance": 1,
                        "chunk_index": 0,
                    }
                ],
                "assertions": [
                    {
                        "text": "Test scores improved by 12%",
                        "kind": "quantity",
                        "verdict": "supported",
                        "claim_value": "12%",
                        "source_value": "12 percent",
                        "quote": "scores rose by 12 percent",
                        "central": True,
                        "entity": "test scores",
                        "measure": "percent change",
                        "quotes": ["scores rose by 12 percent"],
                    },
                    {
                        "text": "for all learners",
                        "kind": "scope",
                        "verdict": "supported",
                        "central": True,  # deliberately also True: several central=True
                    },
                ],
                "explanation": "Matches the reported effect size.",
                "suggested_revision": None,
                "model_reported": "deepseek-v4-flash",
                "citation": "(Smith, 2020)",
                "paper_doi": "10.1/verified",
                "paper_title": "A Verified Paper",
                "acquisition_route": "unpaywall",
                "evidence_location": "chunk 1 (results)",
            },
            {
                "claim_text": "=SUM(1,1) learners improved (Lee, 2019).",
                "paper_id": "33333333-3333-3333-3333-333333333333",
                "status": "verified",
                "model_status": "verified",
                "machine_reasons": [],
                "diagnostics": [],
                "unstated_details": [],
                "evidence_quotes": ["one span"],
                "evidence_quote": "one span",
                "assertions": [
                    {
                        "text": "=SUM(1,1) learners improved",
                        "kind": "quantity",
                        "verdict": "supported",
                        "central": True,
                    },
                ],
                "explanation": "Matches.",
                "suggested_revision": None,
                "model_reported": "deepseek-v4-flash",
                "citation": "(Lee, 2019)",
                "paper_doi": "10.1/nuance",
                "paper_title": "Some Paper",
                "acquisition_route": "stored_url",
                "evidence_location": "chunk 2 (discussion)",
            },
        ],
        "provenance": {"agent": "claim_verification", "calls": 2},
        "full_text_coverage": 1.0,
    }


# --- build_claim_record ----------------------------------------------------------------


def test_build_claim_record_returns_the_expected_shape():
    record = build_claim_record(_job_result())

    assert set(record) == {"provenance", "full_text_coverage", "counts", "records"}
    assert record["provenance"]["agent"] == "claim_verification"
    assert record["full_text_coverage"] == 0.8
    assert record["counts"] == {
        "total": 5,
        "verified": 1,
        "needs_nuance": 1,
        "unsupported": 1,
        "no_full_text": 1,
        "error": 1,
    }

    rows = record["records"]
    assert len(rows) == 5
    assert rows[0] == {
        "index": 1,
        "claim": "Test scores improved by 12% (Smith, 2020).",
        "claim_sentence": "According to the study, test scores improved by 12% (Smith, 2020).",
        "citation": "(Smith, 2020)",
        "paper_id": "22222222-2222-2222-2222-222222222222",
        "doi": "10.1/verified",
        "title": "A Verified Paper",
        "acquisition_route": "unpaywall",
        "status": "verified",
        "model_status": "verified",
        "machine_reasons": [],
        # This fixture predates diagnostics, unstated_details, evidence_quotes and
        # assertions, so all four default to empty.
        "diagnostics": [],
        "unstated_details": [],
        "evidence_quotes": [],
        # This fixture predates quote relocation too.
        "quote_relocations": [],
        "assertions": [],
        "evidence_quote": "scores rose by 12 percent",
        "evidence_location": "chunk 1 (results)",
        "explanation": "Matches the reported effect size.",
        "model_reported": "deepseek-v4-flash",
        # This fixture predates the bounded second pass, so all three default to
        # empty/zero.
        "passes": [],
        "pass_count": 0,
        "first_pass_machine_reasons": [],
    }
    assert rows[1]["index"] == 2
    assert rows[3]["doi"] is None  # unmatched citation: no paper to link to
    assert rows[3]["paper_id"] == "00000000-0000-0000-0000-000000000000"
    # model_status/machine_reasons surface a guard-driven demotion.
    assert rows[1]["model_status"] == "verified"
    assert rows[1]["status"] == "needs_nuance"
    assert rows[1]["machine_reasons"] == ["quote_not_verbatim"]
    assert rows[2]["machine_reasons"] == ["numeric_not_in_source", "quote_not_verbatim"]
    assert rows[3]["model_status"] is None
    assert rows[3]["machine_reasons"] == []
    # claim_sentence is exported next to claim, and defaults to None for a
    # verification that never carried one (an older-shape record, or one where the claim
    # and the sentence were always identical and the field was simply never set).
    assert rows[0]["claim_sentence"] == (
        "According to the study, test scores improved by 12% (Smith, 2020)."
    )
    assert rows[1]["claim_sentence"] is None


def test_build_claim_record_tolerates_an_empty_or_missing_result():
    for result in ({}, None, {"verifications": []}):
        record = build_claim_record(result)
        assert record["provenance"] == {}
        assert record["full_text_coverage"] == 0.0
        assert record["counts"] == {
            "total": 0, "verified": 0, "needs_nuance": 0, "unsupported": 0,
            "no_full_text": 0, "error": 0,
        }
        assert record["records"] == []


def test_build_claim_record_on_an_old_shape_result_yields_empty_v3_fields_without_error():
    """A job result recorded before diagnostics,
    unstated_details, evidence_quotes and assertions existed must still export cleanly,
    with the new fields defaulting to empty rather than raising."""
    rows = build_claim_record(_job_result())["records"]
    assert len(rows) == 5
    for row in rows:
        assert row["diagnostics"] == []
        assert row["unstated_details"] == []
        assert row["evidence_quotes"] == []
        assert row["quote_relocations"] == []
        assert row["assertions"] == []


def test_build_claim_record_carries_diagnostics_unstated_details_and_evidence_quotes():
    rows = build_claim_record(_v3_job_result())["records"]
    assert rows[0]["diagnostics"] == ["centrality_unmarked", "quote_relocated"]
    assert rows[0]["unstated_details"] == ["the claim does not state the cohort size"]
    assert rows[0]["evidence_quotes"] == ["scores rose by 12 percent", "across all learners"]
    # machine_reasons and model_status are untouched by the new fields, not duplicated.
    assert rows[0]["machine_reasons"] == ["assertion_status_inconsistent"]
    assert rows[0]["model_status"] == "needs_nuance"
    assert rows[0]["evidence_quote"] == "scores rose by 12 percent"


def test_build_claim_record_carries_quote_relocations_for_the_human_reviewer():
    """The JSON export must carry the original model
    quote, the replacement, the edit distance and the chunk index for every relocation, not
    just the "quote_relocated" diagnostics slug -- the one artefact a human countersigns
    must show the edit it is being asked to sign off on."""
    rows = build_claim_record(_v3_job_result())["records"]
    assert rows[0]["quote_relocations"] == [
        {
            "type": "quote_relocated",
            "original": "scores rose by 12 percent overall we found",
            "replacement": "scores rose by 12 percent overall we noted",
            "edit_distance": 1,
            "chunk_index": 0,
        }
    ]
    # A verification with no relocations exports an empty list, not a missing key.
    assert rows[1]["quote_relocations"] == []


def test_build_claim_record_carries_the_pass_count_and_first_pass_reasons():
    """For the bounded second pass, the export must
    show a first-pass demotion beside the final ``status`` -- ``pass_count`` and
    ``first_pass_machine_reasons`` -- without a reader needing to inspect the full
    ``passes`` list."""
    result = {
        "verifications": [
            {
                "claim_text": "x",
                "status": "verified",
                "model_status": "verified",
                "machine_reasons": [],
                "passes": [
                    {
                        "model_status": "verified",
                        "quotes": ["a bad quote"],
                        "machine_reasons": ["quote_not_verbatim"],
                        "diagnostics": [],
                        "tokens": {"input_tokens": 100, "output_tokens": 20},
                        "fingerprint": "fp-1",
                    },
                    {
                        "model_status": "verified",
                        "quotes": ["the correct quote"],
                        "machine_reasons": [],
                        "diagnostics": [],
                        "tokens": {"input_tokens": 110, "output_tokens": 25},
                        "fingerprint": "fp-2",
                    },
                ],
            },
            {"claim_text": "y", "status": "verified"},
        ]
    }
    rows = build_claim_record(result)["records"]
    assert rows[0]["pass_count"] == 2
    assert rows[0]["first_pass_machine_reasons"] == ["quote_not_verbatim"]
    assert len(rows[0]["passes"]) == 2
    # A verification with no `passes` recorded at all (an older-shape result, or a
    # deterministic no_full_text path that never ran the policy) reports pass_count 0 and no
    # first-pass demotion, not a missing key.
    assert rows[1]["pass_count"] == 0
    assert rows[1]["first_pass_machine_reasons"] == []
    assert rows[1]["passes"] == []


def test_build_claim_record_assertions_carry_all_ten_fields_with_defaults_filled():
    rows = build_claim_record(_v3_job_result())["records"]
    assertions = rows[0]["assertions"]
    assert len(assertions) == 2
    for assertion in assertions:
        assert set(assertion) == _ASSERTION_FIELDS

    full = assertions[0]
    assert full["text"] == "Test scores improved by 12%"
    assert full["kind"] == "quantity"
    assert full["verdict"] == "supported"
    assert full["claim_value"] == "12%"
    assert full["source_value"] == "12 percent"
    assert full["quote"] == "scores rose by 12 percent"
    assert full["central"] is True
    assert full["entity"] == "test scores"
    assert full["measure"] == "percent change"
    assert full["quotes"] == ["scores rose by 12 percent"]

    # The second assertion in the fixture only supplies text/kind/verdict/central; the
    # remaining six fields must default exactly as ClaimAssertion itself defaults them.
    partial = assertions[1]
    assert partial["text"] == "for all learners"
    assert partial["kind"] == "scope"
    assert partial["verdict"] == "supported"
    assert partial["central"] is True
    assert partial["claim_value"] is None
    assert partial["source_value"] is None
    assert partial["quote"] is None
    assert partial["entity"] is None
    assert partial["measure"] is None
    assert partial["quotes"] == []


# --- claim_record_csv --------------------------------------------------------------------


def test_csv_has_the_exact_header_and_one_row_per_claim():
    text = claim_record_csv(_job_result())
    assert text.startswith(BOM)
    body = text.lstrip(BOM)
    assert body.split("\n", 1)[0] == EXPECTED_HEADER
    assert ",".join(CSV_HEADER) == EXPECTED_HEADER

    parsed = list(csv.DictReader(io.StringIO(body)))
    assert len(parsed) == 5  # one row per claim, even though one title has an embedded newline
    assert [r["status"] for r in parsed] == [
        "verified", "needs_nuance", "unsupported", "no_full_text", "error",
    ]
    assert set(parsed[0]) == set(CSV_HEADER)


def test_csv_shows_the_model_status_and_joins_machine_reasons_with_a_semicolon():
    """The export must be able to show that a code
    guard, not the model, produced the final status -- and multiple fired guard slugs must
    render as a readable joined string, not a Python list repr."""
    text = claim_record_csv(_job_result()).lstrip(BOM)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert parsed[0]["model_status"] == "verified"
    assert parsed[0]["machine_reasons"] == ""
    assert parsed[1]["status"] == "needs_nuance"
    assert parsed[1]["model_status"] == "verified"
    assert parsed[1]["machine_reasons"] == "quote_not_verbatim"
    assert parsed[2]["machine_reasons"] == "numeric_not_in_source; quote_not_verbatim"
    assert parsed[3]["model_status"] == ""  # None -> empty cell, like every other field


def test_csv_quotes_commas_quotes_and_newlines_correctly():
    text = claim_record_csv(_job_result()).lstrip(BOM)
    parsed = list(csv.DictReader(io.StringIO(text)))
    # Commas and embedded quotes in the claim/title round-trip exactly.
    assert parsed[1]["title"] == 'Off, "topic", title\nSecond line'
    assert parsed[4]["title"] == 'Comma, "quoted" title'
    assert parsed[4]["claim"] == "Numbers, with commas (Kim, 2017)."
    # None becomes an empty cell.
    assert parsed[3]["doi"] == ""
    assert parsed[3]["model_reported"] == ""


def test_csv_carries_claim_sentence_next_to_claim_and_blank_when_absent():
    """claim_sentence round-trips through the CSV export, next to claim,
    and is an empty cell (like every other None field) when a verification never carried
    one."""
    text = claim_record_csv(_job_result()).lstrip(BOM)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert parsed[0]["claim"] == "Test scores improved by 12% (Smith, 2020)."
    assert parsed[0]["claim_sentence"] == (
        "According to the study, test scores improved by 12% (Smith, 2020)."
    )
    assert parsed[1]["claim_sentence"] == ""


def test_csv_neutralises_spreadsheet_formula_triggers():
    """Untrusted paper titles and model text must not execute as formulas (CSV injection)."""
    text = claim_record_csv(_job_result()).lstrip(BOM)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert parsed[2]["title"] == "'=SUM(A1:A9) looks like a formula"

    for trigger in ("=", "+", "-", "@", "\t", "\r"):
        result = {"verifications": [{"claim_text": f"{trigger}x", "status": "verified"}]}
        row = claim_record_csv(result).lstrip(BOM)
        assert list(csv.DictReader(io.StringIO(row)))[0]["claim"] == f"'{trigger}x"


def test_csv_header_gains_exactly_five_columns_appended_after_model_reported():
    """CSV_HEADER is the sixteen columns in order,
    plus a seventeenth column, claim_sentence, inserted right after claim, plus an
    eighteenth and nineteenth column, pass_count and
    first_pass_machine_reasons, appended last."""
    assert len(CSV_HEADER) == 19
    assert CSV_HEADER[:12] == (
        "status", "model_status", "machine_reasons", "claim", "claim_sentence", "citation",
        "doi", "title", "acquisition_route", "evidence_quote", "evidence_location",
        "model_reported",
    )
    assert CSV_HEADER[12:] == (
        "diagnostics", "unstated_details", "evidence_quotes", "n_assertions",
        "central_assertion", "pass_count", "first_pass_machine_reasons",
    )
    # No CSV flattening of assertions or passes: the variable-length lists are JSON-only.
    assert "assertions" not in CSV_HEADER
    assert "passes" not in CSV_HEADER


def test_csv_of_an_old_shape_result_leaves_the_five_new_columns_empty_or_zero():
    text = claim_record_csv(_job_result()).lstrip(BOM)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert len(parsed) == 5
    for row in parsed:
        assert row["diagnostics"] == ""
        assert row["unstated_details"] == ""
        assert row["evidence_quotes"] == ""
        assert row["n_assertions"] == "0"
        assert row["central_assertion"] == ""
        assert row["pass_count"] == "0"
        assert row["first_pass_machine_reasons"] == ""


def test_csv_joins_diagnostics_unstated_details_and_evidence_quotes_with_a_semicolon():
    text = claim_record_csv(_v3_job_result()).lstrip(BOM)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert parsed[0]["diagnostics"] == "centrality_unmarked; quote_relocated"
    assert parsed[0]["unstated_details"] == "the claim does not state the cohort size"
    assert parsed[0]["evidence_quotes"] == "scores rose by 12 percent; across all learners"


def test_csv_n_assertions_counts_and_central_assertion_is_empty_for_several_central():
    """Row 0's fixture deliberately marks two assertions central=True: the same "several"
    case the assertion_status_inconsistent guard flags as centrality_unmarked, so the
    summary column must not guess and instead reports empty, per the brief's "the text of
    the single central assertion, else empty"."""
    text = claim_record_csv(_v3_job_result()).lstrip(BOM)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert parsed[0]["n_assertions"] == "2"
    assert parsed[0]["central_assertion"] == ""


def test_csv_central_assertion_is_the_single_central_assertion_text():
    text = claim_record_csv(_v3_job_result()).lstrip(BOM)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert parsed[1]["n_assertions"] == "1"
    # Formula-neutralised, like every other untrusted text cell (the fixture's assertion
    # text starts with "=").
    assert parsed[1]["central_assertion"] == "'=SUM(1,1) learners improved"


def test_csv_of_an_empty_result_is_just_the_header():
    text = claim_record_csv({})
    assert text.startswith(BOM)
    assert text.lstrip(BOM).splitlines() == [EXPECTED_HEADER]


# --- GET /tasks/{task_id}/claim-record ---------------------------------------------------


async def _register(client, email: str) -> str:
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Claim Record User",
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
    job_type: JobType = JobType.claim_verify,
    result: dict | None = None,
) -> UUID:
    headers = {"Authorization": f"Bearer {token}"}
    me = await client.get("/api/v1/auth/me", headers=headers)
    user_id = UUID(me.json()["id"])
    proj = await client.post(
        "/api/v1/projects", json={"title": "Claim Record Project"}, headers=headers
    )
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
async def test_claim_record_json_returns_the_built_record(client, db_session):
    token = await _register(client, "claim-record-json@example.com")
    job_id = await _seed_job(client, db_session, token, result=_job_result())

    res = await client.get(
        f"/api/v1/tasks/{job_id}/claim-record",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 200, res.text
    body = res.json()
    assert body == build_claim_record(_job_result())
    assert len(body["records"]) == 5


@pytest.mark.asyncio
async def test_claim_record_csv_is_an_attachment(client, db_session):
    token = await _register(client, "claim-record-csv@example.com")
    job_id = await _seed_job(client, db_session, token, result=_job_result())

    res = await client.get(
        f"/api/v1/tasks/{job_id}/claim-record",
        params={"format": "csv"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("text/csv")
    assert res.headers["content-disposition"] == (
        f'attachment; filename="claim-record-{job_id}.csv"'
    )
    assert res.content.startswith(BOM.encode("utf-8"))
    assert res.text == claim_record_csv(_job_result())
    assert res.text.lstrip(BOM).splitlines()[0] == EXPECTED_HEADER


@pytest.mark.asyncio
async def test_claim_record_rejects_unknown_format(client, db_session):
    token = await _register(client, "claim-record-fmt@example.com")
    job_id = await _seed_job(client, db_session, token, result=_job_result())

    res = await client.get(
        f"/api/v1/tasks/{job_id}/claim-record",
        params={"format": "xlsx"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_claim_record_is_owner_only(client, db_session):
    owner = await _register(client, "claim-record-owner@example.com")
    other = await _register(client, "claim-record-other@example.com")
    job_id = await _seed_job(client, db_session, owner, result=_job_result())

    res = await client.get(
        f"/api/v1/tasks/{job_id}/claim-record",
        headers={"Authorization": f"Bearer {other}"},
    )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_claim_record_404s_for_other_job_types_and_missing_results(client, db_session):
    token = await _register(client, "claim-record-404@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    not_claim = await _seed_job(
        client, db_session, token, job_type=JobType.smart_search, result={"papers": []}
    )
    res = await client.get(f"/api/v1/tasks/{not_claim}/claim-record", headers=headers)
    assert res.status_code == 404

    no_result = await _seed_job(client, db_session, token, result=None)
    res = await client.get(f"/api/v1/tasks/{no_result}/claim-record", headers=headers)
    assert res.status_code == 404

    res = await client.get(f"/api/v1/tasks/{uuid4()}/claim-record", headers=headers)
    assert res.status_code == 404

"""The report a verification job writes must be readable back over HTTP."""

import json
import os
import uuid
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from app.core.security import create_access_token  # noqa: E402
from app.models.analysis_job import JobType  # noqa: E402
from app.schemas.fulltext import ClaimVerification, ClaimVerificationReport  # noqa: E402
from tests.t5_fixtures import (  # noqa: E402
    FakeAgent,
    make_session_factory,
    seed_draft,
    seed_job,
    seed_paper,
    seed_project,
    seed_user,
    tiptap_paragraph,
)

# demo/expected/claim_report.json is a live artefact: demo/run_demo.py writes it from
# whatever the pipeline last produced, and a later prompt version regenerates it from
# the v3 prompt. It may currently be
# v1.1.0-shaped (prompt sha e3c9f5e99a43, the v2 prompt: assertions carry only
# text/kind/verdict/claim_value/source_value/quote, no unstated_details, evidence_quotes
# or diagnostics key anywhere), but this test file must not assert that shape persists --
# see test_report_schema_accepts_a_v1_1_0_shaped_job_result_from_the_demo_fixture below,
# which derives its expectations from whatever is actually on disk instead of pinning the
# pre-v3 shape.
DEMO_CLAIM_REPORT_PATH = (
    Path(__file__).resolve().parents[2] / "demo" / "expected" / "claim_report.json"
)

CHUNKS = [{"section": "results", "text": "Test scores improved by 12 percent."}]


def test_report_schema_accepts_the_written_contract():
    report = ClaimVerificationReport(
        draft_id=uuid.uuid4(),
        verifications=[],
        verified_count=1,
        unsupported_count=2,
        nuance_count=3,
        abstract_only_count=4,
        error_count=5,
        needs_rewrite=3,
        needs_removal=2,
    )
    assert report.needs_rewrite == 3
    assert report.needs_removal == 2
    assert report.error_count == 5
    # These additions default so that reports written before v1.1.0 still parse.
    assert report.provenance is None
    assert report.full_text_coverage == 0.0
    # These additions default so that reports written before they existed still
    # parse (acceptance criterion 13, section 3.3).
    assert report.contradicted_count == 0
    assert report.guarded_count == 0


def test_report_schema_accepts_an_old_shape_job_result_dict():
    """A dict shaped exactly like an older job.result -- no ``assertions``,
    ``model_status``, ``machine_reasons``, ``contradicted_count`` or ``guarded_count``
    keys anywhere -- must still validate via ``**`` unpacking (acceptance criterion 13)."""
    old_shape_job_result = {
        "draft_id": str(uuid.uuid4()),
        "verifications": [
            {
                "claim_text": "Scores improved (Smith, 2020).",
                "paper_id": str(uuid.uuid4()),
                "status": "needs_nuance",
                "evidence_quote": None,
                "explanation": "Overstated.",
                "suggested_revision": "Scores improved modestly (Smith, 2020).",
                "model_reported": "deepseek-v4-flash",
                "citation": "(Smith, 2020)",
                "paper_doi": None,
                "paper_title": "Cited Work",
                "acquisition_route": None,
                "evidence_location": None,
            }
        ],
        "verified_count": 0,
        "unsupported_count": 0,
        "nuance_count": 1,
        "abstract_only_count": 0,
        "error_count": 0,
        "needs_rewrite": 1,
        "needs_removal": 0,
        "provenance": None,
        "full_text_coverage": 1.0,
    }
    report = ClaimVerificationReport(**old_shape_job_result)
    assert report.nuance_count == 1
    assert report.contradicted_count == 0
    assert report.guarded_count == 0
    assert report.verifications[0].assertions == []
    assert report.verifications[0].model_status is None
    assert report.verifications[0].machine_reasons == []


# ---------- v1.1.0-shaped job result stays readable ----------


def test_report_schema_accepts_a_v1_1_0_shaped_job_result_from_the_demo_fixture():
    """``demo/expected/claim_report.json`` is the demo's own on-disk job result. It is not
    frozen: ``demo/run_demo.py`` writes it from whatever the pipeline last produced, and a
    later prompt version regenerates it from the v3 prompt, at which
    point it gains ``unstated_details``, ``evidence_quotes``, ``diagnostics`` values and a
    ``central`` assertion. This test does not pin its expectations to either shape. It
    reads whatever is actually on disk and asserts the one schema-level guarantee that
    holds regardless: an additive field absent from the raw dict defaults to its
    documented empty value, and one present in the raw dict round-trips through the model
    unchanged. That keeps the test making a claim about
    the schema rather than about whichever prompt version last produced the fixture."""
    v1_1_0_job_result = json.loads(DEMO_CLAIM_REPORT_PATH.read_text(encoding="utf-8"))

    report = ClaimVerificationReport(**v1_1_0_job_result)

    assert report.verified_count == v1_1_0_job_result["verified_count"]
    assert report.contradicted_count == v1_1_0_job_result["contradicted_count"]
    assert report.guarded_count == v1_1_0_job_result["guarded_count"]
    assert len(report.verifications) == len(v1_1_0_job_result["verifications"])

    exercised_an_assertion = False
    for verification, raw_verification in zip(
        report.verifications, v1_1_0_job_result["verifications"]
    ):
        # New top-level ClaimVerification/ClaimVerificationOutput fields default empty
        # when the raw dict omits them, and round-trip unchanged when it doesn't.
        assert verification.unstated_details == raw_verification.get("unstated_details", [])
        assert verification.evidence_quotes == raw_verification.get("evidence_quotes", [])
        assert verification.diagnostics == raw_verification.get("diagnostics", [])
        for assertion, raw_assertion in zip(
            verification.assertions, raw_verification.get("assertions", [])
        ):
            exercised_an_assertion = True
            # New ClaimAssertion fields default to their documented values when the raw
            # dict omits them, and round-trip unchanged when it doesn't.
            assert assertion.central == raw_assertion.get("central", False)
            assert assertion.entity == raw_assertion.get("entity")
            assert assertion.measure == raw_assertion.get("measure")
            assert assertion.quotes == raw_assertion.get("quotes", [])

    # At least one verification in the fixture carries assertions, so the loop above
    # actually exercised the ClaimAssertion field checks and did not pass vacuously.
    assert any(v.assertions for v in report.verifications)
    assert exercised_an_assertion


def test_evidence_quote_stays_the_flat_field_set_to_the_first_span():
    """``evidence_quote`` keeps its name, type and position on ``ClaimVerification``:
    it is a plain optional string, independent of the ``evidence_quotes`` list, so
    that setting it to the first span of ``evidence_quotes`` (the service's job,
    not this schema's) leaves it a flat string and does not change its type or
    position."""
    spans = ["scores rose from 8 percent", "to 16 percent"]

    cv = ClaimVerification(
        claim_text="Scores rose from 8% to 16%.",
        paper_id=uuid.uuid4(),
        status="verified",
        explanation="Two spans compose to support the claim.",
        evidence_quotes=spans,
        evidence_quote=spans[0],
    )

    assert cv.evidence_quote == spans[0]
    assert isinstance(cv.evidence_quote, str)
    assert cv.evidence_quotes == spans

    # Round-tripping through dict/JSON (as the job-result store does) keeps
    # evidence_quote a flat string field, not a list, and not folded into
    # evidence_quotes.
    dumped = cv.model_dump()
    assert dumped["evidence_quote"] == spans[0]
    assert dumped["evidence_quotes"] == spans


@pytest.mark.asyncio
async def test_written_result_is_flat_and_readable(client, db_session):
    from app.services.fulltext import verify_and_heal_claims

    # The draft is authorized by owner (T6): mint the token for the same seeded user that
    # owns the draft, instead of registering an unrelated account via the HTTP endpoint.
    user = await seed_user(db_session)
    token = create_access_token(str(user.id))
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
    )
    draft = await seed_draft(
        db_session, project, user, tiptap_paragraph("Scores improved (Smith, 2020).")
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="needs_nuance",
            explanation="Overstated.",
            suggested_revision="Scores improved modestly (Smith, 2020).",
        )
    )
    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id,
            draft_id=draft.id,
            job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert "report" not in job.result
    assert "removed" not in job.result
    # verify_and_heal_claims actually heals the draft; the
    # flat contract below is otherwise unchanged (every field still describes the raw,
    # unfiltered verification pass), and three keys are additive: `healed`,
    # `finalize_stats`, and `final_report` (the clean, every-row-verified view).
    assert job.result["healed"] is True
    assert job.result["finalize_stats"]["sentences_removed_unverified"] == 1
    assert job.result["final_report"] == {"verifications": [], "verified_count": 0}
    assert job.result["draft_id"] == str(draft.id)
    assert job.result["nuance_count"] == 1
    assert job.result["needs_rewrite"] == 1
    assert job.result["needs_removal"] == 0
    # Provenance and coverage are stored flat next to the counts. A fake agent
    # that reports no model still yields a well-formed record.
    assert job.result["full_text_coverage"] == 1.0
    provenance = job.result["provenance"]
    assert provenance["agent"] == "claim_verification"
    assert provenance["calls"] == 1
    assert provenance["temperature"] == 0.0
    assert provenance["model_reported"] == []
    assert provenance["system_fingerprints"] == []
    assert provenance["input_tokens"] is None
    assert provenance["prompt_version"].startswith("sha256:")
    assert job.result["verifications"][0]["model_reported"] is None

    response = await client.get(
        f"/api/v1/drafts/{draft.id}/claim-verification",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["draft_id"] == str(draft.id)
    assert data["nuance_count"] == 1
    assert data["needs_rewrite"] == 1
    assert len(data["verifications"]) == 1
    assert data["verifications"][0]["suggested_revision"] == (
        "Scores improved modestly (Smith, 2020)."
    )
    assert data["full_text_coverage"] == 1.0
    assert data["provenance"]["agent"] == "claim_verification"
    assert data["provenance"]["calls"] == 1


@pytest.mark.asyncio
async def test_written_result_carries_the_claim_record_trace_fields(db_session):
    """The claim-verification record export (app.services.claim_record) reads five
    system-owned fields straight off each verification: citation, paper_doi,
    paper_title, acquisition_route, evidence_location. A regression that stopped
    _verify_one from populating them would leave the rest of the suite green while
    silently emptying five of the eleven CSV columns, so pin them here against the real
    pipeline rather than only the hand-written dicts in tests/test_claim_record.py.
    """
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="A Traceable Paper",
        doi="10.1/traceable",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_source": "unpaywall",
            "fulltext_chunks": CHUNKS,
        },
    )
    draft = await seed_draft(
        db_session, project, user, tiptap_paragraph("Scores improved (Smith, 2020).")
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote="Test scores improved by 12 percent.",
            explanation="Matches the reported effect size.",
        )
    )
    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id,
            draft_id=draft.id,
            job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["citation"] == "(Smith, 2020)"
    assert verification["paper_doi"] == "10.1/traceable"
    assert verification["paper_title"] == "A Traceable Paper"
    assert verification["acquisition_route"] == "unpaywall"
    assert verification["evidence_location"] == "chunk 1 (results)"

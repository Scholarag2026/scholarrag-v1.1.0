"""Claim verification must build real AnalysisDependencies and surface agent errors."""

import os
import uuid
from types import SimpleNamespace

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from app.models.analysis_job import JobType  # noqa: E402
from app.schemas.fulltext import ClaimVerification  # noqa: E402
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

CHUNKS = [{"section": "results", "text": "Test scores improved by 12 percent."}]


async def _seed_world(db_session):
    user = await seed_user(db_session)
    project = await seed_project(
        db_session, user, description="AI tutoring study", target_journal="Nature"
    )
    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
    )
    draft = await seed_draft(
        db_session,
        project,
        user,
        tiptap_paragraph("Test scores improved (Smith, 2020)."),
    )
    job = await seed_job(db_session, project, JobType.claim_verify)
    return user, project, paper, draft, job


@pytest.mark.asyncio
async def test_verify_and_heal_passes_real_dependencies(db_session):
    from app.services.fulltext import verify_and_heal_claims

    _user, project, paper, draft, job = await _seed_world(db_session)
    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote="Test scores improved by 12 percent.",
            explanation="Supported by the results section.",
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

    assert len(agent.calls) == 1
    _prompt, deps = agent.calls[0]
    assert deps is not None
    assert str(deps.project_id) == str(project.id)
    assert deps.project_description == "AI tutoring study"
    assert deps.target_journal == "Nature"
    assert deps.citation_style == "APA"

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["verified"] == 1
    assert job.result["error_count"] == 0


@pytest.mark.asyncio
async def test_agent_failure_is_reported_as_error_status(db_session):
    from app.services.fulltext import verify_and_heal_claims

    _user, project, _paper, draft, job = await _seed_world(db_session)
    agent = FakeAgent(raises=RuntimeError("deepseek exploded"))

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
    assert job.result["error_count"] == 1
    assert job.result["abstract_only_count"] == 0
    statuses = [v["status"] for v in job.result["verifications"]]
    assert statuses == ["error"]


@pytest.mark.asyncio
async def test_verify_user_edits_passes_real_dependencies(db_session):
    from app.services.fulltext import verify_user_edits

    _user, project, paper, draft, job = await _seed_world(db_session)
    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="needs_nuance",
            explanation="Overstated.",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_user_edits(
            project_id=project.id,
            draft_id=draft.id,
            changed_sections=["Test scores improved (Smith, 2020)."],
            job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    assert len(agent.calls) == 1
    _prompt, deps = agent.calls[0]
    assert str(deps.project_id) == str(project.id)
    assert deps.citation_style == "APA"


@pytest.mark.asyncio
async def test_verify_user_edits_writes_the_same_flat_contract(db_session):
    from app.services.fulltext import verify_user_edits

    _user, project, paper, draft, job = await _seed_world(db_session)
    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="unsupported",
            explanation="Not stated in the source.",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_user_edits(
            project_id=project.id,
            draft_id=draft.id,
            changed_sections=["Test scores improved (Smith, 2020)."],
            job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.result["draft_id"] == str(draft.id)
    assert job.result["sections_checked"] == 1
    assert job.result["unsupported_count"] == 1
    assert job.result["needs_removal"] == 1
    assert job.result["needs_rewrite"] == 0
    assert len(job.result["verifications"]) == 1
    # `healed` is a typed field on ClaimVerificationReport
    # (default None), so it is always present in a dumped report; verify_user_edits
    # never sets it, so it stays None -- not True,
    # the only value that means "this report's draft was actually healed".
    assert job.result.get("healed") is None
    assert job.result.get("finalize_stats") is None
    assert job.result.get("final_report") is None


@pytest.mark.asyncio
async def test_claims_are_verified_concurrently(db_session):
    """Ten claims must not be ten serial round trips."""
    import asyncio

    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    for n in range(10):
        await seed_paper(
            db_session,
            project,
            title=f"Cited Work {n}",
            authors=[{"name": f"Author{n} Xu"}],
            year=2000 + n,
            metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
        )
    sentences = " ".join(f"Finding {n} holds (Xu, {2000 + n})." for n in range(10))
    draft = await seed_draft(db_session, project, user, tiptap_paragraph(sentences))
    job = await seed_job(db_session, project, JobType.claim_verify)

    state = {"in_flight": 0, "max_in_flight": 0}

    class ConcurrencyProbeAgent:
        async def run(self, prompt, deps=None):
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            await asyncio.sleep(0.05)
            state["in_flight"] -= 1
            return type(
                "R",
                (),
                {
                    "output": ClaimVerification(
                        claim_text="c",
                        paper_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                        status="verified",
                        explanation="ok",
                    )
                },
            )()

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=ConcurrencyProbeAgent(),
    ):
        await verify_and_heal_claims(
            project_id=project.id,
            draft_id=draft.id,
            job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    assert state["max_in_flight"] > 1, "claim verification is still strictly sequential"
    assert state["max_in_flight"] <= 4, "must stay bounded by analysis_concurrency"


@pytest.mark.asyncio
async def test_cancelled_job_stops_verifying(db_session):
    from app.models.analysis_job import JobStatus
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    for n in range(20):
        await seed_paper(
            db_session,
            project,
            title=f"W{n}",
            authors=[{"name": f"Cancel{n} Yu"}],
            year=1900 + n,
            metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
        )
    sentences = " ".join(f"Claim {n} (Yu, {1900 + n})." for n in range(20))
    draft = await seed_draft(db_session, project, user, tiptap_paragraph(sentences))
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="c",
            paper_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
            status="verified",
            explanation="ok",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ), patch(
        "app.services.task.should_abort", new_callable=AsyncMock, return_value=True
    ):
        await verify_and_heal_claims(
            project_id=project.id,
            draft_id=draft.id,
            job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status == JobStatus.cancelled


class ProvenanceFakeAgent(FakeAgent):
    """FakeAgent whose run results carry the fields pydantic-ai puts on AgentRunResult."""

    async def run(self, prompt, deps=None):
        result = await super().run(prompt, deps)
        result.response = SimpleNamespace(
            model_name="deepseek-v4-flash",
            provider_name="deepseek",
            provider_response_id=f"resp-{len(self.calls)}",
            provider_details={"system_fingerprint": "fp-1"},
        )
        result.usage = lambda: SimpleNamespace(input_tokens=100, output_tokens=20)
        return result


@pytest.mark.asyncio
async def test_report_records_provenance_and_full_text_coverage(db_session):
    """Records which model answered, under which prompt, and how many claims had text."""
    from app.agents.claim_verification_agent import CLAIM_VERIFICATION_PROMPT_VERSION
    from app.config import settings
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(
        db_session,
        project,
        title="With text A",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
    )
    await seed_paper(
        db_session,
        project,
        title="With text B",
        authors=["Ken Jones"],
        year=2019,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
    )
    await seed_paper(
        db_session,
        project,
        title="Abstract only",
        authors=[{"name": "Amy Brown"}],
        year=2018,
        metadata_={"fulltext_status": "not_found"},
    )
    draft = await seed_draft(
        db_session,
        project,
        user,
        tiptap_paragraph(
            "Scores improved (Smith, 2020). Engagement rose (Jones, 2019). "
            "Costs fell (Brown, 2018)."
        ),
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = ProvenanceFakeAgent(
        output=lambda _prompt: ClaimVerification(
            claim_text="c",
            paper_id=uuid.UUID(int=1),
            status="verified",
            evidence_quote="Test scores improved by 12 percent.",
            explanation="ok",
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
    assert job.status.value == "completed", job.error
    # The claim whose source has no full text never reaches the model.
    assert len(agent.calls) == 2
    assert job.result["verified_count"] == 2
    assert job.result["abstract_only_count"] == 1
    assert job.result["full_text_coverage"] == 0.667
    # Authorised 2026-09-10: `guard_digest`/`quote_relocation_version` sit next to
    # `prompt_version`, recording which frozen guard set and which version (if any) of the
    # quote-relocation step produced this report's claims. `verification_policy_version`
    # (task authorisation 2026-09-10, Part B) records which version of the whole shared
    # policy (relocation and the bounded repair turn together) produced them.
    # `repair_prompt_version` (task authorisation 2026-09-11) records which version of the
    # repair turn's own prompt template this report's claims were repaired under, if any.
    from app.agents.claim_verification_agent import QUOTE_REPAIR_PROMPT_VERSION
    from app.services.fulltext import (
        GUARD_DIGEST,
        QUOTE_RELOCATION_VERSION,
        VERIFICATION_POLICY_VERSION,
    )

    assert job.result["provenance"] == {
        "agent": "claim_verification",
        "model_configured": settings.deepseek_model,
        "model_reported": ["deepseek-v4-flash"],
        "system_fingerprints": ["fp-1"],
        "temperature": 0.0,
        "prompt_version": CLAIM_VERIFICATION_PROMPT_VERSION,
        "guard_digest": GUARD_DIGEST,
        "quote_relocation_version": QUOTE_RELOCATION_VERSION,
        "verification_policy_version": VERIFICATION_POLICY_VERSION,
        "repair_prompt_version": QUOTE_REPAIR_PROMPT_VERSION,
        "calls": 2,
        "input_tokens": 200,
        "output_tokens": 40,
    }
    by_status = {v["status"]: v for v in job.result["verifications"]}
    assert by_status["verified"]["model_reported"] == "deepseek-v4-flash"
    assert by_status["no_full_text"]["model_reported"] is None


@pytest.mark.asyncio
async def test_report_carries_citation_coverage_agreeing_with_the_claims_verified(db_session):
    """citation_coverage is
    additive on ClaimVerificationReport, and its counts agree with what was actually
    extracted and sent to the verifier."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(
        db_session,
        project,
        title="With text A",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
    )
    await seed_paper(
        db_session,
        project,
        title="With text B",
        authors=["Ken Jones"],
        year=2019,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
    )
    draft = await seed_draft(
        db_session,
        project,
        user,
        tiptap_paragraph(
            "Scores improved (Smith, 2020). Engagement rose (Jones, 2019). "
            "Costs fell (Nguyen, 2021)."
        ),
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = ProvenanceFakeAgent(
        output=lambda _prompt: ClaimVerification(
            claim_text="c",
            paper_id=uuid.UUID(int=1),
            status="verified",
            evidence_quote="Test scores improved by 12 percent.",
            explanation="ok",
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
    assert job.status.value == "completed", job.error
    coverage = job.result["citation_coverage"]
    # Three citations found (Smith, Jones -- both in the library; Nguyen -- not), all
    # via the pre-existing author-year regexes (no citation-link map on this draft).
    assert coverage["found"] == 3
    assert coverage["linked"] == 2
    assert coverage["sent_to_verifier"] == 3
    assert coverage["unresolved"] == 1
    assert coverage["unresolved_citations"] == ["(Nguyen, 2021)"]
    assert coverage["by_source"] == {"mapping": 0, "author-year": 3, "numbered": 0}
    # Every claim the coverage says was sent to the verifier appears in the report.
    assert len(job.result["verifications"]) == coverage["sent_to_verifier"]


@pytest.mark.asyncio
async def test_verify_user_edits_never_sets_citation_coverage(db_session):
    """`verify_user_edits` only ever receives plain strings, so
    it has no Tiptap document to build a coverage object from; the field stays `None`,
    same as any report written before it existed."""
    from app.services.fulltext import verify_user_edits

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(
        db_session, project, title="A", authors=[{"name": "Jane Smith"}], year=2020,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
    )
    draft = await seed_draft(db_session, project, user, tiptap_paragraph("Placeholder."))
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Scores improved (Smith, 2020).",
            paper_id=uuid.UUID(int=1),
            status="verified",
            evidence_quote="Test scores improved by 12 percent.",
            explanation="ok",
        )
    )
    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_user_edits(
            project_id=project.id,
            draft_id=draft.id,
            changed_sections=["Scores improved (Smith, 2020)."],
            job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["citation_coverage"] is None


@pytest.mark.asyncio
async def test_provenance_extraction_failure_keeps_the_verdict(db_session):
    """Provenance bookkeeping must never turn a produced verdict into an error."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(
        db_session,
        project,
        title="With text A",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
    )
    draft = await seed_draft(
        db_session, project, user, tiptap_paragraph("Scores improved (Smith, 2020).")
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = ProvenanceFakeAgent(
        output=lambda _prompt: ClaimVerification(
            claim_text="c",
            paper_id=uuid.UUID(int=1),
            status="verified",
            evidence_quote="Test scores improved by 12 percent.",
            explanation="ok",
        )
    )
    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ), patch(
        "app.services.fulltext.provenance_from_run", side_effect=RuntimeError("bad response")
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["verified_count"] == 1
    assert job.result["error_count"] == 0
    assert job.result["verifications"][0]["model_reported"] is None
    assert job.result["provenance"]["calls"] == 0



# ---------------------------------------------------------------- code guards


@pytest.mark.asyncio
async def test_guard_sets_model_status_and_machine_reasons_and_report_counts(db_session):
    """A "verified" answer with no evidence_quote is guard 3's textbook case (brief
    section 4.4): the report keeps the model's own answer in ``model_status`` for the
    "Model answered X; automated check made this stricter" UI copy, records the slug
    in ``machine_reasons``, and the aggregate counts move accordingly (``guarded_count``,
    ``nuance_count``) without touching ``needs_rewrite``/``needs_removal``'s meaning."""
    from app.services.fulltext import verify_and_heal_claims

    _user, project, paper, draft, job = await _seed_world(db_session)
    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            explanation="Supported by the results section.",
            # No evidence_quote: guard 3 (quote fidelity) must demote this.
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
    assert verification["status"] == "needs_nuance"
    assert verification["model_status"] == "verified"
    assert verification["machine_reasons"] == ["quote_not_verbatim"]
    assert job.result["verified_count"] == 0
    assert job.result["nuance_count"] == 1
    assert job.result["needs_rewrite"] == 1
    assert job.result["guarded_count"] == 1
    assert job.result["contradicted_count"] == 0


@pytest.mark.asyncio
async def test_escaped_quotes_are_normalised_before_storage_and_before_the_quote_guard(
    db_session,
):
    """A model that
    pre-escapes its own quote marks -- the defect found in the ``supported-1`` entry of
    ``demo/expected/claim_report.json`` -- must not (a) reach storage with literal
    backslashes, or (b) trip a false quote-fidelity demotion, because the guard reads the
    same, already-normalised ``evidence_quote`` the user will later see."""
    from app.services.fulltext import verify_and_heal_claims

    _user, project, paper, draft, job = await _seed_world(db_session)
    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote='\\"Test scores improved by 12 percent.\\"',
            explanation=(
                '\\"Supported by the results section, which reports the exact figure.\\"'
            ),
            suggested_revision="Consider noting it\\'s a single-cohort result.",
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
    # (a) no literal backslash survives into storage.
    assert "\\" not in verification["explanation"]
    assert "\\" not in verification["evidence_quote"]
    assert "\\" not in verification["suggested_revision"]
    # The wrapping quote pair is stripped from `explanation` only.
    assert verification["explanation"] == (
        "Supported by the results section, which reports the exact figure."
    )
    assert verification["suggested_revision"] == "Consider noting it's a single-cohort result."
    # `evidence_quote` keeps its wrapping quote marks (they mark the quoted span for
    # `_quote_segments`); only the escapes are collapsed.
    assert verification["evidence_quote"] == '"Test scores improved by 12 percent."'
    # (b) the guard saw the same normalised text: no false demotion, no guard fired.
    assert verification["status"] == "verified"
    assert verification["model_status"] == "verified"
    assert verification["machine_reasons"] == []
    assert job.result["guarded_count"] == 0


@pytest.mark.asyncio
async def test_claim_text_stored_is_the_extracted_sentence_not_the_model_transcription(
    db_session,
):
    """`verify_and_heal_claims`
    builds `verification` from `result.output.model_dump()`, so every field the model's
    own `output_type` declares -- `claim_text` included -- starts out as whatever the
    model echoed back, not the sentence the extractor actually sent it. `paper_id`,
    `citation`, `paper_doi`, `paper_title` and `acquisition_route` are all reset after the
    call for exactly this reason; `claim_text` must be too, so the report always carries
    the draft sentence and never the model's own transcription of it."""
    from app.services.fulltext import verify_and_heal_claims

    _user, project, paper, draft, job = await _seed_world(db_session)
    extracted_sentence = "Test scores improved (Smith, 2020)."
    agent = FakeAgent(
        output=ClaimVerification(
            # Deliberately not verbatim: dropped a word and changed a number, the way a
            # model transcribing the sentence back to us might.
            claim_text="Test score improved (Smith, 2021).",
            paper_id=paper.id,
            status="verified",
            evidence_quote="Test scores improved by 12 percent.",
            explanation="Supported by the results section.",
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
    # The stored claim_text is the extracted draft sentence, not the model's echo of it.
    assert verification["claim_text"] == extracted_sentence
    assert verification["claim_text"] != "Test score improved (Smith, 2021)."
    # The prompt, the guards and the verdict all already used the extracted sentence, so
    # neither the status nor the guard bookkeeping moves because of this fix.
    assert verification["status"] == "verified"
    assert verification["model_status"] == "verified"
    assert verification["machine_reasons"] == []
    assert job.result["guarded_count"] == 0
    assert job.result["verified_count"] == 1


@pytest.mark.asyncio
async def test_non_verbatim_assertion_quote_is_dropped_before_storage(db_session):
    """Guard 3 (``_guard_quote_fidelity``) only ever
    inspects the top-level ``evidence_quote``/``evidence_quotes``, never
    ``assertions[].quote``/``assertions[].quotes``, so a paraphrase sitting only on an
    assertion verifies -- but it must still be dropped from what ``verify_and_heal_claims``
    stores, end to end through the real service, not only inside
    ``verify_claim_with_policy``."""
    from app.schemas.fulltext import ClaimAssertion
    from app.services.fulltext import verify_and_heal_claims

    _user, project, paper, draft, job = await _seed_world(db_session)
    good_quote = "Test scores improved by 12 percent."
    bad_quote = "we found scores improved by a large margin"
    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote=good_quote,
            explanation="Supported by the results section.",
            assertions=[
                ClaimAssertion(
                    text="Test scores improved.",
                    kind="direction",
                    verdict="supported",
                    central=False,
                    quote=bad_quote,
                    quotes=[good_quote, bad_quote],
                )
            ],
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    # The guard never saw the assertion's own quote, so the status is untouched.
    assert verification["status"] == "verified"
    assert verification["model_status"] == "verified"
    assert verification["machine_reasons"] == []
    (assertion,) = verification["assertions"]
    assert assertion["quote"] is None  # the non-verbatim singular quote is dropped
    assert assertion["quotes"] == [good_quote]  # only the verbatim one survives


@pytest.mark.asyncio
async def test_deterministic_no_full_text_path_has_no_model_status_or_guard(db_session):
    """No model call is ever made for a source with zero chunks, so no guard runs and
    ``model_status``/``machine_reasons`` stay at their un-guarded defaults."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(
        db_session,
        project,
        title="Abstract only",
        authors=[{"name": "Amy Brown"}],
        year=2018,
        metadata_={"fulltext_status": "not_found"},
    )
    draft = await seed_draft(
        db_session, project, user, tiptap_paragraph("Costs fell (Brown, 2018).")
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=FakeAgent(raises=AssertionError("must never be called")),
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["status"] == "no_full_text"
    assert verification["model_status"] is None
    assert verification["machine_reasons"] == []
    assert job.result["guarded_count"] == 0


@pytest.mark.asyncio
async def test_contradicted_count_counts_verifications_with_a_contradicted_assertion(
    db_session,
):
    from app.schemas.fulltext import ClaimAssertion
    from app.services.fulltext import verify_and_heal_claims

    _user, project, paper, draft, job = await _seed_world(db_session)
    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="unsupported",
            explanation="The source reports a decrease, not an increase.",
            assertions=[
                ClaimAssertion(
                    text="Test scores improved.",
                    kind="direction",
                    verdict="contradicted",
                    claim_value="improved",
                    source_value="declined",
                    quote="Test scores declined by 12 percent.",
                )
            ],
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.result["contradicted_count"] == 1
    assert job.result["unsupported_count"] == 1
    assert job.result["needs_removal"] == 1


# ---------------------------------------------------------------- v3 guards
# (diagnostics, the assertion-status consistency guard, and multi-span evidence_quotes)


@pytest.mark.asyncio
async def test_v3_evidence_quote_is_set_to_the_first_span_of_evidence_quotes(db_session):
    """brief section 1.2/1.3: `evidence_quote` keeps its name, type and position by being
    set to the first span of `evidence_quotes`, overriding whatever the model itself put in
    the single-span field -- this is P3's gate, not P1's."""
    from app.services.fulltext import verify_and_heal_claims

    _user, project, paper, draft, job = await _seed_world(db_session)
    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote="a stale single-span quote the model also filled",
            evidence_quotes=["Test scores improved by 12 percent."],
            explanation="Supported by the results section.",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["evidence_quote"] == "Test scores improved by 12 percent."
    assert verification["evidence_quotes"] == ["Test scores improved by 12 percent."]
    assert verification["status"] == "verified"


@pytest.mark.asyncio
async def test_v3_multi_span_quote_fidelity_checks_every_evidence_quotes_span(db_session):
    """Tier B/guard 3 checks every span in `evidence_quotes`, not just
    the first; a second span that is not verbatim in any chunk still demotes `verified`.

    The second span differs from the source by a numeral ("5" for "3"): quote relocation
    (`app.services.fulltext.relocate_evidence_quotes`) never repoints a span whose edit
    touches a digit-bearing token, so this stays genuinely not verbatim. A spelled-out
    equivalent ("five" for "three") is protected the same way
    (`_is_protected_number_word`), so either spelling stays not verbatim here.
    """
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [
                {
                    "section": "results",
                    "text": (
                        "Test scores improved by 12 percent. "
                        "The effect held across 3 follow-up sessions."
                    ),
                }
            ],
        },
    )
    draft = await seed_draft(
        db_session, project, user, tiptap_paragraph("Test scores improved (Smith, 2020).")
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quotes=[
                "Test scores improved by 12 percent.",
                "The effect held across 5 follow-up sessions.",  # not verbatim
            ],
            explanation="Supported by the results section.",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["status"] == "needs_nuance"
    assert verification["model_status"] == "verified"
    assert verification["machine_reasons"] == ["quote_not_verbatim"]


@pytest.mark.asyncio
async def test_guard_quote_fidelity_tolerates_a_line_break_hyphenation_through_the_full_service(
    db_session,
):
    """Reproduces the
    ``hss-verbatim-01`` acceptance miss through the
    whole ``verify_and_heal_claims`` pipeline, not just the guard function in isolation.
    The stored chunk hyphenates "post-test" at a PDF line break; the model quoted the
    contiguous spelling. Without the fold in ``_normalise_for_match`` this would demote
    a correct ``verified`` answer to ``needs_nuance``."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [
                {
                    "section": "results",
                    "text": (
                        "Given the control group, there is a statistically significant "
                        "difference between post-\ntest and pre-test overall scores "
                        "(z=3.297, p=000<.0167)."
                    ),
                }
            ],
        },
    )
    draft = await seed_draft(
        db_session, project, user, tiptap_paragraph("Test scores improved (Smith, 2020).")
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote=(
                "there is a statistically significant difference between post-test and "
                "pre-test overall scores (z=3.297, p=000<.0167)."
            ),
            explanation="Supported by the results section.",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["status"] == "verified"
    assert verification["machine_reasons"] == []


@pytest.mark.asyncio
async def test_v3_assertion_status_inconsistent_floors_unsupported_when_central_is_absent(
    db_session,
):
    """The new consistency guard reaches through the full service pipeline: a central
    assertion marked `absent` floors `unsupported` even though the model itself answered
    `verified`, and the slug lands in `machine_reasons`, never in `diagnostics`."""
    from app.schemas.fulltext import ClaimAssertion
    from app.services.fulltext import verify_and_heal_claims

    _user, project, paper, draft, job = await _seed_world(db_session)
    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote="Test scores improved by 12 percent.",
            explanation="The paper never actually states this finding.",
            assertions=[
                ClaimAssertion(
                    text="Test scores improved.",
                    kind="direction",
                    verdict="absent",
                    central=True,
                )
            ],
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["model_status"] == "verified"
    assert verification["status"] == "unsupported"
    assert verification["machine_reasons"] == ["assertion_status_inconsistent"]
    assert verification["diagnostics"] == []
    assert job.result["needs_removal"] == 1


@pytest.mark.asyncio
async def test_v3_centrality_unmarked_diagnostic_never_changes_status(db_session):
    """Zero `central=True` assertions never fires the centrality clause (brief 1.3): the
    claim stays `verified` and `centrality_unmarked` is recorded only in `diagnostics`,
    a sibling of `machine_reasons`, never a member of it."""
    from app.schemas.fulltext import ClaimAssertion
    from app.services.fulltext import verify_and_heal_claims

    _user, project, paper, draft, job = await _seed_world(db_session)
    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote="Test scores improved by 12 percent.",
            explanation="Supported by the results section.",
            assertions=[
                ClaimAssertion(
                    text="Test scores improved.", kind="direction", verdict="supported",
                    central=False,
                ),
                ClaimAssertion(
                    text="By 12 percent.", kind="quantity", verdict="supported", central=False,
                ),
            ],
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["status"] == "verified"
    assert verification["machine_reasons"] == []
    assert verification["diagnostics"] == ["centrality_unmarked"]


@pytest.mark.asyncio
async def test_wrong_work_paper_is_verified_through_the_existing_no_full_text_path(db_session):
    """A paper the acquisition guard rejected as ``wrong_work`` carries no
    ``fulltext_chunks`` key, so it reaches the verifier through the same deterministic
    zero-chunk path as any other source with no full text -- no verifier prompt or guard
    change is needed for the rejection to show up here. The explanation must say *why*
    no chunks are available (the fetched full text did not match this paper), not the
    generic message shared with "no OA PDF was ever found", since nothing else in the
    report surfaces ``fulltext_reason`` to the user."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(
        db_session,
        project,
        title="Wrong Work Source",
        authors=[{"name": "Amy Brown"}],
        year=2018,
        metadata_={"fulltext_status": "abstract_only", "fulltext_reason": "wrong_work"},
    )
    draft = await seed_draft(
        db_session, project, user, tiptap_paragraph("Costs fell (Brown, 2018).")
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=FakeAgent(raises=AssertionError("must never be called")),
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["status"] == "no_full_text"
    assert verification["explanation"] == (
        "No full-text chunks available: the fetched full text did not match this "
        "paper's title or author."
    )
    assert job.result["abstract_only_count"] == 1


@pytest.mark.asyncio
async def test_publisher_interstitial_paper_is_verified_through_the_existing_no_full_text_path(
    db_session,
):
    """`_verify_one` gives `publisher_interstitial` the same treatment as `wrong_work` in
    the no-full-text explanation, rather than a paper the acquisition guard rejected
    because the publisher blocked access falling through to the generic "No full-text
    chunks available for this paper.", exactly like a source no OA link was ever found
    for."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(
        db_session,
        project,
        title="Blocked Source",
        authors=[{"name": "Amy Brown"}],
        year=2018,
        metadata_={"fulltext_status": "abstract_only", "fulltext_reason": "publisher_interstitial"},
    )
    draft = await seed_draft(
        db_session, project, user, tiptap_paragraph("Costs fell (Brown, 2018).")
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=FakeAgent(raises=AssertionError("must never be called")),
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["status"] == "no_full_text"
    assert verification["explanation"] == (
        "No full-text chunks available: the publisher blocked automatic access to "
        "the full text (a login wall or interstitial page, not the paper itself)."
    )
    assert job.result["abstract_only_count"] == 1


@pytest.mark.asyncio
async def test_no_full_text_explanation_stays_generic_without_a_wrong_work_reason(db_session):
    """Regression for the fix above: a source with no acquisition attempt at all (no
    ``fulltext_reason`` in its metadata) must keep the original, generic explanation --
    the wrong-work wording is added only when that reason is actually present."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(
        db_session,
        project,
        title="Never Acquired Source",
        authors=[{"name": "Amy Brown"}],
        year=2018,
        metadata_={"fulltext_status": "abstract_only"},
    )
    draft = await seed_draft(
        db_session, project, user, tiptap_paragraph("Costs fell (Brown, 2018).")
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=FakeAgent(raises=AssertionError("must never be called")),
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["status"] == "no_full_text"
    assert verification["explanation"] == "No full-text chunks available for this paper."


@pytest.mark.asyncio
async def test_stored_reference_chunk_is_filtered_out_before_the_verifier_sees_it(db_session):
    """`acquire_full_texts` is only one of four
    chunk-storage paths -- upload, paste and upload-confirm each write `fulltext_chunks`
    with no reference-list drop of their own, and every paper acquired before the guard
    existed keeps its unfiltered chunks forever. `_verify_claims` must re-run
    `drop_reference_and_backmatter_chunks` on every read, so a reference-list chunk already
    sitting in `metadata_` (regardless of how it got there) never reaches the model prompt
    and can never be located as "evidence" for a claim."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    reference_chunk_text = (
        "Rosen, J., & Wedin, A. (2015). Klassrumsinteraktion och flerspraakighet. "
        "Apples, 9(1), 1-18. Retrieved from https://apples.jyu.fi\n"
        "Sinclair, J. M., & Coulthard, R. M. (1975). Towards an analysis of discourse. "
        "Oxford, UK: Oxford University Press. pp. 12-34.\n"
        "Snell, J., Shaw, S., & Copland, F. (2015). Linguistic ethnography. London, UK: "
        "Palgrave Macmillan. 10.1057/9781137035035\n"
        "Swain, M. (1985). Communicative competence. Rowley, MA: Newbury House. "
        "pp. 235-253.\n"
        "Swain, M. (1995). Three functions of output. Oxford, UK: Oxford University "
        "Press. pp. 125-144.\n"
    )
    good_chunk_text = "Test scores improved by 12 percent."
    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [
                {"section": "results", "text": good_chunk_text},
                {"section": "discussion", "text": reference_chunk_text},
            ],
        },
    )
    draft = await seed_draft(
        db_session,
        project,
        user,
        tiptap_paragraph("Test scores improved (Smith, 2020)."),
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote=good_chunk_text,
            explanation="Supported by the results section.",
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

    assert len(agent.calls) == 1
    prompt, _deps = agent.calls[0]
    assert good_chunk_text in prompt
    assert "Retrieved from" not in prompt
    assert "Rosen, J." not in prompt

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["status"] == "verified"
    assert verification["evidence_location"] is not None
    assert "results" in verification["evidence_location"]


@pytest.mark.asyncio
async def test_v3_numeric_diagnostic_and_guard_reach_the_stored_report_end_to_end(
    db_session,
):
    """The renamed `_diagnose_numeric_not_in_source` reaches the stored ``diagnostics``
    field end to end, through its own slug, which never appears in ``machine_reasons``.

    The claim's numeral (``d = 1.62``) absent from the chunk now ALSO drives the new
    status-changing `_guard_numeric_value_absent` guard (its own, different slug,
    ``numeric_value_absent_from_source``), so `status` is no longer left at `verified`
    here -- deliberately, per the guard's own docstring. What the test still proves: the
    diagnostic's own slug still never appears in `machine_reasons`, and both it and the
    guard's slug reach the stored report end to end.
    """
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [
                {"section": "results", "text": "The effect was large, consistent with prior work."}
            ],
        },
    )
    draft = await seed_draft(
        db_session,
        project,
        user,
        tiptap_paragraph(
            "The effect was large (d = 1.62), consistent with prior work (Smith, 2020)."
        ),
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="The effect was large (d = 1.62), consistent with prior work (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote="The effect was large, consistent with prior work.",
            explanation="Supported by the results section.",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["status"] == "needs_nuance"
    assert verification["machine_reasons"] == ["numeric_value_absent_from_source"]
    assert verification["diagnostics"] == ["numeric_not_in_source"]


@pytest.mark.asyncio
async def test_guard_numeric_value_absent_floors_unsupported_through_the_full_service(
    db_session,
):
    """Reproduces the
    `hss-altered-15` run A shape (the lenient miss this guard exists to fix) through
    the whole `verify_and_heal_claims` pipeline. The model marks
    the assertion carrying the altered number (1370, altered from the source's 1000)
    `absent` rather than `contradicted`, so the assertion-consistency guard alone only
    floors `needs_nuance`; `_guard_numeric_value_absent`'s Rule B floors the remaining
    step, to `unsupported`, because the diagnostic independently finds 1370 absent from
    the source and the model's own assertion names that number.
    """
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [
                {
                    "section": "results",
                    "text": (
                        "The total number of instances for each category found in each "
                        "sub-corpus has been normalized per 1000 words. "
                        "Total 275 76.26 478 44.09"
                    ),
                }
            ],
        },
    )
    claim_text = (
        "The total number of interpersonal metadiscourse features (normalized per 1370 "
        "words) is higher in BR sub-corpus (76.26) than in RA sub-corpus (44.09) (Smith, "
        "2020)."
    )
    draft = await seed_draft(db_session, project, user, tiptap_paragraph(claim_text))
    job = await seed_job(db_session, project, JobType.claim_verify)

    quote = (
        "The total number of instances for each category found in each sub-corpus has "
        "been normalized per 1000 words."
    )
    agent = FakeAgent(
        output=ClaimVerification(
            claim_text=claim_text,
            paper_id=paper.id,
            status="needs_nuance",
            evidence_quote=quote,
            explanation="The normalisation base does not match.",
            assertions=[
                {
                    "text": "The total number of interpersonal metadiscourse features is "
                    "higher in the BR sub-corpus than in the RA sub-corpus.",
                    "kind": "direction",
                    "verdict": "supported",
                    "central": True,
                },
                {
                    "text": "The BR sub-corpus value is 76.26.",
                    "kind": "quantity",
                    "verdict": "supported",
                    "central": False,
                },
                {
                    "text": "The RA sub-corpus value is 44.09.",
                    "kind": "quantity",
                    "verdict": "supported",
                    "central": False,
                },
                {
                    "text": "The values are normalized per 1370 words.",
                    "kind": "quantity",
                    "verdict": "absent",
                    "central": False,
                },
            ],
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    verification = job.result["verifications"][0]
    assert verification["status"] == "unsupported"
    assert verification["model_status"] == "needs_nuance"
    assert "assertion_status_inconsistent" in verification["machine_reasons"]
    assert "numeric_value_absent_from_source" in verification["machine_reasons"]
    assert verification["diagnostics"] == ["numeric_not_in_source"]


@pytest.mark.asyncio
async def test_filtered_chunks_are_computed_once_per_paper_not_once_per_claim(db_session):
    """Running `drop_reference_and_backmatter_chunks` inside `_verify_one` would make a
    draft citing the same paper from N claims pay for the same CPU-bound recomputation N
    times. It must be memoized per paper id in the enclosing scope of `_verify_claims`,
    so three claims against one paper compute the filtered chunk list exactly once."""
    import app.services.fulltext as fulltext_module
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": CHUNKS},
    )
    sentences = " ".join(f"Finding {n} holds (Smith, 2020)." for n in range(3))
    draft = await seed_draft(db_session, project, user, tiptap_paragraph(sentences))
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="c",
            paper_id=paper.id,
            status="verified",
            evidence_quote="Test scores improved by 12 percent.",
            explanation="ok",
        )
    )

    factory, engine = make_session_factory()
    with (
        patch(
            "app.agents.claim_verification_agent.get_claim_verification_agent",
            return_value=agent,
        ),
        patch.object(
            fulltext_module,
            "drop_reference_and_backmatter_chunks",
            wraps=fulltext_module.drop_reference_and_backmatter_chunks,
        ) as mock_drop,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert len(job.result["verifications"]) == 3
    assert mock_drop.call_count == 1

"""The standalone action for an already-saved draft runs
link -> verify -> finalize with the same code the write job's gated loop uses.
``verify_and_heal_claims`` keeps its name and its existing flat report contract
(every field still describes the raw, unfiltered verification pass -- product decision
D4's old reports are not broken), and adds three keys: ``healed``, ``finalize_stats``,
and ``final_report`` (the "ONE report of the final text", every row verified).
"""

import json
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from app.models.analysis_job import JobType  # noqa: E402
from app.models.draft import Draft  # noqa: E402
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


def _two_paragraph_doc(first: str, second: str) -> dict:
    return {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": first}]},
            {"type": "paragraph", "content": [{"type": "text", "text": second}]},
        ],
    }


@pytest.mark.asyncio
async def test_verify_and_heal_claims_removes_an_unsupported_sentence_from_the_saved_draft(
    db_session,
):
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session, project, title="Cited Work", authors=[{"name": "Jane Smith"}], year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": "Nothing like that here."}],
        },
    )
    draft = await seed_draft(
        db_session, project, user, tiptap_paragraph("Tutoring cures everything (Smith, 2020).")
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Tutoring cures everything (Smith, 2020).",
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
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    # Backward-compatible top-level fields: unchanged in meaning from before this task.
    assert job.result["unsupported_count"] == 1
    assert job.result["verifications"][0]["status"] == "unsupported"
    # Additive healing fields.
    assert job.result["healed"] is True
    assert job.result["finalize_stats"]["sentences_removed_unverified"] == 1
    assert job.result["final_report"]["verified_count"] == 0
    assert job.result["final_report"]["verifications"] == []

    saved = await db_session.get(Draft, draft.id)
    await db_session.refresh(saved)
    assert saved.content["content"] == []


@pytest.mark.asyncio
async def test_verify_and_heal_claims_keeps_a_verified_sentence_and_repairs_its_link(
    db_session,
):
    """A paragraph with no `attrs.citationLinks` at all (a claim `extract_claims_from_
    document` only ever resolved via the regex fallback) keeps its text untouched and
    gains a repaired citationLinks entry -- "save the draft with citationLinks
    repaired" applies even when none existed before."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session, project, title="Cited Work", authors=[{"name": "Jane Smith"}], year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": "Test scores improved by 12%."}],
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
            evidence_quote="Test scores improved by 12%.",
            explanation="Supported.",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["final_report"]["verified_count"] == 1

    saved = await db_session.get(Draft, draft.id)
    await db_session.refresh(saved)
    paragraph = saved.content["content"][0]
    assert paragraph["content"][0]["text"] == "Test scores improved (Smith, 2020)."
    assert paragraph["attrs"]["citationLinks"] == [
        {
            "sentence": "Test scores improved (Smith, 2020).",
            "keys": ["smith_2020"],
            "citation_text": "(Smith, 2020)",
        }
    ]


@pytest.mark.asyncio
async def test_verify_and_heal_claims_final_report_excludes_a_trimmed_unverified_clause(
    db_session,
):
    """A sentence with two
    citations, one verified and one unsupported, joined by a trailing clause with a
    safe excision boundary, is trimmed to its verified clause rather than removed
    whole -- `final_report` must still never list the unsupported claim, since it is
    not in the saved text either."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    good_paper = await seed_paper(
        db_session, project, title="Good Work", authors=[{"name": "Jane Smith"}], year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": "Outcomes improved markedly."}],
        },
    )
    bad_paper = await seed_paper(
        db_session, project, title="Bad Work", authors=[{"name": "Amy Jones"}], year=2019,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": "Nothing relevant here."}],
        },
    )
    sentence = "Outcomes improved markedly (Smith, 2020), while costs fell (Jones, 2019)."
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {
                            "sentence": sentence,
                            "keys": ["smith_2020"],
                            "citation_text": "(Smith, 2020)",
                            "proposition": "Outcomes improved markedly",
                        },
                        {
                            "sentence": sentence,
                            "keys": ["jones_2019"],
                            "citation_text": "(Jones, 2019)",
                            "proposition": "costs fell",
                        },
                    ]
                },
                "content": [{"type": "text", "text": sentence}],
            }
        ],
    }
    draft = await seed_draft(db_session, project, user, doc)
    job = await seed_job(db_session, project, JobType.claim_verify)

    def _route(prompt: str):
        # Each citation link carries its own narrowed `proposition`, so the two
        # claims sent to the verifier are distinct, unambiguous facts, not the same
        # whole compound sentence.
        if "Claim to verify: Outcomes improved markedly" in prompt:
            return ClaimVerification(
                claim_text="x", paper_id=good_paper.id, status="verified",
                evidence_quote="Outcomes improved markedly.", explanation="ok",
            )
        return ClaimVerification(
            claim_text="x", paper_id=bad_paper.id, status="unsupported",
            explanation="Not stated in the source.",
        )

    agent = FakeAgent(output=_route)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    verifications = job.result["final_report"]["verifications"]
    assert all("costs fell" not in v["claim_text"] for v in verifications)
    assert job.result["final_report"]["verified_count"] == len(verifications)

    saved = await db_session.get(Draft, draft.id)
    await db_session.refresh(saved)
    saved_text = json.dumps(saved.content["content"])
    assert "costs fell" not in saved_text
    assert "Outcomes improved markedly" in saved_text


@pytest.mark.asyncio
async def test_verify_and_heal_claims_saves_no_new_version_when_nothing_changes(db_session):
    """A draft whose citationLinks are already exactly what finalize would repair them
    to, with every claim verified, is a true no-op: update_draft never snapshots a
    spurious new version."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session, project, title="Cited Work", authors=[{"name": "Jane Smith"}], year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": "Test scores improved by 12%."}],
        },
    )
    already_repaired = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {
                            "sentence": "Test scores improved (Smith, 2020).",
                            "keys": ["smith_2020"],
                            "citation_text": "(Smith, 2020)",
                        }
                    ]
                },
                "content": [{"type": "text", "text": "Test scores improved (Smith, 2020)."}],
            }
        ],
    }
    draft = await seed_draft(db_session, project, user, already_repaired)
    job = await seed_job(db_session, project, JobType.claim_verify)
    original_version = draft.current_version

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Test scores improved (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote="Test scores improved by 12%.",
            explanation="Supported.",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error

    saved = await db_session.get(Draft, draft.id)
    await db_session.refresh(saved)
    assert saved.current_version == original_version


@pytest.mark.asyncio
async def test_verify_and_heal_claims_grouped_citation_clean_draft_is_a_byte_identical_no_op(
    db_session,
):
    """A grouped citation -- one stored link naming two
    keys, e.g. "(Smith, 2020; Jones, 2019)" -- must heal to the exact same node when
    every claim it covers is verified. `_rebuild_finalized_node_list` driving
    finalize from one synthetic single-key link per claim would split the group into
    two separate one-key links, losing `proposition` and `evidence_ids`, rewriting the
    paragraph's own attrs, and cutting the draft a new version even though nothing
    needed to change -- the "healing a clean draft changes nothing"
    invariant, failing for exactly the shape the citation-link prompt is told to
    produce for a co-cited group."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    smith_chunk = (
        "Multiple randomized controlled trials confirm that tutoring interventions "
        "substantially improve student outcomes across grade levels."
    )
    jones_chunk = (
        "A large scale meta-analysis similarly finds that tutoring interventions "
        "substantially improve student outcomes for struggling learners."
    )
    smith_paper = await seed_paper(
        db_session, project, title="Smith Work", authors=[{"name": "Jane Smith"}], year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": smith_chunk}],
        },
    )
    jones_paper = await seed_paper(
        db_session, project, title="Jones Work", authors=[{"name": "Amy Jones"}], year=2019,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": jones_chunk}],
        },
    )
    sentence = (
        "Tutoring interventions substantially improve student outcomes "
        "(Smith, 2020; Jones, 2019)."
    )
    proposition = "Tutoring interventions substantially improve student outcomes"
    citation_links = [
        {
            "sentence": sentence,
            "keys": ["smith_2020", "jones_2019"],
            "citation_text": "(Smith, 2020; Jones, 2019)",
            "proposition": proposition,
            "evidence_ids": ["e1"],
        }
    ]
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "origin": "ai",
                    "blockId": "blk-1",
                    "citationLinks": citation_links,
                },
                "content": [{"type": "text", "text": sentence}],
            }
        ],
    }
    draft = await seed_draft(db_session, project, user, doc)
    job = await seed_job(db_session, project, JobType.claim_verify)
    original_version = draft.current_version

    def _route(prompt: str):
        if "Paper: Smith Work" in prompt:
            return ClaimVerification(
                claim_text=proposition, paper_id=smith_paper.id, status="verified",
                evidence_quote=(
                    "tutoring interventions substantially improve student outcomes "
                    "across grade levels."
                ),
                explanation="ok",
            )
        return ClaimVerification(
            claim_text=proposition, paper_id=jones_paper.id, status="verified",
            evidence_quote=(
                "tutoring interventions substantially improve student outcomes "
                "for struggling learners."
            ),
            explanation="ok",
        )

    agent = FakeAgent(output=_route)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["finalize_stats"]["sentences_removed_unverified"] == 0
    assert job.result["final_report"]["verified_count"] == 2

    saved = await db_session.get(Draft, draft.id)
    await db_session.refresh(saved)
    assert saved.current_version == original_version
    paragraph = saved.content["content"][0]
    assert paragraph["attrs"]["citationLinks"] == citation_links
    assert paragraph["attrs"]["origin"] == "ai"
    assert paragraph["attrs"]["blockId"] == "blk-1"


@pytest.mark.asyncio
async def test_verify_and_heal_claims_grouped_citation_second_heal_keeps_the_narrowed_claim(
    db_session,
):
    """The regression the review reproduced end to end: healing the same
    grouped-citation paragraph twice must send the same narrowed `proposition` to the
    verifier both times, not fall back to the whole sentence (with its citation text
    still attached) on the second pass because the first heal already dropped it."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    smith_chunk = (
        "Multiple randomized controlled trials confirm that tutoring interventions "
        "substantially improve student outcomes across grade levels."
    )
    jones_chunk = (
        "A large scale meta-analysis similarly finds that tutoring interventions "
        "substantially improve student outcomes for struggling learners."
    )
    smith_paper = await seed_paper(
        db_session, project, title="Smith Work", authors=[{"name": "Jane Smith"}], year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": smith_chunk}],
        },
    )
    jones_paper = await seed_paper(
        db_session, project, title="Jones Work", authors=[{"name": "Amy Jones"}], year=2019,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": jones_chunk}],
        },
    )
    sentence = (
        "Tutoring interventions substantially improve student outcomes "
        "(Smith, 2020; Jones, 2019)."
    )
    proposition = "Tutoring interventions substantially improve student outcomes"
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {
                            "sentence": sentence,
                            "keys": ["smith_2020", "jones_2019"],
                            "citation_text": "(Smith, 2020; Jones, 2019)",
                            "proposition": proposition,
                            "evidence_ids": ["e1"],
                        }
                    ],
                },
                "content": [{"type": "text", "text": sentence}],
            }
        ],
    }
    draft = await seed_draft(db_session, project, user, doc)

    claim_texts_seen: list[str] = []

    def _route(prompt: str):
        claim_texts_seen.append(prompt.split("Claim to verify: ")[1].split("\n")[0])
        if "Paper: Smith Work" in prompt:
            return ClaimVerification(
                claim_text=proposition, paper_id=smith_paper.id, status="verified",
                evidence_quote=(
                    "tutoring interventions substantially improve student outcomes "
                    "across grade levels."
                ),
                explanation="ok",
            )
        return ClaimVerification(
            claim_text=proposition, paper_id=jones_paper.id, status="verified",
            evidence_quote=(
                "tutoring interventions substantially improve student outcomes "
                "for struggling learners."
            ),
            explanation="ok",
        )

    agent = FakeAgent(output=_route)
    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        job1 = await seed_job(db_session, project, JobType.claim_verify)
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job1.id, session_factory=factory,
        )
        job2 = await seed_job(db_session, project, JobType.claim_verify)
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job2.id, session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job1)
    await db_session.refresh(job2)
    assert job1.status.value == "completed", job1.error
    assert job2.status.value == "completed", job2.error
    # Every claim sent to the verifier, across both passes, is the narrowed
    # proposition -- never the whole sentence with its own citation text attached,
    # which is what the second pass sent before this fix.
    assert set(claim_texts_seen) == {proposition}


@pytest.mark.asyncio
async def test_exit_invariant_saved_draft_has_no_unverified_claim_and_no_needs_citation_marker(
    db_session,
):
    """The exit invariant, exercised through the standalone
    action: the saved draft has no claim with status unsupported or needs_nuance, and
    no literal [NEEDS CITATION] text."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    good_paper = await seed_paper(
        db_session, project, title="Good Work", authors=[{"name": "Jane Smith"}], year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": "Outcomes improved markedly."}],
        },
    )
    bad_paper = await seed_paper(
        db_session, project, title="Bad Work", authors=[{"name": "Amy Jones"}], year=2019,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": "Nothing relevant here."}],
        },
    )
    draft = await seed_draft(
        db_session, project, user,
        _two_paragraph_doc(
            "Outcomes improved markedly (Smith, 2020).",
            "This is unproven speculation (Jones, 2019). [NEEDS CITATION]",
        ),
    )
    job = await seed_job(db_session, project, JobType.claim_verify)

    def _route(prompt: str):
        if "Outcomes improved" in prompt:
            return ClaimVerification(
                claim_text="x", paper_id=good_paper.id, status="verified",
                evidence_quote="Outcomes improved markedly.", explanation="ok",
            )
        return ClaimVerification(
            claim_text="x", paper_id=bad_paper.id, status="needs_nuance",
            explanation="Overstated.",
        )

    agent = FakeAgent(output=_route)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id, draft_id=draft.id, job_id=job.id, session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert all(v["status"] == "verified" for v in job.result["final_report"]["verifications"])

    saved = await db_session.get(Draft, draft.id)
    await db_session.refresh(saved)
    flat_text = " ".join(
        n["content"][0]["text"] for n in saved.content["content"] if n.get("content")
    )
    assert "[NEEDS CITATION]" not in flat_text
    assert "unproven speculation" not in flat_text
    assert "Outcomes improved markedly" in flat_text


def test_claim_verification_report_schema_types_the_new_healing_keys():
    """`GET /drafts/{id}/claim-verification` builds `ClaimVerificationReport(**job.result)`
    directly from the job row; the three additive keys are now typed fields, not just
    tolerated extras, and a report from before this task (missing them) still parses."""
    from uuid import uuid4

    from app.schemas.fulltext import ClaimVerificationReport

    report = ClaimVerificationReport(
        draft_id=uuid4(),
        verifications=[],
        verified_count=0,
        unsupported_count=0,
        nuance_count=0,
        abstract_only_count=0,
        healed=True,
        finalize_stats={"sentences_removed_unverified": 1},
        final_report={"verifications": [], "verified_count": 0},
    )
    assert report.healed is True
    assert report.finalize_stats == {"sentences_removed_unverified": 1}
    assert report.final_report.verified_count == 0
    assert report.final_report.verifications == []

    older = ClaimVerificationReport(
        draft_id=uuid4(),
        verifications=[],
        verified_count=0,
        unsupported_count=0,
        nuance_count=0,
        abstract_only_count=0,
    )
    assert older.healed is None
    assert older.finalize_stats is None
    assert older.final_report is None

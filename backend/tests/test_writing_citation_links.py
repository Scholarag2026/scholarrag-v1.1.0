"""The writing job result carries a citation link map.

``generate_section`` calls ``build_citation_link_map`` after the section text is already
generated, in the same try/except pattern as the citation audit: a failure there must
never touch ``content`` or ``citation_audit``, and must leave ``provenance``'s own
generation-call fields exactly as the generation call alone produced them. On success,
the link call's own provenance is folded into ``provenance`` under new keys: the linker
is a second, real DeepSeek call whose tokens must be recorded, not silently dropped.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from app.agents.citation_link_agent import CitationLinkResult  # noqa: E402
from app.schemas.provenance import LLMCallProvenance  # noqa: E402

GENERATED = (
    "Tutoring improves outcomes (Smith, 2020). Brown et al. (2018) concur. "
    "Some claim otherwise (Nguyen, 2021)."
)

PAYLOAD = {
    "id": "chatcmpl-42",
    "model": "deepseek-v4-flash",
    "system_fingerprint": "fp_test",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": GENERATED}}],
    "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500},
}

LINKS = [
    {
        "paragraph_index": 0,
        "sentence": "Tutoring improves outcomes (Smith, 2020).",
        "keys": ["smith_2020"],
        "citation_text": "(Smith, 2020)",
    }
]

LINK_PROVENANCE = LLMCallProvenance(
    agent="citation_link",
    model_configured="deepseek-chat",
    model_reported="deepseek-v4-flash",
    provider_response_id="resp-link-1",
    prompt_version="sha256:abc123abc123",
    input_tokens=910,
    output_tokens=140,
)


@pytest.mark.asyncio
async def test_generate_section_stores_the_validated_citation_link_map(db_session):
    """The mapped claim also has to reach the verifier as "verified"
    for the section to survive finalize with its citation_links unchanged -- otherwise
    the only cited sentence in ``GENERATED`` would be removed. The paper therefore
    carries full text and the claim-verification agent is mocked to verify it, so this
    test still demonstrates its own stated purpose (the link map reaching the job
    result) end to end through the gated write loop.

    ``LINKS`` covers only the first of ``GENERATED``'s three sentences by design (this
    test's own point is the ONE mapped claim, not the other two); the other two are
    themselves a coverage gap (neither linked nor tagged uncited), so
    the gated loop's own coverage-gap step (`_close_citation_link_coverage_gap`) now
    sends them back to the (still-mocked, still ``LINKS``-only) linker once more. That
    retry call itself succeeds -- it simply answers with the same map again, never
    naming either sentence -- so both become ordinary, removable "finding" entries
    (Fix A: a retry call that actually answers is trusted; only a retry call that
    itself fails keeps a gap as "unclassified"). Finalize removes both."""
    from app.models.analysis_job import JobType
    from app.schemas.fulltext import ClaimVerification
    from app.services.writing import generate_section
    from tests.t5_fixtures import (
        FakeAgent,
        make_session_factory,
        seed_draft,
        seed_job,
        seed_paper,
        seed_project,
        seed_user,
    )

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session, project, title="A", authors=[{"name": "Jane Smith"}], year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": "Tutoring improves outcomes."}],
        },
    )
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    verify_agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Tutoring improves outcomes (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote="Tutoring improves outcomes.",
            explanation="Supported.",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM down"),
    ), patch("httpx.AsyncClient.post", new=fake_post), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        return_value=CitationLinkResult(links=LINKS, provenance=LINK_PROVENANCE),
    ) as mocked_linker, patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=verify_agent,
    ):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    # The one mapped, verified sentence keeps its citation link; the other two, never
    # linked or tagged by either the first pass or the coverage-gap retry (both mocked
    # to the same LINKS-only result, which answers but never names either sentence),
    # are removed as ordinary, uncited findings.
    assert job.result["content"] == "Tutoring improves outcomes (Smith, 2020)."
    assert job.result["citation_links"] == [{**LINKS[0], "paragraph_index": 0}]
    assert job.result["loop_stats"]["sentences_unclassified_kept"] == 0
    assert job.result["loop_stats"]["sentences_removed_uncited_finding"] == 2
    assert job.result["loop_stats"]["link_coverage_retry_failed"] is False
    # No revision -- the claim verified first pass. The section still finalizes to
    # one cited sentence, under `THIN_SECTION_CITED_SENTENCE_MINIMUM`, so the
    # empty-section backstop fires a second, independent pass: it re-sends the same
    # generated text (the mocked HTTP transport returns the same payload every
    # time), re-links it against the same LINKS-only mock, and re-verifies the same
    # one claim -- reproducing the identical result, at the cost of one more
    # verification call.
    assert len(verify_agent.calls) == 2
    assert job.result["loop_stats"]["loop_revised"] is False
    assert job.result["loop_stats"]["first_pass_verified_rate"] == 1.0
    assert job.result["claim_report"]["verified_count"] == 1
    assert job.result["loop_stats"]["empty_section_regenerated"] == 1
    assert job.result["loop_stats"]["empty_section_after_regeneration"] is False
    # The section's own first pass, its coverage-gap step's own bounded retry, and
    # the backstop's own regenerated pass together with its own coverage-gap retry.
    first_call_args = mocked_linker.await_args_list[0]
    assert first_call_args.args[0] == GENERATED
    assert set(first_call_args.args[1].keys()) == {"smith_2020"}
    for call_args in mocked_linker.await_args_list[1:]:
        assert call_args.args[0] in (
            "Brown et al. (2018) concur.\n\nSome claim otherwise (Nguyen, 2021).",
            GENERATED,
        )

    provenance = job.result["provenance"]
    assert provenance["input_tokens"] == 1200
    assert provenance["output_tokens"] == 300
    # Four citation-link calls in all -- the first pass, its own coverage-gap
    # retry, then the backstop's own regenerated pass and ITS OWN coverage-gap
    # retry -- every one of them folded into the running totals, none overwriting
    # an earlier one (`citation_link_calls`, the same list shape a revision's own
    # second call already used).
    assert provenance["citation_link_call"] == LINK_PROVENANCE.model_dump(mode="json")
    assert provenance["citation_link_calls"] == [LINK_PROVENANCE.model_dump(mode="json")] * 4
    assert provenance["total_input_tokens"] == 1200 + 1200 + 910 * 4
    assert provenance["total_output_tokens"] == 300 + 300 + 140 * 4


@pytest.mark.asyncio
async def test_citation_link_failure_leaves_content_audit_and_provenance_untouched(db_session):
    """A linker exception is swallowed exactly like a citation-audit exception: the rest
    of the job result is unaffected and the job still completes."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import (
        make_session_factory,
        seed_draft,
        seed_job,
        seed_project,
        seed_user,
    )

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM down"),
    ), patch("httpx.AsyncClient.post", new=fake_post), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        side_effect=RuntimeError("linker exploded"),
    ):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["content"] == GENERATED
    assert job.result["citation_links"] == []
    assert job.result["citation_audit"] == {
        "matched": [],
        "unmatched": ["Smith, 2020", "Brown, 2018", "Nguyen, 2021"],
        "needs_citation_flags": 0,
        "total": 3,
    }
    assert job.result["provenance"]["model_reported"] == "deepseek-v4-flash"
    assert job.result["provenance"]["input_tokens"] == 1200
    assert job.result["provenance"]["output_tokens"] == 300
    assert job.result["provenance"]["citation_link_call"] is None
    assert job.result["provenance"]["total_calls"] == 1
    assert job.result["provenance"]["total_input_tokens"] == 1200
    assert job.result["provenance"]["total_output_tokens"] == 300


@pytest.mark.asyncio
async def test_loop_stats_carries_every_finalize_reason_list(db_session):
    """``generate_section`` builds ``loop_stats`` from ``finalize_generated_section``'s
    own ``FinalizeResult`` (``app.services.fulltext``). That result carries a growing
    set of reason lists -- ``coverage_incomplete_reasons``, ``meta_evaluation_reasons``,
    ``verb_initial_reasons``, ``heading_shaped_reasons`` and ``headings_removed_
    reasons`` at the time of writing -- each naming why a sentence (or heading) under
    the matching ``sentences_removed_*``/``headings_removed`` count was dropped. Every
    one of them has been added to ``FinalizeResult`` at one time or another without
    also being copied into ``loop_stats``, so a reader of the job result (the demo
    summary that carries ``loop_stats`` whole) could see a non-zero count with no
    reason attached at all. Rather than naming the fields by hand -- the same mistake
    that let this happen more than once -- this test discovers every reason-list field
    by introspecting ``FinalizeResult._field_defaults`` (every optional field defaults
    to ``[]``; ``text``, ``citation_links``, ``stats``, ``healed_sentences`` and
    ``uncited_sentences`` are the only required fields and have no entry there), so a
    future reason list ``generate_section`` forgets to thread through fails this test
    without anyone needing to remember to update it by name. This run fires none of
    finalize's removal rules -- the one linked claim is verified and both remaining
    sentences are kept, unclassified -- so every reason list must still be present,
    each as an empty list, not merely absent."""
    from app.models.analysis_job import JobType
    from app.schemas.fulltext import ClaimVerification
    from app.services.writing import generate_section
    from tests.t5_fixtures import (
        FakeAgent,
        make_session_factory,
        seed_draft,
        seed_job,
        seed_paper,
        seed_project,
        seed_user,
    )

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session, project, title="A", authors=[{"name": "Jane Smith"}], year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": "Tutoring improves outcomes."}],
        },
    )
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    verify_agent = FakeAgent(
        output=ClaimVerification(
            claim_text="Tutoring improves outcomes (Smith, 2020).",
            paper_id=paper.id,
            status="verified",
            evidence_quote="Tutoring improves outcomes.",
            explanation="Supported.",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM down"),
    ), patch("httpx.AsyncClient.post", new=fake_post), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        return_value=CitationLinkResult(links=LINKS, provenance=LINK_PROVENANCE),
    ), patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=verify_agent,
    ):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    loop_stats = job.result["loop_stats"]
    from app.services.fulltext import FinalizeResult

    reason_list_fields = list(FinalizeResult._field_defaults)
    assert reason_list_fields, "FinalizeResult has no optional reason-list fields"
    for field_name in reason_list_fields:
        assert field_name in loop_stats, f"{field_name} missing from loop_stats"
        assert loop_stats[field_name] == [], field_name

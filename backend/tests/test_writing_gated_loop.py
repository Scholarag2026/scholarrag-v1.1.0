"""The gated write loop: compose -> link -> verify -> gate;
when at least one claim is not verified (and not the uncheckable "no_full_text") and
regen_count is 0, revise once -- regenerate the whole section, keeping verified
sentences verbatim and re-verifying only the sentences whose text changed -- then
finalize: remove whatever is still not verified, strip [NEEDS CITATION], and save the
finalized section into the draft with its citationLinks repaired.
"""

import os
import uuid

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from app.agents.citation_link_agent import CitationLinkResult  # noqa: E402
from app.schemas.fulltext import ClaimVerification  # noqa: E402
from app.schemas.provenance import LLMCallProvenance  # noqa: E402
from app.services.writing import WritingCompletion  # noqa: E402

FIRST_TEXT = "Tutoring boosts outcomes (Smith, 2020). Tutoring reduces costs (Jones, 2019)."
REVISED_TEXT = "Tutoring boosts outcomes (Smith, 2020). Tutoring improves retention (Jones, 2019)."

LINKS_PASS_1 = [
    {
        "paragraph_index": 0,
        "sentence": "Tutoring boosts outcomes (Smith, 2020).",
        "keys": ["smith_2020"],
        "citation_text": "(Smith, 2020)",
        "proposition": "Tutoring boosts outcomes",
    },
    {
        "paragraph_index": 0,
        "sentence": "Tutoring reduces costs (Jones, 2019).",
        "keys": ["jones_2019"],
        "citation_text": "(Jones, 2019)",
        "proposition": "Tutoring reduces costs",
    },
]

LINKS_PASS_2 = [
    LINKS_PASS_1[0],  # unchanged sentence -- must not be re-verified
    {
        "paragraph_index": 0,
        "sentence": "Tutoring improves retention (Jones, 2019).",
        "keys": ["jones_2019"],
        "citation_text": "(Jones, 2019)",
        "proposition": "Tutoring improves retention",
    },
]

# Deliberately share no words with any claim's own text above, so a substring check for
# "the claim was verified" can never accidentally match inside the chunk text a prompt
# also carries.
SMITH_CHUNK_TEXT = "A randomized trial found significant gains from the intervention."
JONES_CHUNK_TEXT = "Twelve weeks after the programme, follow-up scores had increased."


def _completion(text: str) -> WritingCompletion:
    return WritingCompletion(
        content=text,
        provenance={
            "agent": "writing", "model_configured": "deepseek-chat",
            "model_reported": "deepseek-v4-flash", "temperature": 0.7,
            "prompt_version": "sha256:aaaaaaaaaaaa", "input_tokens": 100,
            "output_tokens": 20, "max_tokens": 4096,
        },
    )


class _RoutingVerifyAgent:
    """Verifies a claim by a substring of its own text; raises on anything
    unexpected, so an accidental extra or missing call fails loudly rather than
    silently passing."""

    def __init__(self):
        self.calls: list[str] = []

    async def run(self, prompt, deps=None):
        self.calls.append(prompt)
        if "Claim to verify: Tutoring boosts outcomes" in prompt:
            status, quote = "verified", SMITH_CHUNK_TEXT
        elif "Claim to verify: Tutoring reduces costs" in prompt:
            status, quote = "unsupported", None
        elif "Claim to verify: Tutoring improves retention" in prompt:
            status, quote = "verified", JONES_CHUNK_TEXT
        else:
            raise AssertionError(f"unexpected claim prompt: {prompt[:300]}")
        return type(
            "R",
            (),
            {
                "output": ClaimVerification(
                    claim_text="placeholder",
                    paper_id=uuid.uuid4(),
                    status=status,
                    evidence_quote=quote,
                    explanation="Checked against full text.",
                )
            },
        )()


async def _seed_world(db_session):
    from tests.t5_fixtures import seed_paper, seed_project, seed_user

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(
        db_session, project, title="Smith work", authors=[{"name": "Jane Smith"}], year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": SMITH_CHUNK_TEXT}],
        },
    )
    await seed_paper(
        db_session, project, title="Jones work", authors=[{"name": "Amy Jones"}], year=2019,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "results", "text": JONES_CHUNK_TEXT}],
        },
    )
    return user, project


@pytest.mark.asyncio
async def test_gate_triggers_one_revision_and_reuses_unchanged_verdicts(db_session):
    """The gate's own revision leaves exactly two verified, cited sentences -- still
    under `THIN_SECTION_CITED_SENTENCE_MINIMUM`, so the empty-section regeneration
    backstop fires next, independently of the gate: it re-sends the ORIGINAL
    (pre-revision) prompt, not the gate's own revised text, so it reproduces
    `FIRST_TEXT` and re-links it against `LINKS_PASS_1` -- Jones's own "reduces
    costs" claim is still unsupported there, so only Smith's sentence survives this
    second, independent pass. The backstop must never make the section worse: the
    first attempt (the gate's own two verified sentences) beats the regenerated
    attempt's own one, so the FIRST attempt is the one that ships
    (`loop_stats["regeneration_kept"] == "first"`), even though a second attempt
    did run."""
    from app.models.analysis_job import JobType
    from app.models.draft import Draft
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    verify_agent = _RoutingVerifyAgent()

    async def fake_call_deepseek(system_prompt, user_prompt):
        return _completion(FIRST_TEXT if "REVISION REQUIRED" not in user_prompt else REVISED_TEXT)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        side_effect=[
            CitationLinkResult(links=LINKS_PASS_1, provenance=None),
            CitationLinkResult(links=LINKS_PASS_2, provenance=None),
            # The empty-section regeneration backstop's own third call, re-linking
            # the re-sent `FIRST_TEXT` -- see the docstring above.
            CitationLinkResult(links=LINKS_PASS_1, provenance=None),
        ],
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
    result = job.result

    # Gate fired once: revision happened, and exactly one revision (never a loop).
    assert result["loop_stats"]["loop_revised"] is True
    assert result["loop_stats"]["first_pass_verified_rate"] == 0.5

    # The gate's own two passes verify Smith once (unchanged, reused) and Jones once
    # per pass (three calls); the backstop's own regenerated pass re-verifies both
    # of `FIRST_TEXT`'s own claims from scratch -- it has no reuse cache of its own
    # -- for five calls in total.
    assert len(verify_agent.calls) == 5
    assert sum("Claim to verify: Tutoring boosts outcomes" in p for p in verify_agent.calls) == 2
    assert sum("Claim to verify: Tutoring reduces costs" in p for p in verify_agent.calls) == 2
    assert sum("Claim to verify: Tutoring improves retention" in p for p in verify_agent.calls) == 1

    # The first attempt survives finalize with two verified sentences, the
    # regenerated attempt with only one -- the first attempt wins and ships,
    # exactly as it would have without the backstop firing at all.
    assert result["content"] == REVISED_TEXT
    assert result["claim_report"]["verified_count"] == 2
    assert all(v["status"] == "verified" for v in result["claim_report"]["verifications"])
    assert "[NEEDS CITATION]" not in result["content"]
    assert result["loop_stats"]["empty_section_regenerated"] == 1
    assert result["loop_stats"]["regeneration_kept"] == "first"
    assert result["loop_stats"]["empty_section_after_regeneration"] is False

    # The draft itself was saved, with a heading tagged by section_type and a
    # paragraph carrying the repaired citationLinks.
    saved = await db_session.get(Draft, draft.id)
    await db_session.refresh(saved)
    nodes = saved.content["content"]
    assert nodes[0]["type"] == "heading"
    assert nodes[0]["attrs"]["sectionType"] == "methods"
    paragraph = next(n for n in nodes if n["type"] == "paragraph")
    assert paragraph["content"][0]["text"] == REVISED_TEXT
    saved_keys = {
        key for link in paragraph["attrs"]["citationLinks"] for key in link["keys"]
    }
    assert saved_keys == {"smith_2020", "jones_2019"}


FIRST_TEXT_WITH_FINDING = (
    f"{FIRST_TEXT} Teachers reported broad gains across the whole cohort."
)
REVISED_TEXT_WITHOUT_FINDING = (
    f"{REVISED_TEXT} Engagement rose noticeably during the programme."
)

LINKS_PASS_1_WITH_FINDING = LINKS_PASS_1
UNCITED_PASS_1_WITH_FINDING = [
    {
        "paragraph_index": 0,
        "sentence": "Teachers reported broad gains across the whole cohort.",
        "tag": "finding",
    }
]
LINKS_PASS_2_WITH_FINDING = LINKS_PASS_2
UNCITED_PASS_2_WITH_FINDING = [
    {
        "paragraph_index": 0,
        "sentence": "Engagement rose noticeably during the programme.",
        "tag": "framing",
    }
]


@pytest.mark.asyncio
async def test_revision_prompt_names_the_uncited_finding_sentence_finalize_will_remove(
    db_session,
):
    """The revise prompt must carry every sentence
    `finalize_generated_section` is about to remove -- not only the unverified cited
    ones the gate itself found, but the uncited "finding" sentence the FIRST pass's own
    citation-link call already flagged -- plus an instruction to keep the surrounding
    paragraph coherent without it, so a surviving framing sentence never keeps pointing
    at material the model is about to lose."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    verify_agent = _RoutingVerifyAgent()
    revision_prompts: list[str] = []

    async def fake_call_deepseek(system_prompt, user_prompt):
        if "REVISION REQUIRED" in user_prompt:
            revision_prompts.append(user_prompt)
            return _completion(REVISED_TEXT_WITHOUT_FINDING)
        return _completion(FIRST_TEXT_WITH_FINDING)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        side_effect=[
            CitationLinkResult(
                links=LINKS_PASS_1_WITH_FINDING,
                provenance=None,
                uncited_sentences=UNCITED_PASS_1_WITH_FINDING,
            ),
            CitationLinkResult(
                links=LINKS_PASS_2_WITH_FINDING,
                provenance=None,
                uncited_sentences=UNCITED_PASS_2_WITH_FINDING,
            ),
        ],
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

    assert len(revision_prompts) == 1
    prompt = revision_prompts[0]
    assert (
        "These sentences state an empirical finding with no citation and will be "
        "removed:" in prompt
    )
    assert '"Teachers reported broad gains across the whole cohort."' in prompt
    assert "rewrite the paragraph it belongs to so the surrounding text reads " in prompt
    assert "Add no new claim while doing this." in prompt
    # No unclassified entry on this run, so that section must not appear at all.
    assert "not recognised as either a cited claim" not in prompt


FIRST_LINK_PROVENANCE = LLMCallProvenance(
    agent="citation_link", model_configured="deepseek-chat", model_reported="deepseek-v4-flash",
    provider_response_id="resp-link-first", prompt_version="sha256:abc123abc123",
    input_tokens=900, output_tokens=130,
)
SECOND_LINK_PROVENANCE = LLMCallProvenance(
    agent="citation_link", model_configured="deepseek-chat", model_reported="deepseek-v4-flash",
    provider_response_id="resp-link-second", prompt_version="sha256:abc123abc123",
    input_tokens=950, output_tokens=145,
)


THIRD_LINK_PROVENANCE_BACKSTOP = LLMCallProvenance(
    agent="citation_link", model_configured="deepseek-chat", model_reported="deepseek-v4-flash",
    provider_response_id="resp-link-backstop", prompt_version="sha256:abc123abc123",
    input_tokens=90, output_tokens=15,
)


@pytest.mark.asyncio
async def test_a_revisions_second_citation_link_call_does_not_erase_the_firsts_record(
    db_session,
):
    """A revision pass calls
    ``build_citation_link_map`` a second time. ``citation_link_calls`` must keep both
    calls, in order, so neither call's own response id, fingerprint, and token split is
    lost even though both are real, paid DeepSeek calls, and the folded
    ``total_input_tokens``/``total_output_tokens`` must cover both. The gate's own
    revision still leaves only two cited sentences (Smith's, unchanged, plus
    Jones's revised one), under `THIN_SECTION_CITED_SENTENCE_MINIMUM`, so the
    empty-section backstop fires a THIRD citation-link call next -- proving the
    same "no call's record is lost" guarantee across three calls, not only two."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    verify_agent = _RoutingVerifyAgent()

    async def fake_call_deepseek(system_prompt, user_prompt):
        return _completion(FIRST_TEXT if "REVISION REQUIRED" not in user_prompt else REVISED_TEXT)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        side_effect=[
            CitationLinkResult(links=LINKS_PASS_1, provenance=FIRST_LINK_PROVENANCE),
            CitationLinkResult(links=LINKS_PASS_2, provenance=SECOND_LINK_PROVENANCE),
            CitationLinkResult(links=LINKS_PASS_1, provenance=THIRD_LINK_PROVENANCE_BACKSTOP),
        ],
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
    provenance = job.result["provenance"]

    assert provenance["citation_link_calls"] == [
        FIRST_LINK_PROVENANCE.model_dump(mode="json"),
        SECOND_LINK_PROVENANCE.model_dump(mode="json"),
        THIRD_LINK_PROVENANCE_BACKSTOP.model_dump(mode="json"),
    ]
    # The single scalar key keeps its established meaning: the most recent call.
    assert provenance["citation_link_call"] == THIRD_LINK_PROVENANCE_BACKSTOP.model_dump(
        mode="json"
    )
    # All three calls' tokens are folded into the running totals, alongside the
    # initial generation call (100 in / 20 out), the revision's own regeneration
    # call (another 100 in / 20 out) and the backstop's own regeneration call (a
    # third 100 in / 20 out, ``_completion`` returning the same fixed numbers for
    # every text) -- six DeepSeek calls' tokens in total, not two.
    assert provenance["total_input_tokens"] == 100 + 900 + 100 + 950 + 100 + 90
    assert provenance["total_output_tokens"] == 20 + 130 + 20 + 145 + 20 + 15


REVISED_TEXT_WITH_ORPHAN = (
    "Tutoring boosts outcomes (Smith, 2020). Tutoring improves retention (Jones, 2019). "
    "Uncontrolled trials showed similar patterns as controls."
)
ORPHAN_SENTENCE = "Uncontrolled trials showed similar patterns as controls."

THIRD_LINK_PROVENANCE = LLMCallProvenance(
    agent="citation_link", model_configured="deepseek-chat", model_reported="deepseek-v4-flash",
    provider_response_id="resp-link-coverage-gap", prompt_version="sha256:abc123abc123",
    input_tokens=40, output_tokens=10,
)


FOURTH_LINK_PROVENANCE_BACKSTOP = LLMCallProvenance(
    agent="citation_link", model_configured="deepseek-chat", model_reported="deepseek-v4-flash",
    provider_response_id="resp-link-backstop-2", prompt_version="sha256:abc123abc123",
    input_tokens=95, output_tokens=18,
)


@pytest.mark.asyncio
async def test_the_revisions_own_orphaned_sentence_is_closed_and_kept(db_session):
    """The orphaned sentence comes from the REVISION's own citation-link call, not the
    section's first pass. This test reproduces that shape directly: the revision's own
    ``build_citation_link_map`` call returns a map that covers neither of its own two
    real citations' sentences nor the new third sentence it also introduces, which
    carries no citation at all. The coverage-gap step's own retry call (the third
    `build_citation_link_map` call) itself succeeds -- it simply answers with nothing,
    never naming that third sentence either -- so it is removed as an ordinary,
    uncited finding, counted under ``sentences_removed_uncited_finding`` -- proving the
    fix runs on the revision's own citation-link call, not only the section's first
    pass. That leaves two cited sentences, still under `THIN_SECTION_CITED_SENTENCE_
    MINIMUM`, so the empty-section backstop fires a fourth citation-link call next,
    re-sending the ORIGINAL prompt: Jones's own "reduces costs" claim is unsupported
    there, so the regenerated attempt finalizes to only Smith's one sentence -- worse
    than the first attempt's own two, so the backstop keeps the FIRST attempt
    instead (`loop_stats["regeneration_kept"] == "first"`)."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    verify_agent = _RoutingVerifyAgent()

    async def fake_call_deepseek(system_prompt, user_prompt):
        return _completion(
            FIRST_TEXT if "REVISION REQUIRED" not in user_prompt else REVISED_TEXT_WITH_ORPHAN
        )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        side_effect=[
            CitationLinkResult(links=LINKS_PASS_1, provenance=FIRST_LINK_PROVENANCE),
            CitationLinkResult(links=LINKS_PASS_2, provenance=SECOND_LINK_PROVENANCE),
            CitationLinkResult(links=[], uncited_sentences=[], provenance=THIRD_LINK_PROVENANCE),
            CitationLinkResult(links=LINKS_PASS_1, provenance=FOURTH_LINK_PROVENANCE_BACKSTOP),
        ],
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
    result = job.result

    assert result["loop_stats"]["loop_revised"] is True
    # The orphaned sentence was removed on the gate's own revision pass -- the
    # coverage-gap retry answered but never named it, so it was an ordinary uncited
    # finding, not kept under the transient-failure statistic -- before the
    # empty-section backstop replaced the whole result with its own, independent
    # pass (see the docstring above).
    assert result["loop_stats"]["sentences_unclassified_kept"] == 0
    assert result["loop_stats"]["link_coverage_retry_failed"] is False
    assert result["loop_stats"]["empty_section_regenerated"] == 1
    assert result["loop_stats"]["regeneration_kept"] == "first"
    assert result["loop_stats"]["empty_section_after_regeneration"] is False

    # The first attempt -- the gate's own revised, orphan-closed text -- beats the
    # regenerated attempt's own single sentence and ships instead.
    assert result["content"] == REVISED_TEXT
    assert ORPHAN_SENTENCE not in result["content"]

    # All four `build_citation_link_map` calls -- first pass, revision, the
    # revision's own coverage-gap retry, and the backstop's own regeneration --
    # are folded into the section's running provenance in order, none overwriting
    # an earlier one.
    provenance = result["provenance"]
    assert len(provenance["citation_link_calls"]) == 4
    assert provenance["citation_link_calls"][-1] == FOURTH_LINK_PROVENANCE_BACKSTOP.model_dump(
        mode="json"
    )


MIXED_SENTENCE_TEXT = "Tutoring boosts outcomes (Smith, 2020), while costs fall (Jones, 2019)."

MIXED_LINKS = [
    {
        "paragraph_index": 0,
        "sentence": MIXED_SENTENCE_TEXT,
        "keys": ["smith_2020"],
        "citation_text": "(Smith, 2020)",
        "proposition": "Tutoring boosts outcomes",
    },
    {
        "paragraph_index": 0,
        "sentence": MIXED_SENTENCE_TEXT,
        "keys": ["jones_2019"],
        "citation_text": "(Jones, 2019)",
        "proposition": "costs fall",
    },
]


@pytest.mark.asyncio
async def test_final_report_excludes_an_unverified_claim_trimmed_by_excision(db_session):
    """A sentence carrying one verified and one unsupported citation, joined by a
    trailing clause with a safe excision boundary, is trimmed to its verified clause
    rather than removed whole. The claim_report must still never carry the excised,
    unsupported claim -- consistency between the delivered text and the final report
    is the invariant this test protects, not whole-sentence removal specifically.
    The trimmed result carries only one cited sentence, under `THIN_SECTION_CITED_
    SENTENCE_MINIMUM`, so the empty-section backstop fires a third pass; it re-sends
    the same text and re-excises the same clause, reproducing the identical result,
    so every assertion below still holds."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    class _MixedVerifyAgent(_RoutingVerifyAgent):
        async def run(self, prompt, deps=None):
            self.calls.append(prompt)
            if "Claim to verify: Tutoring boosts outcomes" in prompt:
                status, quote = "verified", SMITH_CHUNK_TEXT
            elif "Claim to verify: costs fall" in prompt:
                status, quote = "unsupported", None
            else:
                raise AssertionError(f"unexpected claim prompt: {prompt[:300]}")
            return type(
                "R", (),
                {"output": ClaimVerification(
                    claim_text="placeholder", paper_id=uuid.uuid4(), status=status,
                    evidence_quote=quote, explanation="Checked against full text.",
                )},
            )()

    verify_agent = _MixedVerifyAgent()

    async def fake_call_deepseek(system_prompt, user_prompt):
        # The revision reproduces the same text -- the point here is finalize's own
        # rule 2, not the revision loop.
        return _completion(MIXED_SENTENCE_TEXT)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        side_effect=[
            CitationLinkResult(links=MIXED_LINKS, provenance=None),
            CitationLinkResult(links=MIXED_LINKS, provenance=None),
            # The empty-section backstop's own third pass -- see the docstring above.
            CitationLinkResult(links=MIXED_LINKS, provenance=None),
        ],
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
    result = job.result

    assert "costs fall" not in result["content"]
    assert "Jones" not in result["content"]
    assert "Tutoring boosts outcomes (Smith, 2020)." in result["content"]
    verifications = result["claim_report"]["verifications"]
    assert all("costs fall" not in v["claim_text"] for v in verifications)
    assert result["claim_report"]["verified_count"] == len(verifications)
    assert result["loop_stats"]["empty_section_regenerated"] == 1


@pytest.mark.asyncio
async def test_a_failed_link_call_leaves_the_final_report_null_not_falsely_all_verified(
    db_session,
):
    """When the citation-link call fails, `citation_links` is
    `None`, so `new_claims`/`claim_status` are empty and finalize removes nothing --
    the exit invariant is silently not enforced on this path. The report must say so
    (`claim_report` null, `loop_stats["link_map_failed"]` true) rather than claim an
    empty, all-verified set."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    async def fake_call_deepseek(system_prompt, user_prompt):
        return _completion(FIRST_TEXT)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock, side_effect=RuntimeError("linker down"),
    ):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    result = job.result

    assert result["claim_report"] is None
    assert result["loop_stats"]["link_map_failed"] is True
    assert result["content"] == FIRST_TEXT


@pytest.mark.asyncio
async def test_exit_invariant_no_unsupported_survives_a_failed_revision(db_session):
    """When the revision still leaves a claim unverified, finalize removes it -- the
    exit invariant: the saved draft carries no claim whose
    final status is unsupported or needs_nuance, and no [NEEDS CITATION] text. The
    result carries only one cited sentence, under `THIN_SECTION_CITED_SENTENCE_
    MINIMUM`, so the empty-section backstop fires a third pass; it re-sends the
    same text and Jones's claim is unsupported again there too, reproducing the
    identical one-sentence result, so the exit invariant this test protects still
    holds on the section that actually ships."""
    from app.models.analysis_job import JobType
    from app.models.draft import Draft
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    class _StillFailingAgent(_RoutingVerifyAgent):
        async def run(self, prompt, deps=None):
            if "Claim to verify: Tutoring reduces costs" in prompt:
                self.calls.append(prompt)
                return type(
                    "R", (),
                    {"output": ClaimVerification(
                        claim_text="x", paper_id=uuid.uuid4(), status="unsupported",
                        explanation="Still not supported.",
                    )},
                )()
            return await super().run(prompt, deps)

    verify_agent = _StillFailingAgent()

    # The revision reproduces the SAME (unfixed) Jones sentence, so it is not even a
    # new (sentence, key) pair -- it is re-sent because its text is identical to the
    # unresolved one, proving finalize (not the reuse rule) is what removes it here.
    async def fake_call_deepseek(system_prompt, user_prompt):
        return _completion(FIRST_TEXT)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        side_effect=[
            CitationLinkResult(links=LINKS_PASS_1, provenance=None),
            CitationLinkResult(links=LINKS_PASS_1, provenance=None),
            # The empty-section backstop's own third pass -- see the docstring above.
            CitationLinkResult(links=LINKS_PASS_1, provenance=None),
        ],
    ), patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=verify_agent,
    ):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory,
            # target_words=8 (hard_maximum=10) is
            # exactly FIRST_TEXT's own word count, so the pre-revision length check
            # below does not itself trigger a regeneration -- the only thing that
            # shortens the saved text here is finalize dropping the still-unsupported
            # sentence.
            target_words=8,
        )
    await engine.dispose()

    await db_session.refresh(job)
    result = job.result

    assert result["loop_stats"]["loop_revised"] is True
    assert result["loop_stats"]["empty_section_regenerated"] == 1
    assert result["loop_stats"]["empty_section_after_regeneration"] is False
    # Jones's sentence text is byte-identical across the gate's own two passes, so
    # it is reused there (not re-verified); the backstop's own third pass has no
    # reuse cache of its own, so it re-verifies Jones's claim once more -- two
    # calls in total, not one.
    assert sum("Claim to verify: Tutoring reduces costs" in p for p in verify_agent.calls) == 2
    # All three `build_citation_link_map` calls -- first pass, revision, and the
    # backstop's own regeneration -- are kept, in order, instead of a later one
    # overwriting an earlier one.
    assert result["provenance"]["citation_link_calls"] == [None, None, None]

    assert result["content"] == "Tutoring boosts outcomes (Smith, 2020)."
    assert not any(v["status"] != "verified" for v in result["claim_report"]["verifications"])

    # ``length.words`` is the pre-revision,
    # pre-finalize generation call's own word count (FIRST_TEXT, 10 words) -- a draft
    # that was then partly thrown away. ``length.final_words`` is the length of the
    # text actually saved (5 words, the Jones sentence dropped by finalize), the one to
    # quote as "the finished length".
    length = result["provenance"]["length"]
    assert length["words"] == 10
    assert length["final_words"] == 5
    assert "[NEEDS CITATION]" not in result["content"]
    assert result["loop_stats"]["sentences_removed_unverified"] == 1

    saved = await db_session.get(Draft, draft.id)
    await db_session.refresh(saved)
    flat_text = " ".join(
        n["content"][0]["text"]
        for n in saved.content["content"]
        if n["type"] == "paragraph"
    )
    assert "[NEEDS CITATION]" not in flat_text
    assert "Tutoring reduces costs" not in flat_text


@pytest.mark.asyncio
async def test_generate_section_reports_the_locked_block_stripped_uncited_list(db_session):
    """The job result's own `uncited_sentences` must
    be the locked-block-stripped list finalize actually acted on, not the raw list a
    reader would then see finalize deliberately never removed (the locked-block
    exemption)."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    text = "Locked fact stands alone. Model added finding here."
    raw_uncited = [
        {"paragraph_index": 0, "sentence": "Locked fact stands alone.", "tag": "finding"},
        {"paragraph_index": 0, "sentence": "Model added finding here.", "tag": "finding"},
    ]

    async def fake_call_deepseek(system_prompt, user_prompt):
        return _completion(text)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        return_value=CitationLinkResult(links=[], provenance=None, uncited_sentences=raw_uncited),
    ):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory,
            locked_blocks=[{"position": 0, "text": "Locked fact stands alone."}],
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    result = job.result

    # Finalize kept the locked sentence and removed the model's own uncited finding.
    assert result["content"] == "Locked fact stands alone."
    # The reported list is the stripped one -- the locked sentence's own entry is
    # gone, not just its text -- never the raw list finalize was never handed.
    assert result["uncited_sentences"] == [
        {"paragraph_index": 0, "sentence": "Model added finding here.", "tag": "finding"},
    ]


BOLD_CLAIM_SENTENCE = (
    "**Direct corrections outperformed metalinguistic codes by 32% (Smith, 2020).**"
)
BOLD_CLAIM_TEXT = f"Framing prose opens the paragraph.\n\n{BOLD_CLAIM_SENTENCE}"
# This whole-line bold run ends in sentence punctuation and
# carries a citation, so `_is_bold_heading_candidate` reads it as a claim sentence, not
# a sub-heading -- `_strip_inline_emphasis` therefore strips its "**" markers before the
# (real) citation-link call ever sees it. The mocked link's own ``sentence`` mirrors
# that stripped text, exactly as a real linker call -- which only ever echoes back text
# actually present in what it was given -- would.
BOLD_CLAIM_SENTENCE_STRIPPED = (
    "Direct corrections outperformed metalinguistic codes by 32% (Smith, 2020)."
)
BOLD_CLAIM_LINKS = [
    {
        "paragraph_index": 1,
        "sentence": BOLD_CLAIM_SENTENCE_STRIPPED,
        "keys": ["smith_2020"],
        "citation_text": "(Smith, 2020)",
    }
]
BOLD_CLAIM_UNCITED = [
    {"paragraph_index": 0, "sentence": "Framing prose opens the paragraph.", "tag": "framing"},
]


class _AlwaysUnsupportedAgent:
    """Every claim comes back "unsupported", regardless of prompt -- the review's own
    probe needs no other verdict: the bold sentence must never survive by accident of a
    verifier that happened to verify it."""

    async def run(self, prompt, deps=None):
        return type(
            "R",
            (),
            {
                "output": ClaimVerification(
                    claim_text="placeholder",
                    paper_id=uuid.uuid4(),
                    status="unsupported",
                    evidence_quote=None,
                    explanation="No supporting statistic found.",
                )
            },
        )()


@pytest.mark.asyncio
async def test_a_bold_claim_with_a_citation_is_removed_when_unsupported_not_kept_as_a_heading(
    db_session,
):
    """Run through the real write path
    (compose -> link -> verify -> gate -> revise -> finalize -> save): a whole-line bold
    run that carries a citation and ends in sentence punctuation is a claim sentence the
    writer happened to bold, not a sub-heading. Otherwise it would be exempt from every
    verification rule -- kept verbatim, its link dropped from ``surviving_links`` (no row
    in the final report), and saved as a level-3 heading node no later heal could ever
    reach. The model's "revision" changes nothing (a persistent, unfixable claim), so the
    gate fires exactly once and finalize still has the final say."""
    from app.models.analysis_job import JobType
    from app.models.draft import Draft
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    async def fake_call_deepseek(system_prompt, user_prompt):
        return _completion(BOLD_CLAIM_TEXT)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        return_value=CitationLinkResult(
            links=BOLD_CLAIM_LINKS, provenance=None, uncited_sentences=BOLD_CLAIM_UNCITED,
        ),
    ), patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=_AlwaysUnsupportedAgent(),
    ):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    result = job.result

    # The gate fired (one unsupported claim); finalize still removed it from the
    # revision's own output, exactly as it would have from the first pass.
    assert result["loop_stats"]["loop_revised"] is True
    assert "Direct corrections outperformed" not in result["content"]
    assert result["content"] == "Framing prose opens the paragraph."
    assert result["citation_links"] == []
    assert result["claim_report"]["verified_count"] == 0

    saved = await db_session.get(Draft, draft.id)
    await db_session.refresh(saved)
    nodes = saved.content["content"]
    # No level-3 (or any) heading node was ever created for the bolded claim -- there
    # is nothing left to render as one, since finalize already removed it.
    assert not any(
        node["type"] == "heading" and node.get("attrs", {}).get("level") == 3
        for node in nodes
    )
    paragraphs = [n for n in nodes if n["type"] == "paragraph"]
    assert len(paragraphs) == 1
    assert paragraphs[0]["content"][0]["text"] == "Framing prose opens the paragraph."


# --- Empty-section regeneration backstop ----------------------------------------------

REGEN_TEXT_1 = "Tutoring boosts outcomes (Smith, 2020)."
REGEN_TEXT_2 = "Tutoring boosts outcomes for most learners (Smith, 2020)."

REGEN_LINKS_2 = [{
    "paragraph_index": 0,
    "sentence": REGEN_TEXT_2,
    "keys": ["smith_2020"],
    "citation_text": "(Smith, 2020)",
    "proposition": REGEN_TEXT_2,
}]

REGEN_PROVENANCE_1 = LLMCallProvenance(
    agent="citation_link", model_configured="deepseek-chat", model_reported="deepseek-v4-flash",
    provider_response_id="resp-regen-link-1", prompt_version="sha256:abc123abc123",
    input_tokens=30, output_tokens=8,
)
REGEN_PROVENANCE_2 = LLMCallProvenance(
    agent="citation_link", model_configured="deepseek-chat", model_reported="deepseek-v4-flash",
    provider_response_id="resp-regen-link-2", prompt_version="sha256:abc123abc123",
    input_tokens=30, output_tokens=8,
)


@pytest.mark.asyncio
async def test_empty_section_regenerates_once_when_the_second_attempt_finds_a_link(
    db_session,
):
    """The first attempt's own citation-link call never names the sentence's real
    citation (an uncited "finding" entry instead, which finalize's own rule 1
    removes), leaving the section with a real citation audit but zero cited
    sentences -- comfortably under `THIN_SECTION_CITED_SENTENCE_MINIMUM`, the
    empty-section regeneration backstop's own trigger condition ("fewer than three
    distinct cited sentences survive finalize", not only the literal-zero case).
    The second attempt's own citation-link call names it correctly this time, so
    the regenerated section survives finalize with one verified, cited sentence,
    beating the first attempt's own zero -- the SECOND attempt is the one kept
    (`loop_stats["regeneration_kept"] == "second"`) -- and there is no third
    attempt."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    verify_agent = _RoutingVerifyAgent()
    call_count = {"n": 0}

    async def fake_call_deepseek(system_prompt, user_prompt):
        call_count["n"] += 1
        return _completion(REGEN_TEXT_1 if call_count["n"] == 1 else REGEN_TEXT_2)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        side_effect=[
            CitationLinkResult(
                links=[], provenance=REGEN_PROVENANCE_1,
                uncited_sentences=[{
                    "paragraph_index": 0, "sentence": REGEN_TEXT_1, "tag": "finding",
                }],
            ),
            CitationLinkResult(links=REGEN_LINKS_2, provenance=REGEN_PROVENANCE_2),
        ],
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
    result = job.result

    assert result["loop_stats"]["empty_section_regenerated"] == 1
    assert result["loop_stats"]["regeneration_kept"] == "second"
    assert result["loop_stats"]["empty_section_after_regeneration"] is False
    assert result["content"] == REGEN_TEXT_2
    assert result["citation_links"] == [{**REGEN_LINKS_2[0], "paragraph_index": 0}]
    assert result["claim_report"]["verified_count"] == 1
    assert call_count["n"] == 2  # the regenerated attempt's own write call, never a third

    provenance = result["provenance"]
    assert provenance["regeneration_calls"] == [
        {
            "agent": "writing", "model_configured": "deepseek-chat",
            "model_reported": "deepseek-v4-flash", "temperature": 0.7,
            "prompt_version": "sha256:aaaaaaaaaaaa", "input_tokens": 100,
            "output_tokens": 20, "max_tokens": 4096,
        }
    ]
    assert provenance["citation_link_calls"] == [
        REGEN_PROVENANCE_1.model_dump(mode="json"), REGEN_PROVENANCE_2.model_dump(mode="json"),
    ]


@pytest.mark.asyncio
async def test_empty_section_ships_as_is_when_the_regenerated_attempt_is_also_empty(
    db_session,
):
    """The second attempt's own citation-link call is exactly as unhelpful as the
    first's -- never names the sentence's real citation either -- so the regenerated
    section is also finalized to zero cited sentences, still under `THIN_SECTION_
    CITED_SENTENCE_MINIMUM`. Zero and zero is a tie, so the FIRST attempt is the
    one kept (`loop_stats["regeneration_kept"] == "first"`); the two are equally
    empty here, so the delivered text does not itself change. There is no third
    attempt. `empty_section_after_regeneration` stays tied to the literal zero bar
    (rule 14, `demo/check_delivered.py`, treats it as an unconditional violation),
    which this literally-empty case still crosses regardless of the trigger's own
    wider bound."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    call_count = {"n": 0}

    async def fake_call_deepseek(system_prompt, user_prompt):
        call_count["n"] += 1
        return _completion(REGEN_TEXT_1 if call_count["n"] == 1 else REGEN_TEXT_2)

    def _no_link_result(*args, **kwargs):
        return CitationLinkResult(
            links=[], provenance=None,
            uncited_sentences=[{
                "paragraph_index": 0,
                "sentence": REGEN_TEXT_1 if call_count["n"] == 1 else REGEN_TEXT_2,
                "tag": "finding",
            }],
        )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock, side_effect=_no_link_result,
    ), patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=_RoutingVerifyAgent(),
    ):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    result = job.result

    assert result["loop_stats"]["empty_section_regenerated"] == 1
    assert result["loop_stats"]["regeneration_kept"] == "first"
    assert result["loop_stats"]["empty_section_after_regeneration"] is True
    assert result["content"] == ""
    assert result["citation_links"] == []
    assert call_count["n"] == 2  # exactly one regeneration, never a third attempt
    assert result["provenance"]["regeneration_calls"]
    assert len(result["provenance"]["regeneration_calls"]) == 1


REGEN_TWO_SENTENCE_TEXT = (
    "Tutoring boosts outcomes (Smith, 2020). Tutoring improves retention (Jones, 2019)."
)

REGEN_TWO_SENTENCE_LINKS = [
    {
        "paragraph_index": 0,
        "sentence": "Tutoring boosts outcomes (Smith, 2020).",
        "keys": ["smith_2020"],
        "citation_text": "(Smith, 2020)",
        "proposition": "Tutoring boosts outcomes",
    },
    {
        "paragraph_index": 0,
        "sentence": "Tutoring improves retention (Jones, 2019).",
        "keys": ["jones_2019"],
        "citation_text": "(Jones, 2019)",
        "proposition": "Tutoring improves retention",
    },
]


@pytest.mark.asyncio
async def test_thin_two_sentence_section_still_regenerates_with_nothing_unverified(
    db_session,
):
    """The backstop's own new bound, not only the literal-zero case the two tests
    above exercise: both of the first attempt's own sentences are cited, correctly
    linked and fully verified -- nothing unsupported, no citation-link failure, no
    uncited finding for finalize's own rule 1 to remove -- yet finalize leaves only
    two distinct cited sentences, one under `THIN_SECTION_CITED_SENTENCE_MINIMUM`,
    so the section still regenerates once. The regenerated attempt reproduces the
    same two verified sentences -- a tie, so the FIRST attempt is the one kept
    (`loop_stats["regeneration_kept"] == "first"`), though the two are identical
    here so the delivered text does not itself change -- and
    `empty_section_after_regeneration` stays ``False`` (it is not the
    literal-zero case, and rule 14 must not treat this thinness as a violation)."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    verify_agent = _RoutingVerifyAgent()
    call_count = {"n": 0}

    async def fake_call_deepseek(system_prompt, user_prompt):
        call_count["n"] += 1
        return _completion(REGEN_TWO_SENTENCE_TEXT)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        side_effect=[
            CitationLinkResult(links=REGEN_TWO_SENTENCE_LINKS, provenance=None),
            CitationLinkResult(links=REGEN_TWO_SENTENCE_LINKS, provenance=None),
        ],
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
    result = job.result

    assert result["loop_stats"]["empty_section_regenerated"] == 1
    assert result["loop_stats"]["regeneration_kept"] == "first"
    assert result["loop_stats"]["empty_section_after_regeneration"] is False
    assert len({link["sentence"] for link in result["citation_links"]}) == 2
    assert call_count["n"] == 2  # the regenerated attempt's own write call, never a third


TIE_FIRST_TEXT = "Tutoring boosts outcomes (Smith, 2020)."
TIE_SECOND_TEXT = "Tutoring boosts morale (Smith, 2020)."

TIE_FIRST_LINKS = [{
    "paragraph_index": 0,
    "sentence": TIE_FIRST_TEXT,
    "keys": ["smith_2020"],
    "citation_text": "(Smith, 2020)",
    "proposition": "Tutoring boosts outcomes",
}]

TIE_SECOND_LINKS = [{
    "paragraph_index": 0,
    "sentence": TIE_SECOND_TEXT,
    "keys": ["smith_2020"],
    "citation_text": "(Smith, 2020)",
    "proposition": "Tutoring boosts morale",
}]


class _AlwaysVerifiedAgent:
    """Verifies every claim unconditionally -- used only for the tie test below,
    where both attempts' own single claim must clear regardless of its exact
    wording. ``evidence_quote`` is ``SMITH_CHUNK_TEXT`` itself (the seeded paper's
    own fulltext chunk), not an invented quote, so the verification guard's own
    verbatim check never demotes the status and triggers an unwanted gate
    revision."""

    def __init__(self):
        self.calls: list[str] = []

    async def run(self, prompt, deps=None):
        self.calls.append(prompt)
        return type(
            "R", (),
            {"output": ClaimVerification(
                claim_text="placeholder", paper_id=uuid.uuid4(), status="verified",
                evidence_quote=SMITH_CHUNK_TEXT, explanation="Checked against full text.",
            )},
        )()


@pytest.mark.asyncio
async def test_a_tie_in_cited_sentence_count_keeps_the_first_attempt(db_session):
    """A genuine tie -- both attempts finalize to the SAME number of distinct
    cited sentences (one, under `THIN_SECTION_CITED_SENTENCE_MINIMUM`), but with
    DIFFERENT text on each side, so the outcome cannot be mistaken for the two
    attempts coincidentally producing identical output. The FIRST attempt is the
    one kept."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_draft, seed_job

    user, project = await _seed_world(db_session)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    call_count = {"n": 0}

    async def fake_call_deepseek(system_prompt, user_prompt):
        call_count["n"] += 1
        return _completion(TIE_FIRST_TEXT if call_count["n"] == 1 else TIE_SECOND_TEXT)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        side_effect=[
            CitationLinkResult(links=TIE_FIRST_LINKS, provenance=None),
            CitationLinkResult(links=TIE_SECOND_LINKS, provenance=None),
        ],
    ), patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=_AlwaysVerifiedAgent(),
    ):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    result = job.result

    assert result["loop_stats"]["empty_section_regenerated"] == 1
    assert result["loop_stats"]["regeneration_kept"] == "first"
    assert result["loop_stats"]["empty_section_after_regeneration"] is False
    assert result["content"] == TIE_FIRST_TEXT
    assert result["citation_links"] == [{**TIE_FIRST_LINKS[0], "paragraph_index": 0}]
    assert call_count["n"] == 2  # the regenerated attempt's own write call, never a third

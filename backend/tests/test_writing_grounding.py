"""The write job grounds every selected paper that has full text but no full-text
analysis yet before building the section context.

``analyze_paper_text_with_provenance`` is reused (not duplicated) from
``app.agents.deep_analysis_agent``, exactly as ``analyze_all_papers`` uses
``analyze_paper_text``. The grounding call's own provenance is folded into the
section's provenance exactly as ``_with_citation_link_provenance`` folds the
citation-link call.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from app.agents.citation_link_agent import CitationLinkResult  # noqa: E402
from app.agents.deep_analysis_agent import DeepAnalysisResult, DeepPaperAnalysis  # noqa: E402
from app.schemas.provenance import LLMCallProvenance  # noqa: E402

NO_LINKS = CitationLinkResult(links=[], provenance=None)

GENERATED = "Grounded prose (Full, 2021)."

PAYLOAD = {
    "id": "chatcmpl-1",
    "model": "deepseek-v4-flash",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": GENERATED}}],
    "usage": {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
}

FULL_TEXT_ANALYSIS = DeepPaperAnalysis(
    key_findings="Deep finding.",
    methodology="A survey.",
    theoretical_framework="",
    key_quotes_with_citations=[],
    claims_and_evidence="",
    builds_on="",
    limitations="",
    future_directions="",
    themes=["x"],
)

ANALYSIS_PROVENANCE = LLMCallProvenance(
    agent="deep_analysis",
    model_configured="deepseek-chat",
    model_reported="deepseek-v4-flash",
    prompt_version="sha256:deadbeef0000",
    input_tokens=4000,
    output_tokens=800,
)


@pytest.mark.asyncio
async def test_grounds_a_selected_paper_with_chunks_and_no_full_text_analysis(db_session):
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import (
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
        db_session, project, title="Full text paper",
        authors=[{"name": "Fiona Full"}], year=2021,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"text": "The whole paper text."}],
        },
    )
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.agents.paper_selector_agent.select_papers_for_section",
        new_callable=AsyncMock, return_value=[str(paper.id)],
    ), patch(
        "app.agents.deep_analysis_agent.analyze_paper_text_with_provenance",
        new_callable=AsyncMock,
        return_value=DeepAnalysisResult(output=FULL_TEXT_ANALYSIS, provenance=ANALYSIS_PROVENANCE),
    ) as mocked_analysis, patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock, return_value=NO_LINKS,
    ), patch("httpx.AsyncClient.post", new=fake_post):
        await generate_section(
            draft=draft, section_type="literature_review", context=None,
            job_id=job.id, session_factory=factory,
        )
    await engine.dispose()

    mocked_analysis.assert_awaited_once()
    call_args = mocked_analysis.await_args
    assert call_args.args[0] == "Full text paper"
    assert "Fiona Full" in call_args.args[1]
    assert call_args.args[2] == 2021
    # The raw chunks are passed through (not joined into one string), so
    # the analysis agent can number them and name a chunk_index for each evidence item.
    assert call_args.args[3] == [{"text": "The whole paper text."}]

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    # The generated text carries a real citation ("(Full, 2021)"), but
    # ``build_citation_link_map`` is mocked to ``NO_LINKS`` on every call, so the
    # empty-section regeneration backstop fires once (a section with a real citation
    # audit but zero cited sentences), regenerates with the same mocked writer and
    # linker, and ships the second attempt's own equally empty outcome -- there is no
    # third attempt.
    assert job.result["loop_stats"]["empty_section_regenerated"] == 1
    assert job.result["loop_stats"]["empty_section_after_regeneration"] is True
    provenance = job.result["provenance"]
    assert provenance["analysis_calls"] == [ANALYSIS_PROVENANCE.model_dump(mode="json")]
    assert provenance["total_calls"] == 3  # generation + one grounding analysis + regeneration
    assert provenance["total_input_tokens"] == 500 + 4000 + 500
    assert provenance["total_output_tokens"] == 100 + 800 + 100

    # persisted back onto the paper so later sections/paper chat see it too
    await db_session.refresh(paper)
    assert paper.metadata_["deep_analysis"]["key_findings"] == "Deep finding."


@pytest.mark.asyncio
async def test_grounding_stores_accepted_evidence_and_records_analysis_metrics(db_session):
    """The grounding step validates the analysis agent's proposed
    evidence, stores only the accepted rows, and records ``{papers, evidence_items,
    rejected_items, verbatim_rate}`` on the write job's own result."""
    from sqlalchemy import select

    from app.models.analysis_job import JobType
    from app.models.evidence import Evidence
    from app.services.writing import generate_section
    from tests.t5_fixtures import (
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
        db_session, project, title="Full text paper",
        authors=[{"name": "Fiona Full"}], year=2021,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [
                {"text": "Tutoring improved outcomes for most students in the sample.",
                 "section": "Results"}
            ],
        },
    )
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    analysis_with_evidence = DeepPaperAnalysis(
        key_findings="Deep finding.", methodology="A survey.", theoretical_framework="",
        key_quotes_with_citations=[], claims_and_evidence="", builds_on="",
        limitations="", future_directions="", themes=["x"],
        evidence=[
            {
                "quote": "Tutoring improved outcomes for most students in the sample.",
                "chunk_index": 0, "finding": "Tutoring helps.", "kind": "finding",
                "origin": "own", "concepts": ["Tutoring ", "tutoring"],
            },
            {
                "quote": "Too short.", "chunk_index": 0, "finding": "x",
                "kind": "finding", "origin": "own", "concepts": [],
            },
        ],
    )

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.agents.paper_selector_agent.select_papers_for_section",
        new_callable=AsyncMock, return_value=[str(paper.id)],
    ), patch(
        "app.agents.deep_analysis_agent.analyze_paper_text_with_provenance",
        new_callable=AsyncMock,
        return_value=DeepAnalysisResult(
            output=analysis_with_evidence, provenance=ANALYSIS_PROVENANCE
        ),
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock, return_value=NO_LINKS,
    ), patch("httpx.AsyncClient.post", new=fake_post):
        await generate_section(
            draft=draft, section_type="literature_review", context=None,
            job_id=job.id, session_factory=factory,
        )

    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["metrics"]["analysis"] == {
        "papers": 1, "papers_attempted": 1, "papers_failed": 0,
        "evidence_items": 1, "rejected_items": 1, "verbatim_rate": 0.5,
    }

    stored = (
        await db_session.execute(select(Evidence).where(Evidence.paper_id == paper.id))
    ).scalars().all()
    assert len(stored) == 1
    assert stored[0].quote == "Tutoring improved outcomes for most students in the sample."
    assert stored[0].section == "Results"
    assert stored[0].origin == "own"
    assert stored[0].concepts == ["tutoring"]
    assert stored[0].prompt_version == "sha256:deadbeef0000"


@pytest.mark.asyncio
async def test_a_papers_failed_grounding_call_is_counted_not_just_dropped(db_session):
    """A single paper's analysis call failing (a
    provider outage, a token-limit error) is caught and skipped, but ``papers`` alone
    (successes only) makes a silently reduced corpus indistinguishable from one where
    every paper simply had nothing to analyse. ``papers_attempted`` records the size of
    the corpus before any call ran; ``papers_failed`` is the gap."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import (
        make_session_factory,
        seed_draft,
        seed_job,
        seed_paper,
        seed_project,
        seed_user,
    )

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper_ok = await seed_paper(
        db_session, project, title="Grounds fine", authors=[{"name": "Ann Full"}], year=2021,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"text": "Some chunk text.", "section": "Results"}],
        },
    )
    paper_fails = await seed_paper(
        db_session, project, title="Analysis explodes", authors=[{"name": "Ben Broken"}],
        year=2022,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"text": "Another chunk.", "section": "Results"}],
        },
    )
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    analysis_no_evidence = DeepAnalysisResult(
        output=FULL_TEXT_ANALYSIS, provenance=ANALYSIS_PROVENANCE
    )

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.agents.paper_selector_agent.select_papers_for_section",
        new_callable=AsyncMock,
        return_value=[str(paper_ok.id), str(paper_fails.id)],
    ), patch(
        "app.agents.deep_analysis_agent.analyze_paper_text_with_provenance",
        new_callable=AsyncMock,
        side_effect=[RuntimeError("provider outage"), analysis_no_evidence],
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock, return_value=NO_LINKS,
    ), patch("httpx.AsyncClient.post", new=fake_post):
        await generate_section(
            draft=draft, section_type="literature_review", context=None,
            job_id=job.id, session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["metrics"]["analysis"] == {
        "papers": 1, "papers_attempted": 2, "papers_failed": 1,
        "evidence_items": 0, "rejected_items": 0, "verbatim_rate": 1.0,
    }


@pytest.mark.asyncio
async def test_a_papers_failed_persistence_is_counted_not_reported_as_grounded(db_session):
    """Incrementing ``papers_analyzed`` right
    after the analysis call, before the persistence write below it, would let a paper
    whose ``deep_analysis``/evidence write fails (a stale row, a serialisation error)
    still count in ``papers`` and its (never-stored) evidence rows still count in
    ``evidence_items``, even though nothing was actually saved. The counter must move
    only when the write actually commits, so this paper lands in ``papers_failed``
    instead, exactly like an analysis-call failure."""
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import (
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
        db_session, project, title="Persistence fails", authors=[{"name": "Pat Fail"}],
        year=2023,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"text": "Some chunk text.", "section": "Results"}],
        },
    )
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    analysis_no_evidence = DeepAnalysisResult(
        output=FULL_TEXT_ANALYSIS, provenance=ANALYSIS_PROVENANCE
    )

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.agents.paper_selector_agent.select_papers_for_section",
        new_callable=AsyncMock, return_value=[str(paper.id)],
    ), patch(
        "app.agents.deep_analysis_agent.analyze_paper_text_with_provenance",
        new_callable=AsyncMock, return_value=analysis_no_evidence,
    ), patch(
        "app.services.deep_analysis.replace_paper_evidence",
        new_callable=AsyncMock, side_effect=RuntimeError("stale row"),
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock, return_value=NO_LINKS,
    ), patch("httpx.AsyncClient.post", new=fake_post):
        await generate_section(
            draft=draft, section_type="literature_review", context=None,
            job_id=job.id, session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["metrics"]["analysis"] == {
        "papers": 0, "papers_attempted": 1, "papers_failed": 1,
        "evidence_items": 0, "rejected_items": 0, "verbatim_rate": 1.0,
    }

    await db_session.refresh(paper)
    assert paper.metadata_.get("deep_analysis") is None


@pytest.mark.asyncio
async def test_a_paper_already_grounded_is_not_reanalysed(db_session):
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import (
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
        db_session, project, title="Already grounded",
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"text": "Whole text."}],
            "deep_analysis": {"key_findings": "already", "methodology": "a survey"},
        },
    )
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.agents.paper_selector_agent.select_papers_for_section",
        new_callable=AsyncMock, return_value=[str(paper.id)],
    ), patch(
        "app.agents.deep_analysis_agent.analyze_paper_text_with_provenance",
        new_callable=AsyncMock,
    ) as mocked_analysis, patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock, return_value=NO_LINKS,
    ), patch("httpx.AsyncClient.post", new=fake_post):
        await generate_section(
            draft=draft, section_type="literature_review", context=None,
            job_id=job.id, session_factory=factory,
        )
    await engine.dispose()

    mocked_analysis.assert_not_awaited()
    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["provenance"]["analysis_calls"] == []


@pytest.mark.asyncio
async def test_a_paper_without_full_text_is_never_analysed(db_session):
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import (
        make_session_factory,
        seed_draft,
        seed_job,
        seed_paper,
        seed_project,
        seed_user,
    )

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(db_session, project, title="Abstract only", metadata_={})
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.agents.paper_selector_agent.select_papers_for_section",
        new_callable=AsyncMock, return_value=[str(paper.id)],
    ), patch(
        "app.agents.deep_analysis_agent.analyze_paper_text_with_provenance",
        new_callable=AsyncMock,
    ) as mocked_analysis, patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock, return_value=NO_LINKS,
    ), patch("httpx.AsyncClient.post", new=fake_post):
        await generate_section(
            draft=draft, section_type="literature_review", context=None,
            job_id=job.id, session_factory=factory,
        )
    await engine.dispose()

    mocked_analysis.assert_not_awaited()
    await db_session.refresh(job)
    assert job.status.value == "completed", job.error


#: The writer's context is built from each paper's own stored evidence, not its
#: abstract or a prose summary of its analysis.


@pytest.mark.asyncio
async def test_a_paper_without_full_text_is_listed_as_unavailable(db_session):
    from app.services.writing import _build_evidence_context
    from tests.t5_fixtures import make_session_factory, seed_paper, seed_project, seed_user

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session, project, title="No full text",
        authors=[{"name": "Ann Author"}], year=2019, metadata_={},
    )
    factory, engine = make_session_factory()
    result = await _build_evidence_context([paper], factory)
    await engine.dispose()

    assert "## Not available for citation (no full text)" in result.text
    assert "Ann Author (2019) - No full text" in result.text
    assert result.catalog == []


@pytest.mark.asyncio
async def test_a_full_text_paper_with_no_evidence_rows_is_shown_with_none(db_session):
    from app.services.writing import _build_evidence_context
    from tests.t5_fixtures import make_session_factory, seed_paper, seed_project, seed_user

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session, project, title="Grounded, no evidence yet",
        authors=[{"name": "Ann Author"}], year=2019,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": [{"text": "Text."}]},
    )
    factory, engine = make_session_factory()
    result = await _build_evidence_context([paper], factory)
    await engine.dispose()

    assert "## Evidence available for citation" in result.text
    assert "(no evidence extracted from this paper's full text)" in result.text
    assert result.catalog == []


@pytest.mark.asyncio
async def test_own_origin_evidence_is_grouped_by_concept_with_short_ids(db_session):
    from app.models.evidence import Evidence
    from app.services.writing import _build_evidence_context
    from tests.t5_fixtures import make_session_factory, seed_paper, seed_project, seed_user

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session, project, title="Grounded",
        authors=[{"name": "Ann Author"}], year=2019,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": [{"text": "Text."}]},
    )
    db_session.add_all([
        Evidence(
            paper_id=paper.id, quote="Tutoring improved outcomes for most students.",
            chunk_index=0, finding="Tutoring helps.", kind="finding", origin="own",
            concepts=["tutoring"], prompt_version="sha256:x",
        ),
        Evidence(
            paper_id=paper.id, quote="The sample included 40 undergraduates.",
            chunk_index=0, finding="Sample size.", kind="sample", origin="own",
            concepts=["sample"], prompt_version="sha256:x",
        ),
        Evidence(
            paper_id=paper.id, quote="Prior work reported similar gains elsewhere.",
            chunk_index=0, finding="Cites prior work.", kind="finding", origin="reported",
            concepts=["tutoring"], prompt_version="sha256:x",
        ),
    ])
    await db_session.commit()

    factory, engine = make_session_factory()
    result = await _build_evidence_context([paper], factory)
    await engine.dispose()

    assert "Concept: tutoring" in result.text
    assert "Concept: sample" in result.text
    assert "Tutoring improved outcomes for most students." in result.text
    assert "The sample included 40 undergraduates." in result.text
    # The reported-origin item is never shown to the writer (design section 6, A1).
    assert "Prior work reported similar gains elsewhere." not in result.text

    assert len(result.catalog) == 2
    quotes = {item["quote"] for item in result.catalog}
    assert quotes == {
        "Tutoring improved outcomes for most students.",
        "The sample included 40 undergraduates.",
    }
    assert {item["key"] for item in result.catalog} == {"author_2019"}
    ids = {item["id"] for item in result.catalog}
    assert len(ids) == 2  # each evidence row keeps one id, even if shown under >1 concept


@pytest.mark.asyncio
async def test_an_evidence_item_with_two_concepts_keeps_one_id_under_both_headings(db_session):
    from app.models.evidence import Evidence
    from app.services.writing import _build_evidence_context
    from tests.t5_fixtures import make_session_factory, seed_paper, seed_project, seed_user

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session, project, title="Grounded",
        authors=[{"name": "Ann Author"}], year=2019,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": [{"text": "Text."}]},
    )
    db_session.add(Evidence(
        paper_id=paper.id, quote="Tutoring improved reading and writing scores.",
        chunk_index=0, finding="Tutoring helps.", kind="finding", origin="own",
        concepts=["reading", "writing"], prompt_version="sha256:x",
    ))
    await db_session.commit()

    factory, engine = make_session_factory()
    result = await _build_evidence_context([paper], factory)
    await engine.dispose()

    assert len(result.catalog) == 1
    short_id = result.catalog[0]["id"]
    assert result.text.count(f"[{short_id}]") == 2


@pytest.mark.asyncio
async def test_a_paper_with_no_derivable_key_contributes_no_catalog_entries(db_session):
    from app.models.evidence import Evidence
    from app.services.writing import _build_evidence_context
    from tests.t5_fixtures import make_session_factory, seed_paper, seed_project, seed_user

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session, project, title="No authors", authors=[], year=None,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": [{"text": "Text."}]},
    )
    db_session.add(Evidence(
        paper_id=paper.id, quote="Tutoring improved outcomes for most students.",
        chunk_index=0, finding="Tutoring helps.", kind="finding", origin="own",
        concepts=["tutoring"], prompt_version="sha256:x",
    ))
    await db_session.commit()

    factory, engine = make_session_factory()
    result = await _build_evidence_context([paper], factory)
    await engine.dispose()

    assert "Tutoring improved outcomes for most students." in result.text
    assert result.catalog == []


@pytest.mark.asyncio
async def test_no_papers_at_all_returns_a_plain_message(db_session):
    from app.services.writing import _build_evidence_context
    from tests.t5_fixtures import make_session_factory

    factory, engine = make_session_factory()
    result = await _build_evidence_context([], factory)
    await engine.dispose()

    assert result.text == "No papers available."
    assert result.catalog == []

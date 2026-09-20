"""Drafts are always written with author-year citations; a journal's own citation style
is applied only at export.

``merge_writing_rules`` must never forward a journal's ``citation_style`` /
``reference_format`` into the merge prompt, and ``generate_section`` must never forward
the merge agent's own ``citation_rules`` output into the assembled system prompt --
either one, left in, could tell the writing model to use something other than
author-year, contradicting ``_BASE_RULES``.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402


class _FakeResult:
    def __init__(self, output):
        self.output = output


class _FakeRulesAgent:
    """Captures the prompt it was called with; returns a canned MergedWritingRules."""

    def __init__(self, output):
        self.output = output
        self.calls: list[str] = []

    async def run(self, prompt: str):
        self.calls.append(prompt)
        return _FakeResult(self.output)


@pytest.mark.asyncio
async def test_merge_writing_rules_never_sends_the_journals_citation_style_or_format():
    from app.agents.writing_rules_agent import MergedWritingRules, merge_writing_rules

    fake_agent = _FakeRulesAgent(
        MergedWritingRules(
            section_structure="s",
            word_limit="w",
            citation_rules="author-year only",
            tone_and_style="t",
            content_requirements="c",
            formatting="f",
            special_instructions="i",
        )
    )
    guidelines = {
        "journal_name": "Journal of Examples",
        "citation_style": "IEEE",
        "reference_format": "numbered, in order of appearance",
        "word_limit_total": 8000,
    }
    with patch(
        "app.agents.writing_rules_agent.get_writing_rules_agent", return_value=fake_agent
    ):
        await merge_writing_rules(
            section_type="introduction", paper_type="research_article",
            journal_guidelines=guidelines,
        )

    assert len(fake_agent.calls) == 1
    prompt = fake_agent.calls[0]
    # The journal's own citation_style/reference_format values never reach the prompt at
    # all -- only the golden standard's fixed "APA 7th edition" line does, unaffected by
    # what the journal specifies.
    assert "IEEE" not in prompt
    assert "numbered, in order of appearance" not in prompt
    assert "Reference format" not in prompt
    journal_section = prompt.split("JOURNAL GUIDELINES", 1)[1]
    assert "Citation style" not in journal_section
    # Everything else about the journal guidelines still reaches the prompt.
    assert "Journal of Examples" in prompt
    assert "8000" in prompt


def test_merge_prompt_fixes_in_text_citations_to_author_year():
    from app.agents.writing_rules_agent import MERGE_PROMPT

    assert "author-year" in MERGE_PROMPT.lower()


@pytest.mark.asyncio
async def test_generate_section_never_forwards_the_merge_agents_citation_rules(db_session):
    """Even if the merge agent answers with a non-author-year citation_rules string (a
    model ignoring MERGE_PROMPT's own instruction), generate_section's assembled system
    prompt never carries it -- the append site is removed entirely."""
    from app.agents.writing_rules_agent import MergedWritingRules
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

    merged = MergedWritingRules(
        section_structure="s",
        word_limit="w",
        citation_rules="Use IEEE numbered citations in square brackets.",
        tone_and_style="t",
        content_requirements="c",
        formatting="f",
        special_instructions="i",
    )

    captured: dict = {}

    async def fake_post(self, url, **kwargs):
        captured["json"] = kwargs["json"]
        payload = {
            "id": "chatcmpl-1",
            "model": "deepseek-v4-flash",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Body."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock,
        return_value=merged,
    ), patch("httpx.AsyncClient.post", new=fake_post):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    system_prompt = captured["json"]["messages"][0]["content"]
    assert "IEEE" not in system_prompt
    assert "square brackets" not in system_prompt
    # Everything else the merge agent produced still reaches the prompt.
    assert merged.section_structure in system_prompt
    assert merged.tone_and_style in system_prompt

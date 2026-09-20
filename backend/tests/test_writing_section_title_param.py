"""``generate_section``'s own ``title`` parameter, used
verbatim as the section's saved heading in place of the ``section_type``-derived
default, so a caller (the demo runner's own extra sections) can send the requested
title from ``sections.json`` through to the write job instead of always getting back
"Literature Review" for every ``section_type="literature_review"`` request.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from app.services.writing import WritingCompletion  # noqa: E402


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


async def _run_generate_section(db_session, *, title):
    from app.models.analysis_job import JobType
    from app.models.draft import Draft
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

    async def fake_call_deepseek(system_prompt, user_prompt):
        return _completion("Framing prose with no citation at all.")

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock, side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek", new=fake_call_deepseek,
    ), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        side_effect=RuntimeError("linker not exercised by this test"),
    ):
        await generate_section(
            draft=draft, section_type="literature_review", context=None, job_id=job.id,
            session_factory=factory, title=title,
        )
    await engine.dispose()
    await db_session.refresh(job)

    saved = await db_session.get(Draft, draft.id)
    await db_session.refresh(saved)
    return job, saved


@pytest.mark.asyncio
async def test_generate_section_uses_the_requested_title_over_the_section_type_default(
    db_session,
):
    job, saved = await _run_generate_section(
        db_session, title="Peer written corrective feedback"
    )

    assert job.status.value == "completed", job.error
    heading = saved.content["content"][0]
    assert heading["type"] == "heading"
    assert heading["content"][0]["text"] == "Peer written corrective feedback"


@pytest.mark.asyncio
async def test_generate_section_falls_back_to_the_section_type_default_title_when_none(
    db_session,
):
    """The default, unchanged: every call site before *title* existed passed nothing,
    and must keep getting the section_type-derived heading."""
    job, saved = await _run_generate_section(db_session, title=None)

    assert job.status.value == "completed", job.error
    heading = saved.content["content"][0]
    assert heading["content"][0]["text"] == "Literature Review"

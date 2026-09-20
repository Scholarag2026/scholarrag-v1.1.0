"""writing.py's rules-merge fallback must degrade gracefully, not raise NameError."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

import logging  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402


def test_writing_module_defines_logger():
    from app.services import writing

    assert isinstance(getattr(writing, "logger", None), logging.Logger)
    assert writing.logger.name == "app.services.writing"


@pytest.mark.asyncio
async def test_generate_section_falls_back_when_rules_merge_fails(db_session):
    """When merge_writing_rules raises, the job completes with generated content."""
    from app.models.analysis_job import JobType
    from app.models.draft import Draft
    from app.services.writing import generate_section
    from tests.t5_fixtures import make_session_factory, seed_job, seed_project, seed_user

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    draft = Draft(project_id=project.id, user_id=user.id, title="D")
    db_session.add(draft)
    await db_session.commit()
    job = await seed_job(db_session, project, JobType.writing)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM down"),
    ), patch(
        "app.services.writing.call_deepseek",
        new_callable=AsyncMock,
        # call_deepseek returns WritingCompletion(content, provenance) since v1.1.0.
        return_value=("Generated body text.", {}),
    ):
        await generate_section(
            draft=draft,
            section_type="methods",
            context=None,
            job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["content"] == "Generated body text."

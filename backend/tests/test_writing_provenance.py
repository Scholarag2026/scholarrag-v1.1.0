"""AI Write records which model answered and audits its citations.

The DeepSeek HTTP call is intercepted at ``httpx.AsyncClient.post``; the fake response
carries the ``model`` field the provider returns for the configured alias.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

GENERATED = (
    "Tutoring improves outcomes (Smith, 2020). Brown et al. (2018) concur. "
    "Some claim otherwise (Nguyen, 2021). Costs are rising [NEEDS CITATION]."
)
# The citation-link call is never mocked in this file, so it always fails (no network
# credentials in the test environment): no citation links, no verification, and
# finalize still strips the literal marker unconditionally.
FINALIZED = (
    "Tutoring improves outcomes (Smith, 2020). Brown et al. (2018) concur. "
    "Some claim otherwise (Nguyen, 2021). Costs are rising."
)

PAYLOAD = {
    "id": "chatcmpl-42",
    "model": "deepseek-v4-flash",
    "system_fingerprint": "fp_test",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": GENERATED}}],
    "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500},
}


@pytest.mark.asyncio
async def test_generate_section_records_provenance_and_citation_audit(db_session):
    from app.config import settings
    from app.models.analysis_job import JobType
    from app.schemas.provenance import prompt_version
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
    await seed_paper(db_session, project, title="A", authors=[{"name": "Jane Smith"}], year=2020)
    await seed_paper(db_session, project, title="B", authors=["Amy Brown"], year=2018)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    captured: dict = {}

    async def fake_post(self, url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM down"),
    ), patch("httpx.AsyncClient.post", new=fake_post):
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
    # The generated text is stored finalized: the citation-link call is unmocked here
    # and always fails in this test environment, so there is nothing to verify or
    # remove except the unconditional [NEEDS CITATION] strip.
    assert job.result["content"] == FINALIZED
    assert job.result["section_type"] == "methods"

    request = captured["json"]
    assert request["model"] == settings.deepseek_model
    provenance = job.result["provenance"]
    assert provenance["agent"] == "writing"
    assert provenance["model_configured"] == settings.deepseek_model
    assert provenance["model_reported"] == "deepseek-v4-flash"
    assert provenance["temperature"] == 0.7 == request["temperature"]
    assert provenance["max_tokens"] == 4096 == request["max_tokens"]
    assert provenance["thinking"] == "disabled"
    assert request["thinking"] == {"type": "disabled"}
    assert "reasoning_tokens" not in provenance
    system_prompt = request["messages"][0]["content"]
    assert provenance["prompt_version"] == prompt_version(system_prompt)
    assert provenance["provider_response_id"] == "chatcmpl-42"
    assert provenance["system_fingerprint"] == "fp_test"
    assert provenance["input_tokens"] == 1200
    assert provenance["output_tokens"] == 300

    assert job.result["citation_audit"] == {
        "matched": ["Smith, 2020", "Brown, 2018"],
        "unmatched": ["Nguyen, 2021"],
        "needs_citation_flags": 1,
        "total": 3,
    }


@pytest.mark.asyncio
async def test_call_deepseek_returns_content_and_provenance():
    from app.config import settings
    from app.schemas.provenance import prompt_version
    from app.services.writing import call_deepseek

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        completion = await call_deepseek("SYSTEM", "USER")

    content, provenance = completion
    assert content == GENERATED
    assert completion.content == GENERATED
    assert provenance["model_configured"] == settings.deepseek_model
    assert provenance["model_reported"] == "deepseek-v4-flash"
    assert provenance["prompt_version"] == prompt_version("SYSTEM")
    assert provenance["temperature"] == 0.7
    assert provenance["max_tokens"] == 4096
    assert provenance["thinking"] == "disabled"
    assert "reasoning_tokens" not in provenance


@pytest.mark.asyncio
async def test_call_deepseek_stores_only_content_when_reasoning_content_present():
    """A thinking-mode response carries both ``message.content`` (the visible answer)
    and ``message.reasoning_content`` (the chain-of-thought text). Only ``content`` is
    ever read or stored; ``reasoning_content`` must not leak into the returned text or
    into provenance."""
    from app.services.writing import call_deepseek

    payload = {
        "id": "chatcmpl-43",
        "model": "deepseek-v4-flash",
        "system_fingerprint": "fp_test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": GENERATED,
                    "reasoning_content": "internal chain-of-thought that must not be stored",
                },
            }
        ],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500},
    }

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        content, provenance = await call_deepseek("SYSTEM", "USER")

    assert content == GENERATED
    assert "reasoning_content" not in repr(provenance)
    assert "internal chain-of-thought" not in content


@pytest.mark.asyncio
async def test_call_deepseek_tolerates_missing_metadata():
    from app.services.writing import call_deepseek

    minimal = {"choices": [{"message": {"content": "Text."}}]}

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=minimal, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        content, provenance = await call_deepseek("SYSTEM", "USER")

    assert content == "Text."
    assert provenance["model_reported"] is None
    assert provenance["system_fingerprint"] is None
    assert provenance["input_tokens"] is None


@pytest.mark.asyncio
async def test_call_deepseek_defaults_to_deepseek_model_when_writing_model_unset(monkeypatch):
    """``writing_model`` defaults to ``None``; with it unset, the writer keeps sending
    ``deepseek_model``, exactly as before this override existed."""
    from app.config import settings
    from app.services.writing import call_deepseek

    monkeypatch.setattr(settings, "writing_model", None)
    captured: dict = {}

    async def fake_post(self, url, **kwargs):
        captured["json"] = kwargs["json"]
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        _, provenance = await call_deepseek("SYSTEM", "USER")

    assert captured["json"]["model"] == settings.deepseek_model
    assert provenance["model_configured"] == settings.deepseek_model


@pytest.mark.asyncio
async def test_call_deepseek_uses_writing_model_override(monkeypatch):
    """A configured ``writing_model`` is sent as the request model and recorded as
    ``provenance.model_configured``; only the writer reads it."""
    from app.config import settings
    from app.services.writing import call_deepseek

    monkeypatch.setattr(settings, "writing_model", "deepseek-v4-pro")
    captured: dict = {}

    async def fake_post(self, url, **kwargs):
        captured["json"] = kwargs["json"]
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        _, provenance = await call_deepseek("SYSTEM", "USER")

    assert captured["json"]["model"] == "deepseek-v4-pro"
    assert provenance["model_configured"] == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_call_deepseek_ignores_empty_writing_model(monkeypatch):
    """An empty-string ``writing_model`` (e.g. ``WRITING_MODEL=`` left blank in `.env`)
    counts as unset: the writer falls back to ``deepseek_model``."""
    from app.config import settings
    from app.services.writing import call_deepseek

    monkeypatch.setattr(settings, "writing_model", "")
    captured: dict = {}

    async def fake_post(self, url, **kwargs):
        captured["json"] = kwargs["json"]
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        _, provenance = await call_deepseek("SYSTEM", "USER")

    assert captured["json"]["model"] == settings.deepseek_model
    assert provenance["model_configured"] == settings.deepseek_model


@pytest.mark.asyncio
async def test_call_deepseek_uses_writing_max_tokens_override(monkeypatch):
    """A configured ``writing_max_tokens`` is sent as ``max_tokens`` and recorded in
    provenance; only the writer reads it."""
    from app.config import settings
    from app.services.writing import call_deepseek

    monkeypatch.setattr(settings, "writing_max_tokens", 16384)
    captured: dict = {}

    async def fake_post(self, url, **kwargs):
        captured["json"] = kwargs["json"]
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        _, provenance = await call_deepseek("SYSTEM", "USER")

    assert captured["json"]["max_tokens"] == 16384
    assert provenance["max_tokens"] == 16384


@pytest.mark.asyncio
async def test_call_deepseek_sends_thinking_disabled_by_default_when_unset(monkeypatch):
    """``writing_thinking`` defaults to ``None``, but the request must still carry an
    explicit ``thinking: {"type": "disabled"}`` field, the same shape
    ``app.agents.model_config``'s shared ``ModelSettings`` tiers use for every other
    agent. ``deepseek-flash`` (the model the app default moves to once the retired
    ``deepseek-chat`` alias is gone) defaults to thinking mode ON when no ``thinking``
    field is sent at all, unlike ``deepseek-chat``; every frozen writer run and the
    promoted demo ran with thinking off, so an unset ``WRITING_THINKING`` must keep
    reproducing that, not silently flip to thinking-on under the new model name."""
    from app.config import settings
    from app.services.writing import call_deepseek

    monkeypatch.setattr(settings, "writing_thinking", None)
    captured: dict = {}

    async def fake_post(self, url, **kwargs):
        captured["json"] = kwargs["json"]
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        _, provenance = await call_deepseek("SYSTEM", "USER")

    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert provenance["thinking"] == "disabled"


@pytest.mark.asyncio
async def test_call_deepseek_sends_thinking_enabled_when_set(monkeypatch):
    from app.config import settings
    from app.services.writing import call_deepseek

    monkeypatch.setattr(settings, "writing_thinking", "enabled")
    captured: dict = {}

    async def fake_post(self, url, **kwargs):
        captured["json"] = kwargs["json"]
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        _, provenance = await call_deepseek("SYSTEM", "USER")

    assert captured["json"]["thinking"] == {"type": "enabled"}
    assert provenance["thinking"] == "enabled"


@pytest.mark.asyncio
async def test_call_deepseek_sends_thinking_disabled_when_set(monkeypatch):
    from app.config import settings
    from app.services.writing import call_deepseek

    monkeypatch.setattr(settings, "writing_thinking", "disabled")
    captured: dict = {}

    async def fake_post(self, url, **kwargs):
        captured["json"] = kwargs["json"]
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        _, provenance = await call_deepseek("SYSTEM", "USER")

    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert provenance["thinking"] == "disabled"


@pytest.mark.asyncio
async def test_call_deepseek_records_reasoning_tokens_when_present():
    """``usage.completion_tokens_details.reasoning_tokens``, when the response
    carries it, is recorded in provenance under the same name."""
    from app.services.writing import call_deepseek

    payload = {
        **PAYLOAD,
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 300,
            "total_tokens": 1500,
            "completion_tokens_details": {"reasoning_tokens": 210},
        },
    }

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        _, provenance = await call_deepseek("SYSTEM", "USER")

    assert provenance["reasoning_tokens"] == 210
    assert provenance["output_tokens"] == 300


@pytest.mark.asyncio
async def test_call_deepseek_omits_reasoning_tokens_when_absent():
    """No ``completion_tokens_details`` in the response usage (a non-thinking call,
    or a provider that omits the key entirely) leaves ``reasoning_tokens`` out of
    provenance rather than recording a fabricated ``None`` or ``0``."""
    from app.services.writing import call_deepseek

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, json=PAYLOAD, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        _, provenance = await call_deepseek("SYSTEM", "USER")

    assert "reasoning_tokens" not in provenance


@pytest.mark.asyncio
async def test_generate_section_records_prompt_template_version(db_session):
    """The section prompt template is identifiable across runs even though the full
    system prompt also contains LLM-merged writing rules."""
    from app.models.analysis_job import JobType
    from app.schemas.provenance import prompt_version
    from app.services.writing import generate_section, get_section_prompt
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
    ), patch("httpx.AsyncClient.post", new=fake_post):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory, expertise_level="researcher",
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    provenance = job.result["provenance"]
    assert provenance["prompt_template_version"] == prompt_version(
        get_section_prompt("methods", "en")
    )
    # The expertise modifier is part of the prompt sent, not of the template.
    assert provenance["prompt_template_version"] != provenance["prompt_version"]


@pytest.mark.asyncio
async def test_citation_audit_failure_does_not_fail_the_writing_job(db_session):
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
        "app.services.writing.audit_citations", side_effect=RuntimeError("audit exploded")
    ):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["content"] == FINALIZED
    assert job.result["citation_audit"] is None
    assert job.result["provenance"]["model_reported"] == "deepseek-v4-flash"

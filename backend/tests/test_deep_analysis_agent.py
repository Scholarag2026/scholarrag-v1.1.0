"""The deep-analysis agent must be cached, not rebuilt per paper.

deep_analysis.py calls analyze_paper_text() once per full-text paper; each Agent owns
a never-closed httpx.AsyncClient, so per-paper construction leaked a connection pool
per paper (issue RATE-LIMITER-NOT-SHARED).
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from app.agents.deep_analysis_agent import (  # noqa: E402
    DEEP_ANALYSIS_PROMPT,
    DeepAnalysisResult,
    DeepPaperAnalysis,
    _agents,
    _resolve_instructions,
    analyze_paper_text,
    analyze_paper_text_with_provenance,
    get_deep_analysis_agent,
)
from app.config import settings  # noqa: E402
from app.schemas.provenance import prompt_version  # noqa: E402


def setup_function():
    _agents.clear()


def test_same_expertise_level_returns_the_same_agent():
    first = get_deep_analysis_agent("student")
    second = get_deep_analysis_agent("student")
    assert first is second


def test_different_expertise_levels_get_different_agents():
    student = get_deep_analysis_agent("student")
    faculty = get_deep_analysis_agent("faculty")
    assert student is not faculty
    assert len(_agents) == 2


def test_none_and_unknown_levels_share_the_base_agent():
    """An unknown level yields an empty prompt suffix, so it is the base agent."""
    base = get_deep_analysis_agent()
    assert get_deep_analysis_agent(None) is base
    assert get_deep_analysis_agent("not-a-real-level") is base
    assert len(_agents) == 1


def test_expertise_instructions_are_applied():
    student = get_deep_analysis_agent("student")
    base = get_deep_analysis_agent()
    assert "student researcher" in str(student._instructions)
    assert "student researcher" not in str(base._instructions)


def test_deep_analysis_agent_has_long_timeout_and_retries():
    agent = get_deep_analysis_agent("student")
    assert agent.model_settings["timeout"] == settings.llm_long_timeout_seconds
    assert agent._max_tool_retries == settings.analysis_max_retries


_ANALYSIS = DeepPaperAnalysis(
    key_findings="x", methodology="y", theoretical_framework="",
    key_quotes_with_citations=[], claims_and_evidence="", builds_on="",
    limitations="", future_directions="", themes=[],
)


def _fake_analysis_run_result(output: DeepPaperAnalysis, *, input_tokens=4000, output_tokens=800):
    response = SimpleNamespace(
        model_name="deepseek-v4-flash",
        provider_response_id="resp-da-1",
        provider_details={"system_fingerprint": "fp_da"},
        provider_name="deepseek",
    )
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(output=output, response=response, usage=lambda: usage)


@pytest.mark.asyncio
async def test_analyze_paper_text_with_provenance_returns_output_and_call_provenance():
    """The write job's grounding step needs this call's own model,
    prompt version and token usage to fold into the section's provenance, which
    ``analyze_paper_text`` alone discards."""
    fake_result = _fake_analysis_run_result(_ANALYSIS)
    fake_agent = type("A", (), {"run": AsyncMock(return_value=fake_result)})()
    with patch(
        "app.agents.deep_analysis_agent.get_deep_analysis_agent", return_value=fake_agent
    ):
        result = await analyze_paper_text_with_provenance(
            "T", "A", 2020, [{"text": "Full text."}]
        )

    assert isinstance(result, DeepAnalysisResult)
    assert result.output == _ANALYSIS
    assert result.provenance.agent == "deep_analysis"
    assert result.provenance.model_configured == settings.deepseek_model
    assert result.provenance.model_reported == "deepseek-v4-flash"
    assert result.provenance.prompt_version == prompt_version(_resolve_instructions(None))
    assert result.provenance.input_tokens == 4000
    assert result.provenance.output_tokens == 800


@pytest.mark.asyncio
async def test_analyze_paper_text_still_returns_only_the_output():
    """Existing call sites (``app.services.deep_analysis.analyze_all_papers``) call
    ``analyze_paper_text`` and expect just the analysis, not a tuple."""
    fake_result = _fake_analysis_run_result(_ANALYSIS)
    fake_agent = type("A", (), {"run": AsyncMock(return_value=fake_result)})()
    with patch(
        "app.agents.deep_analysis_agent.get_deep_analysis_agent", return_value=fake_agent
    ):
        output = await analyze_paper_text("T", "A", 2020, [{"text": "Full text."}])

    assert output == _ANALYSIS


# --------------------------------------------------------------------------------------
# Chunks are shown numbered, and the schema carries evidence items.
# --------------------------------------------------------------------------------------


def test_build_numbered_chunks_labels_each_chunk_with_its_0_based_index():
    from app.agents.deep_analysis_agent import _build_numbered_chunks

    chunks = [{"text": "First."}, {"text": "Second.", "section": "Results"}]
    prompt = _build_numbered_chunks(chunks)
    assert "--- Chunk 0 ---\nFirst." in prompt
    assert "--- Chunk 1 (Results) ---\nSecond." in prompt


def test_build_numbered_chunks_tolerates_no_chunks():
    from app.agents.deep_analysis_agent import _build_numbered_chunks

    assert _build_numbered_chunks([]) == "No full-text chunks available."


def test_build_analysis_prompt_tells_the_model_to_reuse_the_chunk_numbers():
    from app.agents.deep_analysis_agent import _build_analysis_prompt

    prompt = _build_analysis_prompt("T", "A", 2020, [{"text": "Body."}])
    assert "chunk_index" in prompt
    assert "--- Chunk 0 ---" in prompt


def test_the_prompt_instructs_evidence_extraction():
    assert "evidence:" in DEEP_ANALYSIS_PROMPT
    assert "chunk_index" in DEEP_ANALYSIS_PROMPT
    assert "verbatim" in DEEP_ANALYSIS_PROMPT
    assert '"own"' in DEEP_ANALYSIS_PROMPT
    assert '"reported"' in DEEP_ANALYSIS_PROMPT


def test_deep_paper_analysis_accepts_evidence_items():
    analysis = DeepPaperAnalysis(
        key_findings="x", methodology="y", theoretical_framework="",
        key_quotes_with_citations=[], claims_and_evidence="", builds_on="",
        limitations="", future_directions="", themes=[],
        evidence=[{
            "quote": "Tutoring improved outcomes for most students in the sample.",
            "chunk_index": 0, "finding": "Tutoring helps.", "kind": "finding",
            "origin": "own", "concepts": ["tutoring"],
        }],
    )
    assert len(analysis.evidence) == 1
    assert analysis.evidence[0].kind == "finding"
    assert analysis.evidence[0].origin == "own"


def test_deep_paper_analysis_defaults_evidence_to_an_empty_list():
    """A stored analysis from before this field existed (or the abstract-only
    placeholder) still parses, with no evidence."""
    assert _ANALYSIS.evidence == []


# --------------------------------------------------------------------------------------
# The output budget, the per-paper item cap, and the shrink-and-retry ladder.
# On the twelve seed papers the analysis call can truncate mid-tool-call, returning
# nothing at all for the affected papers.
# --------------------------------------------------------------------------------------


def test_the_analysis_agent_pins_an_explicit_output_budget():
    """Without ``max_tokens`` the ceiling is whatever the provider defaults to that day,
    which is what the seed run hit: eight calls ended as ``IncompleteToolCall`` with no
    analysis and no evidence for those papers."""
    from app.agents.model_config import ANALYSIS_MODEL_SETTINGS

    agent = get_deep_analysis_agent()
    assert agent.model_settings["timeout"] == settings.llm_long_timeout_seconds
    assert agent.model_settings["max_tokens"] == ANALYSIS_MODEL_SETTINGS["max_tokens"]
    assert ANALYSIS_MODEL_SETTINGS["max_tokens"] >= 8192


def test_the_prompt_demands_a_character_for_character_copy():
    assert "character for character" in DEEP_ANALYSIS_PROMPT
    assert "evidence:" in DEEP_ANALYSIS_PROMPT
    assert "chunk_index" in DEEP_ANALYSIS_PROMPT


def test_the_user_prompt_names_the_item_cap():
    from app.agents.deep_analysis_agent import EVIDENCE_ITEM_CAPS, _build_analysis_prompt

    default_cap = EVIDENCE_ITEM_CAPS[0]
    prompt = _build_analysis_prompt("T", "A", 2020, [{"text": "Body."}])
    assert "at most %d evidence items" % default_cap in prompt

    smaller = _build_analysis_prompt("T", "A", 2020, [{"text": "Body."}], max_evidence_items=3)
    assert "at most 3 evidence items" in smaller


@pytest.mark.asyncio
async def test_a_truncated_call_is_retried_asking_for_fewer_items():
    from pydantic_ai.exceptions import IncompleteToolCall

    from app.agents.deep_analysis_agent import EVIDENCE_ITEM_CAPS

    prompts: list[str] = []

    async def run(prompt, *args, **kwargs):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise IncompleteToolCall("Model token limit (provider default) exceeded")
        return _fake_analysis_run_result(_ANALYSIS)

    fake_agent = type("A", (), {"run": staticmethod(run)})()
    with patch(
        "app.agents.deep_analysis_agent.get_deep_analysis_agent", return_value=fake_agent
    ):
        result = await analyze_paper_text_with_provenance(
            "T", "A", 2020, [{"text": "Full text."}]
        )

    assert result.output == _ANALYSIS
    assert len(prompts) == 2
    assert "at most %d evidence items" % EVIDENCE_ITEM_CAPS[0] in prompts[0]
    assert "at most %d evidence items" % EVIDENCE_ITEM_CAPS[1] in prompts[1]


@pytest.mark.asyncio
async def test_a_call_that_truncates_at_every_cap_still_raises():
    """One paper's analysis failing is caught per paper by both callers
    (``analyze_all_papers`` and the write job's grounding step), so the ladder must not
    swallow a persistent failure into an empty analysis."""
    from pydantic_ai.exceptions import IncompleteToolCall

    from app.agents.deep_analysis_agent import EVIDENCE_ITEM_CAPS

    calls = {"n": 0}

    async def run(prompt, *args, **kwargs):
        calls["n"] += 1
        raise IncompleteToolCall("still too long")

    fake_agent = type("A", (), {"run": staticmethod(run)})()
    with (
        patch(
            "app.agents.deep_analysis_agent.get_deep_analysis_agent", return_value=fake_agent
        ),
        pytest.raises(IncompleteToolCall),
    ):
        await analyze_paper_text_with_provenance("T", "A", 2020, [{"text": "Full text."}])

    assert calls["n"] == len(EVIDENCE_ITEM_CAPS)
